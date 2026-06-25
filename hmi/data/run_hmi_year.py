import os
import sys
import gc
import requests
import numpy as np
import pandas as pd
import zarr
import s3fs
from astropy.io import fits
import sunpy.map
from sunpy.coordinates import frames
import reproject
from scipy.stats import skew
from tqdm import tqdm
import warnings
from sunpy.util.exceptions import SunpyMetadataWarning

warnings.filterwarnings('ignore', category=SunpyMetadataWarning)
warnings.filterwarnings('ignore')

YEAR = int(sys.argv[1])
MAX_ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else None  # optional cap, for quick validation runs

LAT_LINES = [-45, 0, 45]   # 4 latitude bands
LON_LINES = [-30, 30]      # 3 longitude bands
MIN_CH_PIXELS = 10         # blobs smaller than this are treated as segmentation noise, not a real CH


print(f"Connecting to SDO-ML HMI {YEAR} Zarr...", flush=True)
s3 = s3fs.S3FileSystem(anon=True)
AWS_ZARR = f"s3://gov-nasa-hdrl-data1/contrib/fdl-sdoml/fdl-sdoml-v2/sdomlv2_hmi.zarr/{YEAR}"
store = s3fs.S3Map(root=AWS_ZARR, s3=s3)
root = zarr.open(store=store, mode="r")
data_211 = root["Bz"]
zarr_times = pd.to_datetime(list(data_211.attrs['DATE-OBS']))
print(f"Zarr dataset shape: {data_211.shape}", flush=True)
print(f"Available time range: {zarr_times[0]} -> {zarr_times[-1]}", flush=True)

target_times = pd.date_range(
    start=f'{YEAR}-01-01 00:00:00',
    end=f'{YEAR}-12-31 18:00:00',
    freq='6H',
    tz='UTC'
)
target_times = target_times[:MAX_ROWS] if MAX_ROWS else target_times
print(f"Total target timestamps: {len(target_times)}", flush=True)

OUT_CSV = f"hmi_{YEAR}_6h_grid.csv"
CHECKPOINT_EVERY = 25

compiled_results = []
n_skipped_zarr = 0
n_skipped_404 = 0
n_exceptions = 0

if os.path.exists(OUT_CSV):
    existing_df = pd.read_csv(OUT_CSV, parse_dates=['timestamp'])
    if existing_df['timestamp'].dt.tz is None:
        existing_df['timestamp'] = existing_df['timestamp'].dt.tz_localize('UTC')
    compiled_results = existing_df.to_dict('records')
    last_ts = existing_df['timestamp'].max()
    target_times = target_times[target_times > last_ts]
    print(f"Resuming {YEAR} from checkpoint: {len(compiled_results)} rows already done, "
          f"last timestamp {last_ts}, {len(target_times)} remaining.", flush=True)

for i, target_time in enumerate(tqdm(target_times, desc=f"Processing {YEAR}")):

    row_data = {'timestamp': target_time}

    idx = np.argmin(np.abs(zarr_times - target_time))
    obs_time = zarr_times[idx]

    if abs((obs_time - target_time).total_seconds()) > 7200:
        n_skipped_zarr += 1
        continue

    header = {
        'CDELT1': data_211.attrs['CDELT1'][idx], 'CDELT2': data_211.attrs['CDELT2'][idx],
        'CRPIX1': data_211.attrs['CRPIX1'][idx], 'CRPIX2': data_211.attrs['CRPIX2'][idx],
        'CRVAL1': data_211.attrs['CRVAL1'][idx], 'CRVAL2': data_211.attrs['CRVAL2'][idx],
        'CTYPE1': 'HPLN-TAN', 'CTYPE2': 'HPLT-TAN',
        'CUNIT1': 'arcsec', 'CUNIT2': 'arcsec',
        'DATE-OBS': obs_time.tz_localize(None).isoformat(),
        'RSUN_OBS': data_211.attrs['RSUN_OBS'][idx],
        'NAXIS1': data_211.shape[2], 'NAXIS2': data_211.shape[1]
    }
    img_data = data_211[idx]
    sdoml_map = sunpy.map.Map(img_data, header)

    time_str = target_time.strftime("%Y%m%d_%H%M%S")
    fits_url = f"https://spoca.oma.be/spoca4tap/rob_spoca_ch/ch_map/{time_str}.ch_map.fits"
    temp_fits_file = f"temp_{YEAR}_{time_str}.fits"

    try:
        response = requests.get(fits_url, timeout=15)
        if response.status_code != 200:
            n_skipped_404 += 1
            continue
        with open(temp_fits_file, 'wb') as f:
            f.write(response.content)

        with fits.open(temp_fits_file) as hdu_list:
            ch_data = hdu_list[1].data
            ch_header = hdu_list[1].header

        ch_map_full_res = sunpy.map.Map(ch_data, ch_header)
        # nearest-neighbor preserves SPoCA's original integer CH IDs (no interpolation blending
        # between adjacent CHs' ID numbers, unlike the default bilinear order)
        ch_id_reprojected, _ = reproject.reproject_interp(
            ch_map_full_res, sdoml_map.wcs, shape_out=sdoml_map.data.shape, order='nearest-neighbor'
        )
        ch_id_map = np.rint(np.nan_to_num(ch_id_reprojected, nan=0.0)).astype(np.int64)

    except Exception as e:
        n_exceptions += 1
        print(f"Failed to process {fits_url}: {e}", flush=True)
        if os.path.exists(temp_fits_file):
            os.remove(temp_fits_file)
        continue
    finally:
        if os.path.exists(temp_fits_file):
            os.remove(temp_fits_file)

    hpc_coords = sunpy.map.all_coordinates_from_map(sdoml_map)
    # sunpy 2.1.0 (this env) predates SphericalScreen -- off-disk pixels just warn (silenced by
    # the blanket filterwarnings above) and come back NaN, which np.nanmean already handles.
    hgs_coords = hpc_coords.transform_to(frames.HeliographicStonyhurst)
    lat = hgs_coords.lat.to('deg').value
    lon = hgs_coords.lon.to('deg').value

    unique_ids = np.unique(ch_id_map)
    unique_ids = unique_ids[unique_ids > 0]

    cell_best = {}  # (i, j) -> (pixel_count, R2, skewness) of the largest CH assigned so far
    n_ch_kept = 0

    for cid in unique_ids:
        cid_mask = (ch_id_map == cid)
        n_pix = int(cid_mask.sum())
        if n_pix < MIN_CH_PIXELS:
            continue

        ch_pix = img_data[cid_mask]
        ch_pix = ch_pix[~np.isnan(ch_pix)]
        if len(ch_pix) == 0:
            continue

        phi_pos = np.sum(ch_pix[ch_pix > 0])
        phi_neg = np.sum(ch_pix[ch_pix < 0])
        denom = phi_pos + np.abs(phi_neg)
        if denom == 0:
            continue

        ch_r2 = 2 * np.abs(0.5 - phi_pos / denom)
        ch_skew = skew(ch_pix)
        n_ch_kept += 1

        lat_centroid = np.nanmean(lat[cid_mask])
        lon_centroid = np.nanmean(lon[cid_mask])
        grid_i = int(np.digitize(lat_centroid, LAT_LINES))
        grid_j = int(np.digitize(lon_centroid, LON_LINES))

        key = (grid_i, grid_j)
        if key not in cell_best or n_pix > cell_best[key][0]:
            cell_best[key] = (n_pix, ch_r2, ch_skew)

    for grid_i in range(4):
        for grid_j in range(3):
            entry = cell_best.get((grid_i, grid_j))
            row_data[f'R2_{grid_i + 1}_{grid_j + 1}'] = entry[1] if entry else np.nan
            row_data[f'skew_{grid_i + 1}_{grid_j + 1}'] = entry[2] if entry else np.nan
    row_data['n_ch_total'] = len(unique_ids)
    row_data['n_ch_kept'] = n_ch_kept

    compiled_results.append(row_data)

    del sdoml_map, ch_map_full_res, ch_id_reprojected, ch_id_map, hpc_coords, hgs_coords
    gc.collect()

    if (i + 1) % CHECKPOINT_EVERY == 0:
        pd.DataFrame(compiled_results).to_csv(OUT_CSV, index=False)
        print(f"[checkpoint] {i+1}/{len(target_times)} processed, {len(compiled_results)} rows so far "
              f"(skipped: zarr={n_skipped_zarr}, 404={n_skipped_404}, exceptions={n_exceptions})", flush=True)

df_results = pd.DataFrame(compiled_results)
df_results.to_csv(OUT_CSV, index=False)
print(f"\nProcessing complete. {len(compiled_results)} / {len(target_times)} records collected.", flush=True)
print(f"Skipped - no zarr match: {n_skipped_zarr}, HTTP 404: {n_skipped_404}, exceptions: {n_exceptions}", flush=True)
print(f"Saved to '{OUT_CSV}'", flush=True)
