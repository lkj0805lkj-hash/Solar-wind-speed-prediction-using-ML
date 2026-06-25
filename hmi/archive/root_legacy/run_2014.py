import os
import gc
import requests
import numpy as np
import pandas as pd
import zarr
import s3fs
from astropy.io import fits
import sunpy.map
import reproject
from tqdm import tqdm
import warnings
from sunpy.util.exceptions import SunpyMetadataWarning

warnings.filterwarnings('ignore', category=SunpyMetadataWarning)
warnings.filterwarnings('ignore')


def compute_global_stats(pix):
    """Return (energy, entropy, variance) for an array of magnetogram pixel values."""
    energy = np.nanmean(pix ** 2)
    counts, _ = np.histogram(pix, bins=100)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    variance = np.nanvar(pix)
    return energy, entropy, variance


print("Connecting to SDO-ML HMI 2014 Zarr...", flush=True)
s3 = s3fs.S3FileSystem(anon=True)
AWS_ZARR_2014 = "s3://gov-nasa-hdrl-data1/contrib/fdl-sdoml/fdl-sdoml-v2/sdomlv2_hmi.zarr/2014"
store = s3fs.S3Map(root=AWS_ZARR_2014, s3=s3)
root = zarr.open(store=store, mode="r")
data_211 = root["Bz"]
zarr_times = pd.to_datetime(list(data_211.attrs['DATE-OBS']))
print(f"Zarr dataset shape: {data_211.shape}", flush=True)
print(f"Available time range: {zarr_times[0]} -> {zarr_times[-1]}", flush=True)

target_times = pd.date_range(
    start='2014-01-01 00:00:00',
    end='2014-12-31 18:00:00',
    freq='6H',
    tz='UTC'
)
print(f"Total target timestamps: {len(target_times)}", flush=True)

OUT_CSV = "hmi_2014_6h.csv"
CHECKPOINT_EVERY = 25

compiled_results = []
n_skipped_zarr = 0
n_skipped_404 = 0
n_exceptions = 0

for i, target_time in enumerate(tqdm(target_times, desc="Processing 2014")):

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
    temp_fits_file = f"temp_{time_str}.fits"

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
        ch_data_reprojected, _ = reproject.reproject_interp(
            ch_map_full_res, sdoml_map.wcs, shape_out=sdoml_map.data.shape
        )
        binary_mask = (ch_data_reprojected > 0.5).astype(np.uint8)

    except Exception as e:
        n_exceptions += 1
        print(f"Failed to process {fits_url}: {e}", flush=True)
        if os.path.exists(temp_fits_file):
            os.remove(temp_fits_file)
        continue
    finally:
        if os.path.exists(temp_fits_file):
            os.remove(temp_fits_file)

    ch_pix = img_data[binary_mask.astype(bool)]
    ch_pix = ch_pix[~np.isnan(ch_pix)]

    if len(ch_pix) > 0:
        phi_pos = np.sum(ch_pix[ch_pix > 0])
        phi_neg = np.sum(ch_pix[ch_pix < 0])
        denom = phi_pos + np.abs(phi_neg)
        row_data['global_R2'] = 2 * np.abs(0.5 - phi_pos / denom) if denom > 0 else np.nan
        row_data['global_energy'], row_data['global_entropy'], row_data['global_variance'] = compute_global_stats(ch_pix)
    else:
        coords = sunpy.map.all_coordinates_from_map(sdoml_map)
        r_dist = np.sqrt(coords.Tx**2 + coords.Ty**2)
        disk_mask = r_dist < (sdoml_map.rsun_obs * 0.99)
        disk_pix = img_data[disk_mask]
        disk_pix = disk_pix[~np.isnan(disk_pix)]

        row_data['global_R2'] = 1
        if len(disk_pix) > 0:
            row_data['global_energy'], row_data['global_entropy'], row_data['global_variance'] = compute_global_stats(disk_pix)
        else:
            row_data['global_energy'] = np.nan
            row_data['global_entropy'] = np.nan
            row_data['global_variance'] = np.nan

    compiled_results.append(row_data)

    del sdoml_map, ch_map_full_res, ch_data_reprojected, binary_mask
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
