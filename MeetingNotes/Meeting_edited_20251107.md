# Meeting with Kaijie, Farzad, 2025-11-07

# Presentation on predicting geomagnetic storms

pptx file: geomagnetic_storm_prediction_deep_learning20251107.pptx

## Introduction

Different place in the heliopshere  where in-situ solar wind measurements are taken (Parker Solar probe, Solar orbiter, stereo, bepi columbo, etc,...), but for operational purpose, we need a fixed place in time, and this is given by the satellite which are at L1 : ACE, DISCOVR, and which are merged in the OMNIWEB dataset

## Goal of the project:

- Estimate of solar wind speed
- Estimate of Kp with a large forecast horizon (a few days). See how you can relate what is at the Sun, with what you observe at L1, knowing that many processes will happen in between, that you won't be able to model here

\-> forecast the large trend

## Empirical model of solar wind velocity using coronal holes areas

Prediction of solar wind speed at L1, as a function of the fractional CH area observed four days before

- Rotter, T., Veronig, A.M., Temmer, M. et al. Real-Time Solar Wind Prediction Based on SDO/AIA Coronal Hole Data. Sol Phys 290, 1355–1370 (2015).  
  [link](https://doi.org/10.1007/s11207-015-0680-5)
- Reiss, M. A., M. Temmer, A. M. Veronig, L. Nikolic, S. Vennerstrom, F. Schöngassner, and S. J. Hofmeister (2016), Verification of high-speed solar wind stream forecasts using operational solar wind models, Space Weather, 14, 495–510, doi:10.1002/2016SW001390.

In Rotter et al (2015), the idea is to relate the area of meridional coronal holes to solar wind speed measured at L1, about 3 days later via  a linear regression. Resulting R2 are poor.

To improve, they considered to include magnetic field information, and gradient boosting:

- Bailey, R. L., Reiss, M. A., Arge, C. N., Möstl, C., Henney, C. J., Owens, M. J., et al. (2021). Using gradient boosting regression to improve ambient solar wind model predictions. Space Weather, 19, e2020SW002673. [link](https://doi.org/10.1029/2020SW002673)

## Empirical model to forecast solar wind speed using deep learning architecture.

More complex model that take as input one (or a time  series of) AIA images.

- Upendran, V., Cheung, M. C. M., Hanasoge, S., & Krishnamurthi, G. (2020). Solar wind prediction using deep learning. Space Weather, 18, e2020SW002478. [link](https://doi.org/10.1029/2020SW002478)  : Each pair of images (193A, 211 A) goes to a feature extractor, and a time series of feature extractor is inputted into LSTM
- Brown, E. J. E., Svoboda, F., Meredith, N. P., Lane, N., & Horne, R. B. (2022). Attention-based machine vision models and techniques for solar wind speed forecasting using solar EUV images. Space Weather, 20, e2021SW002976. [link](https://doi.org/10.1029/2021SW002976)  Q: does it use time series ?
- Raju, H., Das, S. CNN-Based Deep Learning Model for Solar Wind Forecasting. Sol Phys 296, 134 (2021). [link](https://doi.org/10.1007/s11207-021-01874-6)

## Kp probalistic forecast from EUV images

- Bernoux, G., Brunet, A., Buchlin, É., Janvier, M., & Sicard, A. (2022). Forecasting the geomagnetic activity several days in advance using neural networks driven by solar EUV imaging. Journal of Geophysical Research: Space Physics, 127, e2022JA030868. [link](https://doi.org/10.1029/2022JA030868)

Note:

1. Same model can be used to forecast solar wind speed
2. More efficient feature extraction method than using pre-trained model are studied in:

- Tahtouh, M., Bernoux, G., Brunet, A., Standarovski, D., Nguyen, G., & Sicard, A. (2025). Comparison of solar imaging feature extraction methods in the context of space weather prediction with deep learning-based models. Journal of Geophysical Research: Machine Learning and Computation, 2, e2024JH000566.  [link](https://doi.org/10.1029/2024JH000566)


* dataset curation is important. : ML-ready dataset, as extended as possible. (cfr presentation Guillerme at ESWW)

## (short-term) Forecast of Kp index solar wind measurement at L1,

- Zhelavskaya, R. Vasile, Y. Y. Shprits, C. Stolle, and J. Matzka. Systematic Analysis of Machine Learning and Feature Selection Techniques for Prediction of the Kp Index. Space Weather, 17(10): 1461–1486, October 2019. doi: 10.1029/2019SW002271 [link](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019SW002271)
- Yao Tan, Qinghua Hu, Zhen Wang, and Qiuzhen Zhong. Geomagnetic Index Kp Forecasting With LSTM. Space Weather, 16(4):406–416, April 2018.
  [link](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1002/2017SW001764)
- Peter Wintoft, Magnus Wik, Jürgen Matzka, and Yuri Shprits. Forecasting Kp from solar wind data: input parameter study using 3-hour averages and 3-hour range values. Journal of Space Weather and Space Climate, 7:A29, November 2017.
  [link](https://www.swsc-journal.org/component/article?access=doi&doi=10.1051/swsc/2017027)