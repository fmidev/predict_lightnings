# Experiments on the statistical predictability of the lightning count in Finland
Statistical learning methods are used to predict the daily lightning count based on large scale predictors. ERA-Interim reanalysis data over the Scandinavian region were dowloaded for the key predictor parameters, namely T850, CAPE, R700 and MSLP. Each predictor parameter were decomposed into principal components, which were further modified, and then used as potential predictors in a random sampling based Poisson regression modeling system.

## Dependencies
The code was developed using Python 3.6 and several external libraries, which were installed with the Miniconda installer, available [here](https://conda.io/miniconda.html).

The conda-forge repository was used to install the libraries:
`conda install -c conda-forge numpy scipy matplotlib cartopy xarray seaborn pandas scikit-learn statsmodels`

## Running the experiments
Use the script `run_everything.sh` to run downloading of the data, to fit the models, and to plot the results. Before running it, make sure that the folder paths are correctly defined inside the script.

If you want to run only the experiments, comment out the dowloading and preprocessing part. The code will then use the previously preprocessed, default predictor and predictand data, stored in files `predictor_and_predictand_data.csv`, `weather_types.csv`, and `daily_lightnings_1998-2018.txt`.

## Interpretation of the predictor data file
The file `predictor_and_predictand_data.csv` contains predictor values for each day in 1998-2018. Predictand values, column `N_lightnings`, are also included. The naming of each predictor column follows the pattern `XXXX-YY-ZZ`, where `XXXX` are the initials of the parameter, `YY` is the running number of the principal components, and `ZZ` is the number of lags of the predictor in days. For example, `ZZ=0` means that the predictor values are from the same day than the predictand values, `ZZ=1` means that previous day values are presented, and `ZZ=2` that two days old predictor values are used.
