# Experiments on statistical predictability of lightning count in Finland
Statistical learning methods are used to predict the daily lightning count based on ERA-Interim large scale predictors, namely T850, CAPE, R700 and MSLP.

## Dependencies
The code was developed using Python 3.6 and several external libraries, which were installed with the Miniconda installer, available [here](https://conda.io/miniconda.html).

The conda-forge repository was used to install the libraries:
`conda install -c conda-forge numpy scipy matplotlib cartopy xarray seaborn pandas scikit-learn`

## Running the experiments
Use the script `run_everything.sh` to run downloading of the data, to fit the models, and to plot the results. Before running it, make sure that the folder paths are correctly defined inside the script.

If you want to run only the experiments, comment out the dowloading and preprocessing part. The code will then use the previously preprocessed, default predictor and predictand data, stored in files `predictor_and_predictand_data.csv`, `weather_types.csv`, and `daily_lightnings_1998-2018.txt`.
