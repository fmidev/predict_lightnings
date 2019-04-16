# predict_lightnings
Experiments on statistical predictability of lightning count in Finland.

Statistical learning methods are used to predict the daily lightning count based on ERA-Interim large scale predictors, namely T850, CAPE, R700 and MSLP.

Use the script ´run_everything.sh´ to run downloading of the data, to fit the models, and to plot the results. If you want to run only the experiments, comment out the dowloading and preprocessing part. The code will then use the previously preprocessed, default predictor and predictand data, stored in files ´predictor_and_predictand_data.csv´, ´weather_types.csv´, and daily_lightnings_1998-2018.txt´.
