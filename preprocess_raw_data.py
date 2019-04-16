#!/usr/bin/env python



# Read modules
import sys, glob, ast
import numpy as np
import pandas as pd

# Directories from command line arguments
codir, indir, oudir = str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])

# Read own functions
sys.path.append(codir)
import functions as fcts



# Read data from the namelist file
exp_name,years,test_yrs,train_yrs,_,estim, \
    n_estim,n_pcs,lags,p_feat,p_smpl = fcts.read_namelist(codir+'namelist.csv')


print('Preprocessing data for',exp_name)  



# Read lightning data
lightnings = pd.read_csv(codir+'daily_lightnings_1998-2018.txt', index_col='Date',parse_dates=[0])


# Input data info container
eraint_data_info =  {
    'PMSL': ['MSL',     np.sort(glob.glob(indir+'pmsl*nc'))],
    'T850': ['T',       np.sort(glob.glob(indir+'t850*nc'))],
    'R700': ['R',       np.sort(glob.glob(indir+'rh70*nc'))],
    'CAPE': ['var59',   np.sort(glob.glob(indir+'cape*nc'))],
                    }    



# Parameters to be read for all potential predictors 
all_params = ('PMSL', 'T850', 'R700', 'CAPE')

X_vars = []
for i, par_name in enumerate(all_params):
    print('Reading raw data for',par_name)
    
    # Read ERA-Interim data for Scandinavian summer months
    # Resample to daily time resolution, but do not remove trends, climatology etc. 
    param = fcts.read_and_select(   eraint_data_info[par_name][1], 
                                    eraint_data_info[par_name][0], 
                                    (1979, 2018), '1D', 'scandi', False)
    
    param = param.rename({eraint_data_info[par_name][0]: par_name})
    X_vars.append(param[par_name])





# Extract PCs, lag them, and organize everyhting into a Pandas DataFrame
X, _ = fcts.prepare_X_array(lightnings['N_lightnings'], X_vars, -99, 
                n_pcs, train_yrs, years, lags, include_persistence=False)


# Merge predictor and predictand data, exclude fall, winter, and spring months
data = pd.merge(X, lightnings, left_on='Date', right_on='Date', how='inner').sort_index() 
include_months = (data.index.month==6)|(data.index.month==7)|(data.index.month==8) 
data = data.iloc[include_months]



# Calculate and save weather types separately just in case
Z, _ = fcts.calc_weather_types(X_vars, 7, 15, (0,))
include_months = (Z.index.month==6)|(Z.index.month==7)|(Z.index.month==8) 
Z = Z[include_months]




# Save results into csv files
data.to_csv(codir+'predictor_and_predictand_data.csv')
Z.to_csv(codir+'weather_types.csv')

print('Finished preprocessing!')




