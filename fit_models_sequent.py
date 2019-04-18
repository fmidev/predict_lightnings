#!/usr/bin/env python





# Read modules
import sys, ast, imp, glob
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.utils import resample
from sklearn.model_selection import ShuffleSplit

import statsmodels.api as sm


# Directories from command line arguments
codir, indir, oudir = str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])

# Read own functions
sys.path.append(codir)
import functions as fcts



# Read data from the namelist file
exp_name,years,test_yrs,train_yrs,params,estim, \
    n_estim,n_pcs,lags,p_feat,p_smpl = fcts.read_namelist(codir+'namelist.csv')


print('Fitting models for',exp_name)  






# Read previously processed input data 
data = pd.read_csv(codir+'predictor_and_predictand_data.csv', index_col='Date', parse_dates=[0])

# Divide data to predictors and predictands
X = data.drop(columns=['N_lightnings', 'TS_day'])
Y = data['N_lightnings']

# Summer months only
months = data.index.month.unique()


# Uncomment and run the following lines if you want to test regression estimators
"""
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.discrete.discrete_model import NegativeBinomial

model = sm.GLM(Y, X, family=sm.families.Poisson(link=sm.genmod.families.links.log())).fit()
#model = sm.GLM(Y, X, family=sm.families.NegativeBinomial()).fit_regularized(method='elastic_net')
model = sm.GLM(Y, X, family=sm.families.NegativeBinomial(alpha=1e-03)).fit()
#model = sm.GLM(Y, X, family=sm.families.Tweedie()).fit()
#model = NegativeBinomial(Y, X).fit(method='minimize',maxiter=200)



print(model.summary())

#mod = model.predict(X); mod.loc[mod>60000]=0
df = pd.merge(Y.rename('Obs').to_frame(), pd.DataFrame(mod, columns=['Mod']), left_on='Date', right_on='Date')
g = sns.jointplot('Obs', 'Mod', data=df, kind="reg",color="m", height=7, ratio=2, )

plt.tight_layout(); plt.show()
"""









# Potential base-estimators
estimators =    {  
                'poisson':sm.families.Poisson(),
                'tweedie':sm.families.Tweedie(),
                'negbinm':sm.families.NegativeBinomial(alpha=5e-04),
                }

# Initialize output data container
columns = ['N of PCs', 
           'Train RMSE', 'Validation RMSE', 'Test RMSE', 
           'Train CORR', 'Validation CORR', 'Test CORR', 
           'Fitted model','Variable sample', 'Best model'] 

models_out = pd.DataFrame(columns=columns)


# Indices defining the fitting and testing periods
fit_idx = fcts.bool_index_to_int_index(np.isin(data.index.month, months.astype(int)) & 
                                       np.isin(data.index.year, train_yrs))
tst_idx = fcts.bool_index_to_int_index(np.isin(data.index.month, months.astype(int)) & 
                                       np.isin(data.index.year, test_yrs))

# Data arrays containing fitting and testing data
X_fit = X.ix[fit_idx]; Y_fit = Y.ix[fit_idx]
X_tst = X.ix[tst_idx]; Y_tst = Y.ix[tst_idx]





# Fit models. Add more PCs one by one and resample input data 
# randomly in each step

variable_sample = np.array([])
np.random.seed(5)
for pc in range(1,n_pcs+1):
    
    # A subset of variables to be included only!
    for i,vr in enumerate(X.keys()):
        
        param_name, cpn, lag = vr.rsplit('-')
        param_names_joined = ''.join(params)
        
        itis_param = param_names_joined.find(param_name) >= 0
        itis_pc    = int(cpn) == pc 
        itis_lag   = np.isin(int(lag), lags)
        
        if itis_param and itis_pc and itis_lag:
            variable_sample = np.append(variable_sample, vr)
    
    variable_sample = np.unique(variable_sample)
    
    n_feat = int(len(variable_sample)*p_feat)
    #n_feat = int(np.sqrt(len(variable_sample))) 
    
    ss  = ShuffleSplit(n_splits=n_estim, train_size=p_smpl, random_state=99)
    
    mdls = pd.DataFrame(columns=columns)
    for trn_idx, vld_idx in ss.split(X_fit):
        
        X_trn, X_vld = X_fit.ix[trn_idx], X_fit.ix[vld_idx]
        Y_trn, Y_vld = Y_fit.ix[trn_idx], Y_fit.ix[vld_idx]
        
        variable_sample_reduced = resample(variable_sample, replace=False, n_samples=n_feat)
        print(variable_sample_reduced) 
        #print((variable_sample_reduced.shape[0]/variable_sample.shape[0])*100., variable_sample.shape, variable_sample_reduced.shape)
        
        try:
            fitted_model = sm.GLM(Y_trn, X_trn[variable_sample_reduced], family=estimators[estim]).fit()
            train_rmse = fcts.calc_rmse(fitted_model.predict(X_trn[variable_sample_reduced]), Y_trn)
            train_corr = fcts.calc_corr(fitted_model.predict(X_trn[variable_sample_reduced]), Y_trn)
            vldt_rmse  = fcts.calc_rmse(fitted_model.predict(X_vld[variable_sample_reduced]), Y_vld)
            vldt_corr  = fcts.calc_corr(fitted_model.predict(X_vld[variable_sample_reduced]), Y_vld)
            test_rmse  = fcts.calc_rmse(fitted_model.predict(X_tst[variable_sample_reduced]), Y_tst)
            test_corr  = fcts.calc_corr(fitted_model.predict(X_tst[variable_sample_reduced]), Y_tst)
            
            # Discard too large models to save space, save only their validation results
            if(pc >= 8): fitted_model = []
            
            df = pd.DataFrame(np.array([[int(pc), train_rmse, vldt_rmse, test_rmse,
                                                    train_corr, vldt_corr, test_corr, fitted_model, 
                                                    variable_sample_reduced, False]]), columns=columns)
            
            mdls = mdls.append(df, ignore_index=True)
        except: 
            pass
    
    best_mdl = mdls['Validation CORR'].values.argmax(); mdls['Best model'][best_mdl] = True   
    models_out = models_out.append(mdls, ignore_index=True)






# Save results into a Pandas pickle object
models_out.to_pickle(oudir+'fitted_models_'+exp_name+'.pkl'); 

print('Finished fitting models!')






