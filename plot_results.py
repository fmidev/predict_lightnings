#!/usr/bin/env python




import sys, imp, ast, glob
import numpy as np
import pandas as pd
import xarray as xr


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns; sns.set(style="white")
import cartopy; cartopy.config['data_dir'] = '/lustre/tmp/kamarain/cartopy/'
import cartopy.crs as ccrs
from sklearn.preprocessing import StandardScaler


# Directories from command line arguments
codir, indir, oudir = str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])

# Read own functions
sys.path.append(codir)
import functions as fcts



# Read data from the namelist file
exp_name,years,test_yrs,train_yrs,params,estim, \
    n_estim,n_pcs,lags,p_feat,p_smpl = fcts.read_namelist(codir+'namelist.csv')


print('Plotting results for',exp_name)  


# Read previously processed input data 
data = pd.read_csv(codir+'predictor_and_predictand_data.csv', index_col='Date', parse_dates=[0])

# Divide data to predictors and predictands
X = data.drop(columns=['N_lightnings', 'TS_day'])
Y = data['N_lightnings']


# Read previously calculated weather types
wtypes = pd.read_csv(codir+'weather_types.csv', index_col='time', parse_dates=[0])
wtypes.index.names = ['Date']



# Read fitted models
results = pd.read_pickle(oudir+'fitted_models_'+exp_name+'.pkl')
numeric_columns = ["N of PCs", "Test RMSE", "Validation RMSE", "Train RMSE", "Test CORR", "Validation CORR", "Train CORR", "N of PCs"] 
results[numeric_columns] = results[numeric_columns].apply(pd.to_numeric)



# Use only best models
#tgr_idx = results['Best model']; results = results[tgr_idx]


# Find optimal N of PCs
opt_n_pcs = results.groupby('N of PCs')['Validation CORR'].median().idxmax()
opt_idx = results['N of PCs'] == opt_n_pcs; opt_results = results[opt_idx]



months = data.index.month.unique()
all_idx = fcts.bool_index_to_int_index(np.isin(Y.index.month, months.astype(int)) & np.isin(Y.index.year, years))
trn_idx = fcts.bool_index_to_int_index(np.isin(Y.index.month, months.astype(int)) & np.isin(Y.index.year, train_yrs))
tst_idx = fcts.bool_index_to_int_index(np.isin(Y.index.month, months.astype(int)) & np.isin(Y.index.year, test_yrs))

forecast = np.full((Y.shape[0], opt_results.shape[0]), np.nan)
ens_corr = np.full((3, opt_results.shape[0]), np.nan)

j = 0; 
for m, estimator in opt_results.iterrows(): 
    mdl = estimator['Fitted model']
    vrs = estimator['Variable sample']
    raw = mdl.predict(X[vrs].ix[all_idx])
    forecast[all_idx,j] = raw
    forecast[all_idx,j] = fcts.q_mapping(Y[trn_idx], forecast[trn_idx,j], forecast[all_idx,j], 100)
    
    if(raw.max() < 10e6):
        ens_corr[0,j] = fcts.calc_corr(np.nanmedian(forecast[trn_idx], axis=1), Y[trn_idx])
        ens_corr[1,j] = fcts.calc_corr(np.nanmedian(forecast[tst_idx], axis=1), Y[tst_idx])
        ens_corr[2,j] = fcts.calc_corr(np.nanmedian(forecast[all_idx], axis=1), Y[all_idx])
        tst_corr = fcts.calc_corr(forecast[tst_idx,j], Y[tst_idx])
        print(estimator['N of PCs'],j+1, estimator['Test CORR'], 
                estimator['Validation CORR'], estimator['Train CORR'], ens_corr[1,j]) #, estimator['Best model'])
        j += 1


ens_corr_df = pd.DataFrame(ens_corr.T, index=np.arange(ens_corr.shape[1])+1, 
    columns=['Training period','Testing period','All years'])
ens_corr_df.index.names = ['Ensemble size']



p_95 = np.nanpercentile(forecast, q=[2.5, 97.5], axis=1)
comparison = pd.DataFrame(Y); 
comparison['Regression'] = np.nanmedian(forecast, axis=1)
comparison['Regression'][comparison['Regression'] < 0] = 0
comparison = pd.merge(comparison, wtypes['Weather type'], left_on='Date', right_on='Date', how='inner') 
comparison['p_2.5'] = p_95[0]; comparison['p_97.5'] = p_95[1]

comparison['Period'] = ''; comparison['Period'][trn_idx] = 'Train'; comparison['Period'][tst_idx] = 'Test'









sns.set(style="whitegrid")
nrow = 1; ncol = 2
fig, ax = plt.subplots(nrow,ncol, figsize=(4*ncol,4*nrow))

c = ('c', 'b', 'r', 'g', 'm', 'Orange', 'gray')

sns.lineplot(x='N of PCs',y='Train CORR',color='b', data=results, estimator='median', ci='sd', label='Training sample', ax=ax[0]) 
sns.lineplot(x='N of PCs',y='Validation CORR',color='r', data=results, estimator='median', ci='sd', label='Validation sample', ax=ax[0])
sns.lineplot(x='N of PCs',y='Test CORR',color='Orange', data=results, estimator='median', ci='sd', label='Testing period', ax=ax[0])

sns.lineplot(x=ens_corr_df.index,y='Training period',color='b', data=ens_corr_df,  label='Training period', ax=ax[1]) 
sns.lineplot(x=ens_corr_df.index,y='Testing period',color='Orange', data=ens_corr_df,  label='Testing period', ax=ax[1]) 
sns.lineplot(x=ens_corr_df.index,y='All years',color='g', data=ens_corr_df, label='All years', ax=ax[1]) 

ax[0].set_ylabel('Correlation'); ax[1].set_ylabel('Correlation'); #ax[0].legend()
ax[0].set_title('Correlation skill of\nindividual ensemble members')
ax[1].set_title('Correlation skill of\nthe median of the ensemble');
#ax[0].xaxis.set_major_locator(ticker.MultipleLocator(5))


plt.tight_layout()
fig.savefig(oudir+'fig_goodnessoffit_'+exp_name+'.png'); plt.close()




df1 = pd.DataFrame(comparison[['N_lightnings', 'Weather type']]).assign(Data='Observed')
df2 = pd.DataFrame(comparison[['Regression', 'Weather type']]).assign(Data='Modeled')
df2 = df2.rename(columns={'Regression':'N_lightnings'})
cdf = pd.concat([df1, df2])


wt_medians = cdf[cdf['Data']=='Observed'].groupby('Weather type').median()['N_lightnings']
hierarchy = wt_medians.sort_values().index

sns.set(style="whitegrid")
fig, axes = plt.subplots(1,1, figsize=(hierarchy.max()*0.5,4),subplot_kw={'yscale':'symlog'}) 

sns.boxenplot(data=cdf, x='Weather type', y='N_lightnings', palette="colorblind", width=0.70,
                ax=axes, hue='Data', order=hierarchy)

axes.set_ylabel('')
axes.set_title('Number of lightnings in weather type categories')
plt.tight_layout()
fig.savefig(oudir+'fig_wtypes_'+exp_name+'.png'); plt.close()






fig, axes = plt.subplots(1, 1, figsize=(4,4)) 

axes.scatter(Y[trn_idx], np.nanmean(forecast[trn_idx], axis=1), c='b', label='Training')
axes.scatter(Y[tst_idx], np.nanmean(forecast[tst_idx], axis=1), c='Orange', label='Testing')

#axes.set_xscale("symlog"); axes.set_yscale("symlog")

axes.legend(); plt.tight_layout(); 
fig.savefig(oudir+'fig_scatter_'+exp_name+'.png'); plt.close()







df = comparison[['N_lightnings', 'Regression']] #[comparison['Period']=='Train']
sns.jointplot('N_lightnings', 'Regression', data=df, kind="reg",color="m", height=7, ratio=5)
#axes.set_xscale("symlog"); axes.set_yscale("symlog")
plt.tight_layout(); 
plt.savefig(oudir+'fig_scatter_reg_'+exp_name+'.png')
plt.show(); plt.close()










sns.set_style('whitegrid') #("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
fig, ax = plt.subplots(len(months), 1, figsize=(len(months)*3, 5)) 

for i,mon in enumerate(months):
    
    all_idx = fcts.bool_index_to_int_index( np.isin(comparison.index.month, int(mon)) & 
                                            np.isin(comparison.index.year, years))
    tst_idx = fcts.bool_index_to_int_index( np.isin(comparison.index.month, int(mon)) & 
                                            np.isin(comparison.index.year, test_yrs))
    
    x_vals = np.arange(comparison.ix[tst_idx].shape[0])
    x_lbls = comparison.ix[tst_idx].index.astype(str)
    
    ax[i].plot(x_vals, comparison.ix[tst_idx]['N_lightnings'].values, color='k')
    ax[i].plot(x_vals, comparison.ix[tst_idx]['Regression'].values, color='r')
    ax[i].fill_between(x=x_vals,    y1=comparison.ix[tst_idx]['p_2.5'].values, 
                                    y2=comparison.ix[tst_idx]['p_97.5'].values, 
                                    color="r", alpha=0.2, label='Ensemble 95% range')
    
    corr = fcts.calc_corr(  comparison.ix[tst_idx]['N_lightnings'].values, 
                            comparison.ix[tst_idx]['Regression'].values)
    
    ax[i].set_yscale('symlog')
    ax[i].set_xticks(x_vals[0::10])
    ax[i].set_xticklabels(x_lbls[0::10], rotation=15, fontsize=7)
    ax[i].set_title('Mon: '+str(mon)+', Correlation: '+str(round(corr,2)))
    ax[i].set_ylabel('Daily $\sum$')

plt.tight_layout(); 
fig.savefig(oudir+'fig_timeser_day_'+exp_name+'.png'); plt.close()






monsum = comparison.resample('1M').sum()

    
all_idx = fcts.bool_index_to_int_index(np.isin(monsum.index.month, months.astype(int)) & np.isin(monsum.index.year, years))
trn_idx = fcts.bool_index_to_int_index(np.isin(monsum.index.month, months.astype(int)) & np.isin(monsum.index.year, train_yrs))
tst_idx = fcts.bool_index_to_int_index(np.isin(monsum.index.month, months.astype(int)) & np.isin(monsum.index.year, test_yrs))

monsum['Period'] = ''; monsum['Period'][trn_idx] = 'Train'; monsum['Period'][tst_idx] = 'Test'
#monsum = monsum.ix[all_idx]

fig, ax = plt.subplots(len(months), 1, figsize=(len(months)*3, 5)) 

for i,mon in enumerate(months):
    
    all_idx = fcts.bool_index_to_int_index(np.isin(monsum.index.month, int(mon)) & np.isin(monsum.index.year, years))
    #trn_idx = fcts.bool_index_to_int_index(np.isin(monsum.index.month, mon.astype(int)) & np.isin(monsum.index.year, train_yrs))
    #tst_idx = fcts.bool_index_to_int_index(np.isin(monsum.index.month, mon.astype(int)) & np.isin(monsum.index.year, test_yrs))
    
    g=sns.lineplot(x=monsum.ix[all_idx].index.date, y='N_lightnings', data=monsum.ix[all_idx], color='k',markers='o', ax=ax.ravel()[i])
    h=sns.lineplot(x=monsum.ix[all_idx].index.date, y='Regression',  data=monsum.ix[all_idx], color='red',markers='o', ax=ax.ravel()[i])
    ax.ravel()[i].axvline(x=monsum.ix[tst_idx].index.date[0]) 
    
    #g.set_xticklabels(g.get_xticklabels(), rotation=90); 
    g.set_title('Mon: '+str(mon)); g.set_ylabel('Monthly $\sum$')
    #ax.ravel()[i].xaxis.set_major_locator(ticker.MultipleLocator(5))

plt.tight_layout(); 
fig.savefig(oudir+'fig_timeser_mon_'+exp_name+'.png'); plt.close()








yrsum = comparison.resample('1Y').sum()

    
all_idx = fcts.bool_index_to_int_index(np.isin(yrsum.index.year, years))
trn_idx = fcts.bool_index_to_int_index(np.isin(yrsum.index.year, train_yrs))
tst_idx = fcts.bool_index_to_int_index(np.isin(yrsum.index.year, test_yrs))

yrsum['Period'] = ''; yrsum['Period'][trn_idx] = 'Train'; yrsum['Period'][tst_idx] = 'Test'

fig, ax = plt.subplots(1, 1, figsize=(8,3)) 


g=sns.lineplot(x=yrsum.ix[all_idx].index.date, y='N_lightnings', data=yrsum.ix[all_idx], color='k',markers='o', ax=ax)
h=sns.lineplot(x=yrsum.ix[all_idx].index.date, y='Regression',  data=yrsum.ix[all_idx], color='red',markers='o', ax=ax)
ax.axvline(x=yrsum.ix[tst_idx].index.date[0]) 

g.set_title(''); g.set_ylabel('Annual $\sum$')


plt.tight_layout(); 
fig.savefig(oudir+'fig_timeser_years_l'+exp_name+'.png'); plt.close()










# Input data info container
eraint_data_info =  {
    'PMSL': ['MSL',     np.sort(glob.glob(indir+'pmsl*nc'))],
    'T850': ['T',       np.sort(glob.glob(indir+'t850*nc'))],
    'R700': ['R',       np.sort(glob.glob(indir+'rh70*nc'))],
    'CAPE': ['var59',   np.sort(glob.glob(indir+'cape*nc'))],
                    }    

# Read raw data for Europe (not only to Scandinavia) to get a better picture
# about synoptics of the weather types

all_params = ('PMSL', 'T850', 'R700', 'CAPE')
X_vars = []; WT_means = []; wtypes.index.names = ['time']
for i, par_name in enumerate(all_params):
    print('Reading raw data for',par_name)
    
    # Read ERA-Interim data for summer months
    # Resample to daily time resolution, but do not remove trends, climatology etc.
    param = fcts.read_and_select(   eraint_data_info[par_name][1], 
                                    eraint_data_info[par_name][0], 
                                    (1998, 2018), '1D', 'scandi', False)
    
    param = param.rename({eraint_data_info[par_name][0]: par_name})
    param['Weather type'] = wtypes['Weather type']
    
    param_mean = param.groupby('Weather type').mean('time')[par_name].load().to_dataset()
    if(par_name=='PMSL'): param_mean[par_name] = param_mean[par_name]/100.
    if(par_name=='T850'): param_mean[par_name] = param_mean[par_name]-273.15
    
    X_vars.append(param)
    WT_means.append(param_mean)









lightnings_medians = comparison.groupby('Weather type')['N_lightnings'].median()
upper_limit = np.sort(lightnings_medians)[-4]
lower_limit = np.sort(lightnings_medians)[3]

upper_types = lightnings_medians.index[lightnings_medians > upper_limit]
lower_types = lightnings_medians.index[lightnings_medians < lower_limit]


upper_lower_types = {'upper':upper_types, 'lower':lower_types}

for wtype in upper_lower_types:
    nrow = len(upper_lower_types[wtype]) #wtypes['Weather type'].max(); 
    ncol = 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*2.5,nrow*2.5), 
        subplot_kw={'projection': ccrs.Orthographic(central_longitude=18, central_latitude=60, globe=None)}) #ccrs.PlateCarree()}) 
    all_params = ('PMSL', 'T850', 'R700', 'CAPE')
    for i,wtp in enumerate(upper_lower_types[wtype]):
        
        
        idx = WT_means[0]['Weather type']==wtp
        
        #print(i,wtp,idx.values,param_mean['T'][idx])
        ctp_t, cnt_t = fcts.plot_contours(WT_means[1]['T850'][idx].squeeze(),   ccrs.PlateCarree(), 
               axes[i,0], fcts.scientific_colormaps('batlow'),      np.linspace(0,12,21), np.linspace(0,12,11), None)
        
        ctp_m, cnt_m = fcts.plot_contours(WT_means[0]['PMSL'][idx].squeeze(), ccrs.PlateCarree(), 
               axes[i,1], 'YlGnBu_r', np.linspace(1006,1020,21), np.linspace(1006,1020,11), None)
                                        
        ctp_r, cnt_r = fcts.plot_contours(WT_means[2]['R700'][idx].squeeze(),   ccrs.PlateCarree(), 
               axes[i,2], 'bone',   np.linspace(35,75,11), np.linspace(35,75,6), None)
 
        ctp_c, cnt_c = fcts.plot_contours(WT_means[3]['CAPE'][idx].squeeze(),   ccrs.PlateCarree(), 
               axes[i,3], fcts.scientific_colormaps('acton'),   np.linspace(0,350,11), np.linspace(0,350,6), None)
        
        axes[i,0].coastlines(alpha=0.5); axes[i,1].coastlines(alpha=0.5); 
        axes[i,2].coastlines(alpha=0.5); axes[i,3].coastlines(alpha=0.5)
        
        axes[i,0].set_title('WT'+str(wtp)+', T850 ($\degree$C)')
        axes[i,1].set_title('WT'+str(wtp)+', PMSL (hPa)')
        axes[i,2].set_title('WT'+str(wtp)+', R700 (%)')
        axes[i,3].set_title('WT'+str(wtp)+', CAPE (J kg$^{-1}$)')
    
    
    plt.tight_layout(); 
    fig.savefig(oudir+'fig_'+wtype+'_wtype_param_means_'+exp_name+'.png',dpi=100); plt.close()




"""
skw_data = np.random.gamma(np.random.standard_normal(5000)))
nrl_data = fcts.make_data_normal_distributed(skw_data)

plt.hist(skw_data,50)
plt.hist(nrl_data,50);plt.show()
"""






# Plot PCs and EOFs 

var = X_vars[3]
    
nme = var.name #data['y_var'] 
ncomps = opt_n_pcs #n_pcas

cps_full, pca_full, svl = fcts.apply_PCA(var.values, ncomps)
cps_full[0] = cps_full[1]
cps_full = StandardScaler().fit_transform(cps_full)

patterns = StandardScaler().fit_transform(pca_full.components_.T).T

print('Tot evr:', np.sum(pca_full.explained_variance_ratio_*100))
pattern_ds = xr.full_like(var[0:ncomps].load(),   np.nan).rename({'time': 'Comp'})
compnnt_ds = xr.full_like(var[:,0:ncomps].load(), np.nan).rename({'gridcell': 'Comp'})
pattern_ds['Comp'] = np.arange(1,ncomps+1)
compnnt_ds['Comp'] = np.arange(1,ncomps+1)

sns.set(style="white", font_scale=1.0) 
fig, axes = plt.subplots(ncomps, 2, figsize=(7,ncomps*1.5), gridspec_kw={'width_ratios':[2,1]})

levels = [-2, -1, -0.5, 0, 0.5, 1, 2]   

for j,cp in enumerate(np.arange(ncomps)):
    
    pattern = patterns[j] 
    compnnt = cps_full[:,j] 
    evr = pca_full.explained_variance_ratio_[j]*100
    
    pattern_ds[j]   = np.squeeze(pattern)
    compnnt_ds[:,j] = np.squeeze(compnnt)
    #
    
    fgrd = pattern_ds[j].unstack('gridcell').plot.contourf(ax=axes[j,1], 
            levels=levels,center=0, add_colorbar=True, cbar_kwargs={'label': ''})
    
    tser = compnnt_ds[:,j].to_dataframe().drop(columns=['Comp']).plot(ax=axes[j,0], legend=False)
    #
    
    axes[j,0].set_title(nme+' PC'+str(j+1)+', expl. variance: '+str(evr)[0:4]+'%'); 
    axes[j,1].set_title(''); axes[j,1].set_xlabel(''); axes[j,1].set_ylabel(''); 
    axes[j,0].set_xlabel(''); axes[j,0].set_ylabel(''); axes[j,0].set_ylim([-3.1, 3.1])


plt.tight_layout(); plt.show()








# Plot time series graphs 


sns.set(style="whitegrid")


n_models = 100
observations = Y.values

fig, axes = plt.subplots(len(months), 1, figsize=(7,2.2*len(months))) 
fig, axes = plt.subplots(1, 1, figsize=(4,4)) 
#for i,mon in enumerate(months): #enumerate(data['seasons']): 

#tgr_idx = (results['Month']==str(mon)) & (results['Best model']) 
tgr_idx = results['Best model']

all_idx = fcts.bool_index_to_int_index(np.isin(Y.index.month, months.astype(int)) & np.isin(Y.index.year, years))
trn_idx = fcts.bool_index_to_int_index(np.isin(Y.index.month, months.astype(int)) & np.isin(Y.index.year, train_yrs))
tst_idx = fcts.bool_index_to_int_index(np.isin(Y.index.month, months.astype(int)) & np.isin(Y.index.year, test_yrs))

#forecast = np.full((Y.shape[0], results[tgr_idx].shape[0]*n_models), np.nan)
forecast = np.full((Y.shape[0], results[tgr_idx].shape[0]), np.nan)

j = 0; 
for m, model in results[tgr_idx].iterrows(): 
    mdl = model['Fitted model']
    vrs = model['Variable sample']
    raw = mdl.predict(X[vrs].ix[all_idx])
    forecast[all_idx,j] = raw
    forecast[all_idx,j] = fcts.q_mapping(observations[trn_idx], forecast[trn_idx,j], forecast[all_idx,j], 100)
    
    print(mon, model['N of PCs'], model['Test CORR'], model['Validation CORR'], model['Train CORR'], model['Best model'])
    j += 1

ens_size = forecast.shape[1]
print('Fcast shape',forecast.shape)
persistence = np.full(observations.shape[0], np.nan)

persistence[all_idx] = fcts.q_mapping(observations[trn_idx], observations[trn_idx-1], observations[all_idx-1], 100)
obs = observations[tst_idx] 
cli = np.zeros(observations[tst_idx].shape)
per = persistence[tst_idx]

fcs = np.nanmean(forecast[tst_idx], axis=1)

 
L = 5; B = 100
corf = fcts.calc_bootstrap(fcs, obs, fcts.calc_corr,  L, B, [5,95])
corp = fcts.calc_bootstrap(per, obs, fcts.calc_corr,  L, B, [5,95])
cora = fcts.calc_bootstrap(np.nanmean(forecast[all_idx], axis=1), 
                            observations[all_idx], 
                            fcts.calc_corr,  L, B, [5,95])

dtaxis = pd.to_datetime(Y.index[all_idx].values)
p_95 = np.nanpercentile(forecast[all_idx], q=[2.5, 97.5], axis=1)

axes.plot(observations[all_idx],c='k',linewidth=0.8, label='Observations')
axes.plot(persistence[all_idx],c='k',linestyle='--',linewidth=0.8,label='Persistence, CORR='+str(round(corp[0],2)))
axes.plot(np.nanmean(forecast[all_idx], axis=1),c='r',label='Ensemble mean')
#axes[i].fill_between(x=dtaxis, y1=p_95[0], y2=p_95[1], color="r", alpha=0.2, label='Ensemble 95% range')

#axes[i].axvline(x=Y.index[trn_idx][-1], c='k', linewidth=2.0)
axes.legend(loc='upper right',ncol=2,frameon=True,fontsize=6)
axes.set_title(str(mon)+',  Correlation: '+str(round(corf[0],2))+str(corf[2])) #+', MSSS$_{clim}$: '+
                    #str(round(msss_clim[1],2))+msss_clim[3]+', MSSS$_{pers}$: '+
                    #str(round(msss_pers[1],2))+msss_pers[3]) #+', a='+str(np.mean(alp))[0:5])
#axes[i].set_xlim(['1940-10-31 00:00:00', '2012-10-31 00:00:00'])
#axes[i].set_ylim([-3, +3])
#axes[i].set_yscale('log')


plt.tight_layout(); 
#fig.savefig(oudir+'fig_probab_nmodels'+str(ens_size)+'_'+data['basename']+'.png',dpi=120)
fig.savefig(oudir+'fig_predicted_timeseries.png'); plt.close()














# Apply a moving ACC analysis window for different seasons and 
# Study the effect of ensemble size

sns.set(style="whitegrid")
fig, axes = plt.subplots(1,2, figsize=(9,3), gridspec_kw={'width_ratios':[2,1]}, sharey=True)

window=12; ssn_c = ('r', 'b', 'k', 'c')
n_smpls = int(data['n_estim'])
#acc_ens = np.full((len(data['time_info'][2]), n_smpls), np.nan)

for i,tgroup in enumerate(data['time_info'][2]): #enumerate(data['seasons']): 
    
    tgr_idx = (results[data['time_info'][3]]==str(tgroup)) & (results['Best model']) 
    
    all_idx = fcts.bool_index_to_int_index(np.isin(data['Y_raw'][data['time_info'][1]], tgroup) & np.isin(data['Y_raw']['time.year'], data['all_yrs']))
    trn_idx = fcts.bool_index_to_int_index(np.isin(data['Y_raw'][data['time_info'][1]], tgroup) & np.isin(data['Y_raw']['time.year'], data['trn_yrs']))
    tst_idx = fcts.bool_index_to_int_index(np.isin(data['Y_raw'][data['time_info'][1]], tgroup) & np.isin(data['Y_raw']['time.year'], data['tst_yrs']))
    
    forecast = np.full((data['Y'][data['y_var']].shape[0], results[tgr_idx].shape[0]*n_smpls), np.nan)
    acc_ens  = np.full(results[tgr_idx].shape[0]*n_smpls, np.nan)
    
    j = 0; 
    for m, model in results[tgr_idx].iterrows(): 
        mdl = model['Fitted model']
        vrs = model['Variable sample']
        # Refit
        #mdl = mdl.fit(X.ix[trn_idx][vrs], Y[trn_idx])
        
        #forecast[all_idx,j] = mdl.predict(X[vrs].ix[all_idx])
        #forecast[all_idx,j] = fcts.q_mapping(observations[trn_idx], forecast[trn_idx,j], forecast[all_idx,j], 100)
        print(tgroup, model['N of PCs'], model['Test ACC'], model['Train ACC'])
        for k,estm in enumerate(mdl.estimators_):
            est_vrs = mdl.estimators_features_[k]
            forecast[all_idx,j] = estm.predict(X[vrs].ix[all_idx,est_vrs])
            forecast[all_idx,j] = fcts.q_mapping(observations[trn_idx], forecast[trn_idx,j], forecast[all_idx,j], 100)
            
            acc_ens[j] = fcts.calc_corr(observations[tst_idx], np.nanmean(forecast[tst_idx],axis=1))
            j+=1
    
    fcs = np.nanmean(forecast[all_idx], axis=1)
    obs = observations[all_idx] 
    
    acc_mov = np.full(observations[all_idx].shape, np.nan); 
    for k,tstep in enumerate(all_idx):
        try: 
            acc_mov[k] = fcts.calc_corr(fcs[k-window:k+window], obs[k-window:k+window])
        except: pass
    
    acc_mov[-window:] = np.nan
    dtaxis = pd.to_datetime(data['Y']['time'][all_idx].values).values
    ens_50 = np.arange(1,51).astype(int)
    
    axes[0].plot(dtaxis, acc_mov,  linewidth=1.8, label=str(tgroup))   #c=ssn_c[s],
    axes[1].plot(ens_50, acc_ens[:ens_50.shape[0]], linewidth=1.8, label=str(tgroup)) #c=ssn_c[s], 
    
    if(str(tgroup)=='1'): axes[0].axvline(x=data['Y']['time'][trn_idx][-1].values, c='k', linewidth=2.0)


axes[0].set_title('a) Temporal evolution of ACC in '+data['y_area'][0:2].upper()); 
axes[1].set_title('b) ACC vs. ensemble size in '+data['y_area'][0:2].upper());
axes[0].set_ylabel('ACC'); axes[1].set_ylabel('ACC');
axes[0].set_xlabel('Year'); axes[1].set_xlabel('Ensemble size')
axes[1].set_ylabel('ACC');plt.tight_layout();plt.legend(loc='lower right',ncol=4,fontsize='xx-small')
fig.savefig(oudir+'fig_movingACC_nmodels'+str(n_smpls)+'_'+data['basename']+'.png',dpi=120);
#fig.savefig(oudir+'fig_movingACC_nmodels'+str(n_smpls)+'_'+data['basename']+'.png'); plt.close()









print('Finished plotting results!')  





