#!/usr/bin/env python



import itertools
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs




# --- Bias correction with quantile mapping ---

def interp_extrap(x, xp, yp):
    """      
    This function is used by q_mapping().
    Projects the x values onto y using the values of 
    [xp,yp], extrapolates conservatively if needed.
    """
    
    y = np.interp(x, xp, yp)
    y[x<xp[ 0]] = yp[ 0] + (x[x<xp[ 0]]-xp[ 0])
    y[x>xp[-1]] = yp[-1] + (x[x>xp[-1]]-xp[-1])
    return y

def q_mapping(obs,ctr,scn,nq):
    """ 
    Quantile mapping. Three (n,) or (n,m) numpy arrays expected for input, 
        where n = time dimension and m = station index or grid cell index.
    
    First argument represents the truth, usually observations.
    
    Second is a sample from a model.
    
    Third is another sample which is to be corrected based on the quantile 
        differences between the two first arguments. Second and third 
        can be the same.
    
    Fourth is the number of quantiles to be used, i.e. the accuracy of correction.

    Linear extrapolation is applied if the third sample contains values 
        outside the quantile range defined from the two first arguments.
    """
    
    # Calculate quantile locations to be used in the next step
    q_intrvl = 100/float(nq); qtl_locs = np.arange(0,100+q_intrvl,q_intrvl) 

    # Calculate quantiles
    q_obs = np.percentile(obs, list(qtl_locs), axis=0)
    q_ctr = np.percentile(ctr, list(qtl_locs), axis=0) 
    
    if(len(obs.shape)==1):
        # Project the data using the correction function 
        return interp_extrap(scn,q_ctr,q_obs)
    
    if(len(obs.shape)==2):
        # Project the data using the correction function, separately for each location 
        out = np.full(scn.shape,np.nan)
        for i in range(out.shape[1]):
            out[:,i] = interp_extrap(scn[:,i],q_ctr[:,i],q_obs[:,i])

        return out




def make_data_normal_distributed(data):
    #from scipy.stats import norm
                
    obs = np.random.standard_normal(data.shape)
                
    return q_mapping(obs,data,data,1000)
    

# --- Evaluation tools ---


def calc_rmse(a,b,*kwargs): 
    return np.sqrt(np.nanmean((a-b)**2))

def calc_corr(a,b,*kwargs): 
    return np.corrcoef(a,b)[0,1]

def calc_msss(fcs,obs,ref):
    mse_f = np.nanmean((fcs-obs)**2)
    mse_r = np.nanmean((ref-obs)**2) 
    return 1 - mse_f/mse_r

def acorr(x, length=15):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
        for i in range(1, length)])



def calc_bootstrap(fcs, obs, func,  L, B=1000, bootstrap_range=[2.5, 97.5]):
    """ 
    Calculates moving block bootstrap estimates for an
    evaluation metric defined and calculated inside 'func' argument.

    INPUT
    fcs:                forecasted time series
    obs:                observed time series
    func:               function to be evaluated, e.g. calc_corr    
    L:                  length of the bootstrap block
    B:                  number of bootstrap samples
    bootstrap_range:    lower and upper percentage points to be estimated
    
    OUTPUT
    res:                result from evaluation function
    est_range:          estimated percentage point values
    sgn:                True if the lower %-point value is positive
    """
    
    from sklearn.utils import resample
    
    idxs = np.arange(len(fcs))
    results = []
    
    random_state = 0
    for smp in range(B):
        block_sample = np.array([]).astype(int)
        while(len(block_sample) < len(fcs)):
            random_state += 1
            rolls = resample(idxs, n_samples=1, random_state=random_state)[0]
            block = np.roll(idxs, rolls)[0:L]
            block_sample = np.append(block_sample, block)
        
        block_sample = block_sample[0:len(idxs)]
        results.append(func(fcs[block_sample],obs[block_sample]))
    
    est_range = np.percentile(results, bootstrap_range)
    lower = est_range[0]
    
    res = func(fcs,obs)
    
    if(lower > 0):  sgn = True
    else:           sgn = False
    
    return res, est_range, sgn








# --- Manipulators ---

def apply_PCA(data, ncomp):
    """ Decomposition of data with principal component analysis. 
        Assumes data to be of shape (time, gridcell) """
    
    import sklearn.decomposition as dc
    
    pca = dc.PCA(n_components=ncomp, whiten=False, svd_solver='full')
    cps = pca.fit_transform(data)
    svl = pca.singular_values_
    
    return cps,pca,svl




# --- Reading and data extraction --

def bool_index_to_int_index(bool_index):
    return np.where(bool_index)[0]




def prepare_X_array(Y, X_vars, predef_pcas, n_pcs, trn_yrs, all_yrs, lags,
                    include_persistence=False):

    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer as Imputer
    from sklearn.utils import resample
    
    # An empty dataframe containing only dates as an index variable,
    # and probably persistence variables
    try:
        X = pd.DataFrame(index=X_vars[0].time.to_index())
    except:
        X = pd.DataFrame(index=X_vars[0].index)

    X.index.names = ['Date']
    Y.index.names = ['Date']
    
    if(include_persistence):
        X = pd.merge(Y, X, left_on='Date', right_on='Date', how='outer').sort_index() 
        y_var = Y.name
        X[y_var+'-1'] = np.roll(X[y_var], 1)
        X[y_var+'-1'][0:1] = X[y_var].values[0]
    
    # Remove unnecessary columns
    drops = ['season','season_x','season_y','month_x','month_y','month','year_x','year_y','year']
    for i,dr in enumerate(drops):
        try:    X = X.drop(columns=dr)
        except: pass



    # Variables from gridded datasets
    X_pcas = {}
    for i,vr in enumerate(X_vars):

        trn_idx = bool_index_to_int_index(np.isin(X_vars[i]['time.year'], trn_yrs))
        all_idx = bool_index_to_int_index(np.isin(X_vars[i]['time.year'], all_yrs))
        
        _, pca, svl = apply_PCA(X_vars[i].values[trn_idx],n_pcs)
        cps = pca.transform(X_vars[i].values)
        
        
        for cpn in range(n_pcs):
            vrb_name = vr.name+'-'+str(cpn+1); print(vrb_name, cpn, cps.shape, X.shape)
            X[vrb_name] = cps[:,cpn]
            
            
            for lag in lags: # range(1,n_lags+1):
                X[vrb_name+'-'+str(lag)] = np.roll(X[vrb_name], lag)
                X[vrb_name+'-'+str(lag)][0:lag] = X[vrb_name][0]
            
            X = X.drop(columns=vrb_name)
        
        X_pcas[vr.name] = pca; 

    
    # Normalize to N(0,1)
    #X[:] = StandardScaler().fit_transform(X.values)
    #X[:] = make_data_normal_distributed(X.values)
    
    return X, X_pcas





def adjust_lats_lons(ds, var):
    coord_names =   [
                    ['longitude', 'latitude'],
                    ['X', 'Y'],
                    ]
    for nmes in coord_names:
        try:
            ds = ds.rename({nmes[0]: 'lon', nmes[1]: 'lat'}) 
        except: 
            pass  
    
    if(ds.lon.values.max() > 350):
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        rolls = np.sum(ds.lon.values < 0); ds = ds.roll(lon=rolls*(-1))
    
    if(ds.lat.values[0] > ds.lat.values[-1]):
        ds['lat'] = np.flipud(ds['lat'])
        ds[var].values = np.flip(ds[var], axis=1)
    
    return ds
    



def read_namelist(namelist_file):
    """ Read data from the namelist file """
    
    import pandas as pd
    import ast
    
    namelist = pd.read_csv(namelist_file,index_col='Key')
    
    # Github/code, Data input, and Results directories
    #codir = str(namelist.loc['codir'].values[0])
    #indir = str(namelist.loc['indir'].values[0])
    #oudir = str(namelist.loc['oudir'].values[0])
    
    yr_first = int(namelist.loc['yr_first'])
    yr_last  = int(namelist.loc['yr_last'])
    
    # Number of estimators
    n_estim  = int(namelist.loc['n_estim']) 
    
    # Number of PCs to be studied from ERA-Interim variables
    n_pcs    = int(namelist.loc['n_pcs'])   

    # Percentages of data dimensions to be included to each random sample
    p_feat = float(namelist.loc['p_feat']) 
    p_smpl = float(namelist.loc['p_smpl']) 

    # Lags of predictor data (time steps)
    # Include 0 if you want to predict with no lags
    lags = ast.literal_eval(namelist.loc['lags'].values[0])

    # Define year lists for train and test sets
    years  = list(np.arange(yr_first, yr_last+1))
    test_yrs  = list(years[-6:]) # Last six years for testing
    train_yrs = list(np.array(years)[~np.isin(years,test_yrs)])   

    # Parameters to be included in the experiment
    params = ast.literal_eval(namelist.loc['params'].values[0])

    # Acronym for the base-estimator
    estim = str(namelist.loc['estimator'].values[0])

    # Name for the experiment
    exp_name = '_'.join(params)+'_lags'+str(lags)+'_p_smpl'+ \
                    str(p_smpl)+'_p_feat'+str(p_feat)+'_n_estim'+str(n_estim)+estim
                    
    return exp_name,years,test_yrs,train_yrs, \
            params,estim,n_estim,n_pcs,lags,p_feat,p_smpl




def read_and_select(fles, var, year_range, timeresol_out, area, remove_clim):
    
    # Open files in parallel using Xarray and Dask
    ds = xr.open_mfdataset(fles, parallel=True) 
    
    # Select correct time steps
    t_index = (ds['time.year'] >= year_range[0]) & (ds['time.year'] <= year_range[1])
    ds = ds.sel(time=t_index.values)
        
    # Assure lats to be [-90,90] and lons [-180,180]
    ds = adjust_lats_lons(ds, var)
    
    # Select region
    if(area=='europe'): ds = ds.squeeze().sel(lat=slice( 33,73), lon=slice(-12,40)) 
    if(area=='scandi'): ds = ds.squeeze().sel(lat=slice( 53,73), lon=slice(  4,36)) 
    if(area=='finlnd'): ds = ds.squeeze().sel(lat=slice( 59,70), lon=slice( 20,32)) 
    if(area=='norhem'): ds = ds.squeeze().sel(lat=slice(-10,87)) 
    
    # Apply time averaging
    ds = ds.resample(time=timeresol_out).mean()
    ds = ds.stack(gridcell=('lat', 'lon')).fillna(0)
    
    # Remove cliomatology e.g. calculate anomalies (optional)
    if(remove_clim):
        clim = ds.groupby('time.month').mean('time')
        ds = ds.groupby('time.month') - clim
    
    return ds






# --- Weather typing


def calc_weather_types(raw_input, n_components, n_weather_types, lags):
    
    from sklearn import cluster
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    
    
    months = raw_input[0].time.to_index().month
    summer = (months==6) | (months==7) | (months==8) 
    

    components = pd.DataFrame(index=raw_input[0].time.to_index()) 
    
    pca_objects = []
    for i,parameter in enumerate(raw_input):
        
        param_name = parameter.name
        
        # Remove climatology
        clim = parameter.groupby('time.month').mean('time')
        parameter = parameter.groupby('time.month') - clim
        
        # Perform PCA for all data, not only for summer months
        _, pca, svl = apply_PCA(parameter.values[summer],n_components)
        cps_raw = pca.transform(parameter.values)
        pca_objects.append(pca)
        
        # Store components to a dataframe
        for cpn in range(n_components):
            vrb_name = param_name+str(cpn+1)
            components.loc[:,vrb_name] = cps_raw[:,cpn]
    
    # Standardize all components and perform KMeans clustering
    components[:] = StandardScaler().fit_transform(components.values)
    #components[:] = make_data_normal_distributed(components.values)
    
    estimator = cluster.KMeans(n_clusters=n_weather_types, random_state=50, 
                                n_init=100, n_jobs=4).fit(components.values)
    
    # Apply clustering to the data and save results to datasets
    wtypes_raw = estimator.predict(components.values) + 1
    print(wtypes_raw); print(n_weather_types)
    
    dummy_classes = OneHotEncoder(categories='auto', 
                        sparse=False).fit_transform(wtypes_raw.reshape(-1,1))
    
    wtypes_out = pd.DataFrame(data=dummy_classes, index=components.index, 
                                columns=(np.arange(n_weather_types)+1).astype(str))
    

    # Lag weather types 
    for col in list(wtypes_out.keys()):
        for lag in lags:
            wtypes_out['WTP'+col+'-'+str(lag)] = np.roll(wtypes_out[col], lag)
            wtypes_out['WTP'+col+'-'+str(lag)][0:lag] = wtypes_out[col][0]
        
        wtypes_out = wtypes_out.drop(columns=[col])

    wtypes_out['Weather type'] = wtypes_raw   
    
    return wtypes_out, pca_objects













# --- Plotting ---


def scientific_colormaps(cmap_name, 
        scm_base_dir='/lustre/tmp/kamarain/ScientificColourMaps5/'):
    
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    
    cmap_file = scm_base_dir+cmap_name+'/'+cmap_name+'.txt'
    cmap_data = np.loadtxt(cmap_file)
    
    return LinearSegmentedColormap.from_list(cmap_name, cmap_data)
    
    


def plot_contours(ds, trns, ax, cmap, lvl_ctp, lvl_cnt, center):
    
    ctp = ds.unstack('gridcell').plot.contourf(transform=trns, 
        ax=ax, cmap=cmap,levels=lvl_ctp, center=center, add_colorbar=False)
    
    cnt = ds.unstack('gridcell').plot.contour(transform=trns, 
        ax=ax, levels=lvl_cnt, colors='k', linewidths=1.5, center=center)
    
    clbls = ax.clabel(cnt, fontsize=11, inline=True, inline_spacing=1, fmt='%1.0f')
    
    return ctp, cnt



class LowerThresholdRobinson(ccrs.Robinson):   
    @property
    def threshold(self):
        return 1e3

class LowerThresholdOrtho(ccrs.Orthographic):  
    @property
    def threshold(self):
        return 1e3

class LowerThresholdLConf(ccrs.LambertConformal):
    @property
    def threshold(self):
        return 1e3

def plot_rectangle(ax,trans, lonmin,lonmax,latmin,latmax,clr,lw,alp):
    xs = [lonmin,lonmax,lonmax,lonmin,lonmin]
    ys = [latmin,latmin,latmax,latmax,latmin]
    ax.plot(xs, ys,transform=trans,color=clr,linewidth=lw,alpha=alp)
    pass

def plot_scatter(ax,trans, lons,lats,clr1,clr2,sze,mrk,alp):
    ax.scatter(lons, lats,transform=trans,s=sze,marker=mrk,
                edgecolors=clr2,c=clr1,linewidth=0.3,alpha=alp)
    pass

def plot_text(ax,trans,lons,lats,txt,clr,fs,rot,box):
    
    font = {'family': 'sans-serif',
            'color':  clr,
            'weight': 'black',
            'style': 'italic',
            'size': fs,
            'rotation':rot,
            'ha':'center',
            'va':'center',
            }
    
    for i in range(len(txt)):
        if(box): 
            bbox_props = dict(boxstyle="square,pad=0.1", fc="w", ec="k", lw=1)
            ax.text(lons[i],lats[i],txt[i],transform=trans,fontdict=font,bbox=bbox_props)
        else:
            ax.text(lons[i],lats[i],txt[i],transform=trans,fontdict=font)





