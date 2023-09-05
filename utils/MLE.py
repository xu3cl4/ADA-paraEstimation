# import from built-in modules 
from scipy.optimize import minimize

import numpy  as np
import pandas as pd

# import from personal modules 
from .dfModifier import getTimeAttriVal_mdf_sim, matchData, modify_df_sim

# variables of interest
attributes_95 = ['Tritium', 'Uranium', 'Aluminum', 'pH']
attributes_110 = ['Tritium', 'Uranium', 'pH']
attributes_sets = {95: attributes_95, 110: attributes_110}

def getDistances(obs, well, sim_path, para_ens):
    ''' get distances between the observations and  simulated data (each column is for an observation)
        the order of columns follow the one in attributes_setsi '''
   
    # modify the simulated data     
    sim    = pd.read_csv(sim_path, skiprows=2)
    sim  = modify_df_sim(sim, well)    

    # calculate the distance between observation and simulated data
    distances = None 
    for attribute in attributes_sets[well]: 
        dates, values = getTimeAttriVal_mdf_sim(sim95, attribute, well, para_ens) 
        # a dataframe with columns: matched_dates_tr (truncated to month), obs_dates, obs_{attribute}, sim_dates, sim_{attribute}
        matchedData = matchTime(obs95, attribute, dates, values)  
        matchedData[attribute] = matchedData[f'obs_{attribute}'] - matchedData[f'sim_{attribute}']
        
        # update the distances dataframe 
        data = matchedData[['obs_dates', attribute]]
        if distances is None: distances = data
        else: distances.merge(data, on='obs_dates', how='outer')
    
    distances.drop(columns=['obs_dates'], inplace=True)
    distances = distances[attributes_sets[well]] # make sure the column order
    
    return distance

def variance_optimizer(obs95, obs110, sim_path, para_ens):
    
    fnum = int( ''.join(list(filter(str.isdigit, sim_path.name))) )

    # get distances (each column is for an observation) 
    # the order of columns follow the one in attributes_sets
    distances95  = getDistances(obs95,  95, sim_path, para_ens)
    distances110 = getDistances(obs110, 110, sim_path, para_ens)
   
    # calculate T_i's: the number of certain observations at certain well 
    # the order of columns follow the one in attributes_sets
    Ts_95  = obs95[attributes_95].count()
    Ts_100 = obs110[attributes_110].count()

    # define the log-likelihood 
    def loglik(var_s):
        var1, var2, var3, var4, var5, var6, var7 = var_s
        vars_95  = [var1, var2, var3, var4]
        vars_110 = [var5, var6, var7]
        term1 = np.sum(np.array(Ts_95)*np.log(vars_95)) + np.sum(np.array(Ts_110)*np.log(vars_110)) 
        term2 = np.sum(np.nansum(np.square(distances95), axis=0)/np.array(vars_95))
                + np.sum(np.nansum(np.square(distances110), axis=0)/np.array(vars_110)) 
        # omit the term of sqrt(2*pi) and a factor of 0.5
        return term1 + term2

    x0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None))
    res = minimize(loglik, x0, bounds=bnds)
    if not res.success:
        print(f'sim{fnum}')
        print(res.message)
        return 

    variance, val = (res.x)**2, res.fun

    # shift fnum by 1 so that it starts with index 0 
    return (fnum - 1, variance, val)
