# import from built-in modules 
from scipy.optimize import minimize

import numpy  as np
import pandas as pd

# import from personal modules 
from .dfModifier import getTimeAttriVal_mdf_sim, matchData, modify_df_sim

# variables of interest
attributes_95 = ['Tritium', 'Uranium', 'Aluminum', 'pH']
attributes_110 = ['Tritium', 'Uranium', 'pH']

def getDistances(obs95, obs110, sim_path, para_ens):
   
    # modify the simulated data     
    sim    = pd.read_csv(sim_path, skiprows=2)
    sim95  = modify_df_sim(sim, 95)    
    sim110 = modify_df_sim(sim, 110)

    # calculate the distance between observation and simulated data
    distances95 = [] 
    for attribute in attributes_95: 
        dates, values = getTimeAttriVal_mdf_sim(sim95, attribute, well, para_ens) 
        matchedData = matchTime(obs95, attribute, dates, values)  # a dataframe with columns: matched_dates_tr (truncated to month), obs_dates, sim_dates, obs_vals, sim_vals
        matchedData['dist'] = matchedData['obs_vals'] - matchedData['sim_vals']
        distances95.append(matchedData[['obs_dates', 'dist']])
    distances95 = pd.concat(distances95, join='outer')
    distances95.drop(columns=['obs_dates'], inplace=True)
    
    distances110 = []
    for attribute in attributes_110:
        dates, values = getTimeAttriVal_mdf_sim(sim110,attribute, well, para_ens)
        matchedData = matchTime(obs110, attribute, dates, values) # a dataframe with columns: matched_dates_tr (truncated to month), obs_dates, sim_dates, obs_vals, sim_vals
        matchedData['dist'] = matchedData['obs_vals'] - matchedData['sim_vals']
        distances110.append(matchedData[['obs_dates', 'dist']])
    distances110 = pd.concat(distance110, join='outer')
    distances110.drop(columns=['obs_dates'], inplace=True)

    return (distances95, distances110)

def variance_optimizer(obs95, obs110, sim_path, para_ens):
    
    fnum = int( ''.join(list(filter(str.isdigit, sim_path.name))) )

    # get distances (each column is for an observation) 
    # the order of columns follow the one in attributes_sets
    distances95, distances110 = getDistances(obs95, obs110, sim_path, para_ens)
   
    # calculate T_i's: the number of certain observations at certain well 
    # the order of columns follow the one in attributes_sets
    Ts_95  = obs95[attributes_95].count()
    Ts_100 = obs110[attributes_110].count()

    # define the log-likelihood 
    def loglik(sigmas):
        sig1, sig2, sig3, sig4, sig5, sig6, sig7 = sigmas
        sigs_95  = [sig1, sig2, sig3, sig4]
        sigs_110 = [sig5, sig6, sig7]
        term1 = np.sum(np.array(Ts_95)*np.log(sigs_95)) + np.sum(np.array(Ts_110)*np.log(sigs_110)) 
        term2 = 0.5*( np.sum(np.nansum(np.square(distances95), axis=0)/np.square(sigs_95))
                    + np.sum(np.nansum(np.square(distances110), axis=0)/np.square(sigs_110)) )
        return term1 + term2

    x0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    res = minimize(loglik, x0)
    if not res.success:
        print(f'sim{fnum}')
        print(res.message)
        return 

    variance, val = (res.x)**2, res.fun

    # shift fnum by 1 so that it starts with index 0 
    return (fnum - 1, variance, val)
