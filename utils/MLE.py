# import from built-in modules 
from scipy.optimize import minimize

import numpy  as np
import pandas as pd

# import from personal modules 
from .dfModifier import getTimeAttriVal_mdf_sim, matchTime, modify_df_sim
from .extract    import bias_indices, listNlines
from .others     import checkFeasibility

# variables of interest
attributes_95 = ['Tritium', 'Uranium', 'Aluminum', 'pH']
attributes_110 = ['Tritium', 'Uranium', 'pH']
attributes_sets = {95: attributes_95, 110: attributes_110}

# bias 
bias_names = {
        95:  {'Tritium': 'bias_95_tri', 'Uranium':'bias_95_uran', 'pH':'bias_95_ph', 'Aluminum':'bias_95_al'}, 
        110: {'Tritium': 'bias_110_tri','Uranium':'bias_110_uran','pH':'bias_110_ph'}
        }

ones = np.array([1 for i in range(len(attributes_95) + len(attributes_110))])
default_bias_factors = pd.Series(data=ones, index=bias_indices)

def getDistances(obs, well, sim_path, bias_factors=default_bias_factors):
    ''' get distances between the observations and  simulated data (each column is for an observation)
        the order of columns follow the one in attributes_sets
        
        obs:          a dataframe storing the real observations at well {well}
        well:         well number 
        sim_path:     a file path to a simulated data
        bias_factors: a pandas series of length 7 
                      columns: [bias_95_tri, bias_95_uran, bias_95_al, bias_95_ph, bias_110_tri, bias_110_uran, bias_110_ph]
    '''
     
    # modify the simulated data     
    sim = pd.read_csv(sim_path, skiprows=2)
    sim = modify_df_sim(sim, well)    

    # calculate the distance between observation and simulated data
    distances = None 
    for attribute in attributes_sets[well]: 
        dates, values = getTimeAttriVal_mdf_sim(sim95, attribute, bias_factors[bias_names[well][attribute]]) 
        # a dataframe with columns matched_dates_tr (truncated to month), obs_dates, obs_{attribute}, sim_dates, sim_{attribute}
        matchedData = matchTime(obs95, attribute, dates, values)  
        matchedData[attribute] = matchedData[f'obs_{attribute}'] - matchedData[f'sim_{attribute}']
        
        # update the distances dataframe 
        data = matchedData[['obs_dates', attribute]]
        if distances is None: distances = data
        else: distances.merge(data, on='obs_dates', how='outer')
    
    distances.drop(columns=['obs_dates'], inplace=True)
    distances = distances[attributes_sets[well]] # make sure the column order
    
    return distance

def variance_optimizer(obs95, obs110, sim_path, dir_log, bias_factors=default_bias_factors):
    
    fnum = int( ''.join(list(filter(str.isdigit, sim_path.name))) )
    # check the feasibility of the parameter configuration based on the simulation .log/.out files
    fname_log = Path(dir_log).joinpath(f'sim{fnum}.out')
    lastTwoLines = listNlines(f, 2)
    feasible = checkFeasibility(lastTwoLines)
    if feasible != "feasible": return

    # get distances (each column is for an observation) 
    # the order of columns follow the one in attributes_sets
    distances95  = getDistances(obs95,  95, sim_path, bias_factors)
    distances110 = getDistances(obs110, 110, sim_path, bias_factors)
   
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
        # the outer sum is summing over i in I
        # the inner sum is summing over t in T_i
        term2 = np.sum(np.nansum(np.square(distances95), axis=0)/np.array(vars_95))
                + np.sum(np.nansum(np.square(distances110), axis=0)/np.array(vars_110)) 
        # omit (1) the term involving sqrt(2*pi) and (2) a factor of 0.5
        return term1 + term2

    x0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    bnds = tuple( [(0, None) for i in range(len(x0))] )
    res = minimize(loglik, x0, bounds=bnds) # L-BFGS-B algorithm
    if not res.success:
        print(f'sim{fnum}')
        print(res.message)
        return 

    variances, val = (res.x)**2, res.fun

    # shift fnum by 1 so that it starts with index 0 
    return (fnum - 1, np.array(variances), val)

