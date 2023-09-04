# import from built-in modules 
from scipy.optimize import minimize

from numpy as np

# import from personal modules 
from .dfModifier import modify_df_sim

attributes_95 = ['Tritium', 'Uranium', 'Aluminum', 'pH']
attributes_110 = ['Tritium', 'Uranium', 'pH']

attributes_sets = {95: attributes_95, 110: attributes_110}

def getDistances(obs95, obs110, sim_path):
    
    sim    = pd.read_csv(sim, skiprows=2)
    sim95  = modify_df_sim(sim, 95, para)    
    sim110 = modify_df_sim(sim, 110, para)

        for attribute in attributes_sets[95]:
            distances95[attribute].loc[i]  = getDistances(real95,  sim95,  attribute)

        for attribute in attributes_sets[110]:
            distances110[attribute].loc[i] = getDistances(real110, sim100, attribute) 

    for i, attribute in enumerate(attributes):
        sim_attr = sim[sim['variable'] == attribute.lower()]
        sim_attr = sim_attr.pivot(index="time", columns="region", values="value")
        sim_attr_avg = sim_attr.mean(axis=1)        # average over the chosen depths
        dates = (sim_attr_avg.index).to_series()
        values = (sim_attr_avg.to_frame())[0]


def variance_optimizer(obs95, obs110, sim_path, para_ens):
    
    fnum = int( ''.join(list(filter(str.isdigit, sim.name))) )

    # get distances (each column is for an observation) 
    # the order of columns follow the one in attributes_sets
    distances95, distances110 = getDistances(obs95, obs110, sim_path, para_ens)
   
    # calculate T_i's: the number of certain observations at certain well 
    # the order of columns follow the one in attributes_sets
    Ts_95  = obs95.count()
    print('Well 95')
    print(Ts_95)
    Ts_100 = obs110.count()
    print('Well 110')
    print(Ts_110)

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

    variance, val = res.x, res.fun

    # shift fnum by 1 so that it starts with index 0 
    return (fnum - 1, variance, val)
