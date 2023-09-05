# import from built-in modules 
from argparse       import ArgumentParser, RawTextHelpFormatter as RT
from joblib         import delayed, Parallel
from pathlib        import Path

import numpy  as np
import pandas as pd

# import from personal modules 
from utils.dfModifier import modify_df_real
from utils.MLE        import attributes_95, attributes_110, variance_optimizer

FPATH = Path(__file__)
DIR = FPATH.parent

def getArguments():
    ''' parse the command-line interface
        the command line takes four required arguments  

        real95:  the file path to the real observations at well 95 
        real110: the file path to the real observations at well 110 
        para:    the file path to .csv file of paramter ensemble
        dif_sim: the directory to the simulated results from the parameter ensemble
        opt:     the file path to output the estimation results
    '''  
    parser = ArgumentParser(formatter_class=RT)
    parser.add_argument('real95',  type = str, help="the file path to the real observations at well 95")
    parser.add_argument('real110', type = str, help="the file path to the real observations at well 110")
    parser.add_argument('para',    type = str, help="the file path to .csv file of parameter ensemeble")
    parser.add_argument('dir_sim', type = str, help="the directory to the simulated results from the parameter ensemble")
    parser.add_argument('opt',     type = str, help="the file path to output the estimation results")

    return parser.parse_args()

def main():
    ''' the basic structure of the python script '''  
    args = getArguments()

    real95  = DIR.joinpath(args.real95)
    real110 = DIR.joinpath(args.real110)
    para    = DIR.joinpath(args.para)
    dir_sim = DIR.joinpath(args.dir_sim)
    opt     = DIR.joinpath(args.opt)

    # read files of real observations  
    real95 = pd.read_csv(real95)
    real95 = modify_df_real(real95)

    real110 = pd.read_csv(real110)
    real110 = modify_df_real(real110)
    
    print('---- number of observations ---- ')
    print('Well 95:')
    print(real95[attributes_95].count())
    print('Well 110:')
    print(real110[attributes_110].count())

    # read the parameter ensemble 
    para = pd.read_csv(para) 

    # read simulated data and compute the MLE, using parallelism  
    
    sims = Path(dir_sim).glob('*.out')  # iterate through all the simulated .out files 

    nc = int(min(n, ncpu()))
    n = len(list(sims))
    r = Parallel(n_jobs=nc, verbose=1, backend="multiprocessing")\
            (delayed(variance_optimizer)(
                    obs95=real95, obs110=real110, sim_path=sim, para_ens=para
                )
            for sim in sims
        )

    ens_num, variances, vals = zip(*r)       # ens_num starts with 0
    idx_max = vals.index(max(vals))
    ens_num_max = ens_num[idx_max]
    para_MLE = para.iloc[ens_num_max]
    print(para_MLE)
    variance_MLE = variances[idx_max]
    print(f'variance estimate: {variance_MLE}')
        

if __name__ == "__main__":
    main()
