# Importation
import expe as exp
import numpy as np
from tqdm import tqdm
import MAB as mab
import matplotlib.pyplot as plt
import BernoulliMAB as BM

param = {
    'UCB1': {'rho': np.sqrt(2)},
    'BayesUCB': {'p1': 1, 'p2': 1, 'c':0},
    'MOSS': {'rho': 0.2},
    'ExploreCommit': {'m': 50},
    'Tuned_GPUCB': {'c': 0.9},
    'IDS_approx': {'N_steps': 10000, 'display_results': False}
}

methods = ['UCB1', 'TS', 'UCB_Tuned', 'BayesUCB', 'KG', 'Approx_KG_star', 'MOSS', 'IDS_approx']
bernoulli_methods = ['UCB1', 'TS', 'IDS_approx']
np.random.seed(1234)

check_Finite = False
check_Bernoulli = True
check_Gaussian = False
check_linear = False
check_single_expe = False
if __name__ == '__main__':
    if check_Finite:
        p, q, R = exp.build_finite(L=1000, K=10, N=200)
        exp.check_finite(p, q, R, theta=0, N=100, T=1000)
    if check_Bernoulli:
        # param['IDS_approx']['beta1'] = (np.ones(10)*10).astype(int)
        # param['IDS_approx']['beta2'] = (np.ones(10)*1).astype(int)
        # exp.beta_bernoulli_expe(T=500, n_expe=100, n_arms=10, methods=bernoulli_methods,
        #                         param_dic=param, doplot=True)
        param['IDS_approx']['beta1'] = (np.ones(2)*10).astype(int)
        param['IDS_approx']['beta2'] = (np.ones(2)*1).astype(int)
        exp.beta_bernoulli_expe(T=1000, n_expe=10, n_arms=2, methods=bernoulli_methods,
                                param_dic=param, doplot=True)
    if check_Gaussian:
        exp.check_gaussian(n_expe=20, n_arms=10, T=400, doplot=True)

    if check_linear:
        exp.check_linearGaussianMAB(n_expe=100, n_features=10, n_arms=10, T=500)

    if check_single_expe:
        beta_1 = (np.ones(10)*10).astype(int)
        beta_2 = (np.ones(10)*1).astype(int)
        p = [0.6, 0.15, 0.97, 0.07, 0.83, 0.37, 0.55, 0.04, 0.47, 0.88]
        my_mab = BM.BetaBernoulliMAB(p)
        print(my_mab.IDS_approx(T=1000, N_steps=10000, beta1=beta_1, beta2=beta_2, display_results=True))
