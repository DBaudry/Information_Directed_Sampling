""" Packages import """
import expe as exp
import numpy as np
import pickle as pkl
from utils import build_finite
import os
import utils

np.random.seed(46)

path = r'C:\Users\dobau\Desktop\MVA\Reinforcement Learning\Project'

param = {
    'UCB1': {'rho': np.sqrt(2)},
    'LinUCB': {'lbda': 10e-4, 'alpha': 10e-1},
    'BayesUCB': {'p1': 1, 'p2': 1, 'c': 0},
    'MOSS': {'rho': 0.2},
    'ExploreCommit': {'m': 10},
    'Tuned_GPUCB': {'c': 0.9},
    'IDS': {'M': 1000},
    'IDS_approx': {'N': 1000, 'display_results': False},
    'IDS_sample': {'M': 10000, 'VIDS': False},
    'VIDS_approx': {'rg': 10., 'N': 1000},
    'VIDS_sample': {'M': 10000, 'VIDS': True},
}

colors = {'KG': 'yellow', 'Approximate KG*':'orchid', 'KG*': 'orchid', 'Exact IDS':'chartreuse',
          'Thompson Sampling': 'blue', 'Bayes UCB': 'cyan', 'Tuned UCB': 'red', 'Linear UCB': 'yellow',
          'MOSS': 'black', 'GPUCB': 'black', 'Tuned GPUCB': 'red', 'Grid V-IDS': 'purple', 'Sample V-IDS': 'green',
          'Grid IDS': 'chartreuse', 'Sample IDS': 'orange'}

"""methods available : UCB1, TS, UCB_Tuned, BayesUCB, KG, KG_star, Approx_KG_star, MOSS, IDS, IDS_approx"""

finite_methods = ['UCB1', 'ExploreCommit', 'UCB_Tuned', 'MOSS']
# bernoulli_methods = ['IDS_approx', 'VIDS_sample', 'IDS_sample', 'TS', 'UCB_Tuned', 'BayesUCB', 'KG', 'Approx_KG_star', 'MOSS']
# gaussian_methods = ['TS', 'KG', 'BayesUCB', 'GPUCB', 'Tuned_GPUCB', 'VIDS_approx', 'VIDS_sample', 'KG_star']
# linear_methods = ['TS', 'LinUCB', 'BayesUCB', 'GPUCB', 'Tuned_GPUCB', 'VIDS_sample']

bernoulli_methods = ['TS', 'BayesUCB', 'VIDS_sample', 'IDS_sample']
gaussian_methods = ['VIDS_sample']
linear_methods = ['VIDS_sample']

"""Kind of Bandit problem"""
check_Finite = False
check_Bernoulli = True
check_Gaussian = False
check_Linear = False
store = False  # if you want to store the results

if __name__ == '__main__':
    if check_Finite:
        p, q, R = utils.build_finite(L=10000, K=20, N=500)
        labels = finite_methods
        exp.finite_expe(methods=finite_methods, labels=labels, colors=False,
                        param_dic=param, prior=p, q=q, R=R, N=2000, T=1000, theta=0)
    if check_Bernoulli:
        labels = bernoulli_methods
        colors = [colors[t] for t in labels]
        beta=exp.bernoulli_expe(T=10000, n_expe=200, n_arms=3, methods=bernoulli_methods,
                                param_dic=param, labels=labels, colors=False)
        if store:
            pkl.dump(beta, open(os.path.join(path, 'beta_asymp.pkl'), 'wb'))

    if check_Gaussian:
        labels = gaussian_methods
        colors = [colors[t] for t in labels]
        gau = exp.gaussian_expe(n_expe=1, n_arms=10, T=100, methods=gaussian_methods,
                                param_dic=param, labels=colors, colors=False)
        if store:
            pkl.dump(gau, open(os.path.join(path, 'gau.pkl'), 'wb'))

    if check_Linear:
        labels = linear_methods

        colors = [colors[t] for t in labels]
        lin = exp.LinMAB_expe(n_expe=20, n_features=0, n_arms=0, T=1000, methods=linear_methods, param_dic=param,
                              labels=labels, colors=colors, movieLens=True)
        if store:
            pkl.dump(lin, open(os.path.join(path, 'lin10features.pkl'), 'wb'))
