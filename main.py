""" Packages import """
import expe as exp
import numpy as np
import pickle as pkl
import os
import utils
import time


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

colors = {'KG': 'yellow', 'Approximate KG*': 'orchid', 'KG*': 'orchid', 'Exact IDS':'chartreuse',
          'Thompson Sampling': 'blue', 'Bayes UCB': 'cyan', 'Tuned UCB': 'red', 'Linear UCB': 'yellow',
          'MOSS': 'black', 'GPUCB': 'black', 'Tuned GPUCB': 'red', 'Grid V-IDS': 'purple', 'Sample V-IDS': 'green',
          'Grid IDS': 'chartreuse', 'Sample IDS': 'orange'}

"""methods available : UCB1, TS, UCB_Tuned, BayesUCB, KG, KG_star, Approx_KG_star, MOSS, IDS, IDS_approx"""

finite_methods = ['UCB1', 'ExploreCommit', 'UCB_Tuned', 'MOSS']
# bernoulli_methods = ['IDS_approx', 'VIDS_sample', 'IDS_sample', 'TS', 'UCB_Tuned', 'BayesUCB', 'KG', 'Approx_KG_star', 'MOSS']
# gaussian_methods = ['TS', 'KG', 'BayesUCB', 'GPUCB', 'Tuned_GPUCB', 'VIDS_approx', 'VIDS_sample', 'KG_star']
# linear_methods = ['TS', 'LinUCB', 'BayesUCB', 'GPUCB', 'Tuned_GPUCB', 'VIDS_sample']

bernoulli_methods = ['IDS_sample']
gaussian_methods = ['VIDS_sample']
linear_methods = ['VIDS_sample']

"""Kind of Bandit problem"""
check_Finite = False
check_Bernoulli = True
check_Gaussian = False
check_Linear = False
store = False # if you want to store the results
check_time = False


if __name__ == '__main__':
    if check_Finite:
        p, q, R = utils.build_finite(L=10000, K=20, N=500)
        labels = finite_methods
        exp.finite_expe(methods=finite_methods, labels=labels, colors=False, param_dic=param, prior=p, q=q, R=R, theta=0, N=100, T=1000)

    if check_Bernoulli:
        labels = bernoulli_methods
        beta = exp.bernoulli_expe(T=2000, n_expe=1000, n_arms=3, methods=bernoulli_methods,
                                param_dic=param, labels=labels,
                                  colors=False, doplot=False, track_ids=True)
        if store:
            pkl.dump(beta, open(os.path.join(path, 'beta_IR.pkl'), 'wb'))

    if check_Gaussian:
        labels = gaussian_methods
        # colors = [colors[t] for t in labels]
        t0 = time.time()
        gau = exp.gaussian_expe(n_expe=300, n_arms=10, T=1000, methods=gaussian_methods,
                                param_dic=param, labels=gaussian_methods, colors=False, track_ids=True)

        if store:
            pkl.dump(gau, open(os.path.join(path, 'gau.pkl'), 'wb'))

    if check_Linear:
        labels, colors = utils.labelColor(linear_methods)
        lin = exp.LinMAB_expe(n_expe=1, n_features=5, n_arms=10, T=100, methods=linear_methods, param_dic=param,
                              labels=labels, colors=colors, movieLens=False)
        if store:
            pkl.dump(lin, open(os.path.join(path, 'lin10features.pkl'), 'wb'))

    if check_time:
        import LinMAB as LM
        for i, j in zip([15, 30, 50, 100], [3, 5, 20, 30]):
            model = LM.PaperLinModel(u=np.sqrt(1/5), n_features=j, n_actions=i)
            model.threshold = 0.999
            alg = LM.LinMAB(model)
            t = time.time()
            for _ in range(100):
                model.flag = False
                alg.VIDS_sample(T=1000, M=10000)
            print((time.time()-t)/1000/100)
