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

"""methods available : UCB1, TS, UCB_Tuned, BayesUCB, KG, KG_star, Approx_KG_star, MOSS, IDS, IDS_approx"""

finite_methods = ['UCB1', 'ExploreCommit', 'UCB_Tuned', 'MOSS']
bernoulli_methods = ['VIDS_sample', 'IDS_sample', 'TS', 'UCB_Tuned', 'BayesUCB', 'KG', 'Approx_KG_star', 'MOSS']
gaussian_methods = ['TS', 'KG', 'BayesUCB', 'GPUCB', 'Tuned_GPUCB', 'VIDS_sample', 'KG_star']
linear_methods = ['TS', 'LinUCB', 'BayesUCB', 'GPUCB', 'Tuned_GPUCB', 'VIDS_sample']


"""Kind of Bandit problem"""
check_Finite = True
check_Bernoulli = True
check_Gaussian = True
check_Linear = True
store = False # if you want to store the results
check_time = False


if __name__ == '__main__':
    if check_Finite:
        p, q, R = utils.build_finite(L=100, K=10, N=100)
        exp.finite_expe(methods=finite_methods, labels=finite_methods, colors=False,
                        param_dic=param, prior=p, q=q, R=R, theta=0, N=10, T=500)

    if check_Bernoulli:
        labels, colors = utils.labelColor(bernoulli_methods)
        beta = exp.bernoulli_expe(T=200, n_expe=10, n_arms=3, methods=bernoulli_methods,
                                param_dic=param, labels=labels,
                                  colors=colors, doplot=False, track_ids=False)
        if store:
            pkl.dump(beta, open(os.path.join(path, 'beta_IR.pkl'), 'wb'))

    if check_Gaussian:
        labels, colors=utils.labelColor(gaussian_methods)
        gau = exp.gaussian_expe(n_expe=10, n_arms=3, T=500, methods=gaussian_methods,
                                param_dic=param, labels=gaussian_methods, colors=False, track_ids=False)

        if store:
            pkl.dump(gau, open(os.path.join(path, 'gau.pkl'), 'wb'))

    if check_Linear:
        labels, colors = utils.labelColor(linear_methods)
        lin = exp.LinMAB_expe(n_expe=10, n_features=5, n_arms=10, T=100, methods=linear_methods, param_dic=param,
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
