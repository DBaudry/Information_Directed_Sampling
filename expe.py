""" Packages import """
from MAB import GenericMAB
from BernoulliMAB import BetaBernoulliMAB
from GaussianMAB import GaussianMAB
from FiniteSetsMAB import FiniteSets
from LinMAB import PaperLinModel, ColdStartMovieLensModel, LinMAB
from utils import plotRegret, storeRegret, cmap
import matplotlib.pyplot as plt
import numpy as np


def bernoulli_expe(n_expe, n_arms, T, methods, param_dic, labels, colors, doplot=True):
    """
    Compute regrets for a given set of algorithms (methods) over t=1,...,T and for n_expe number of independent
    experiments. Here we deal with n_arms Bernoulli Bandits
    :param n_expe: int, number of experiments
    :param n_arms: int, number of arms
    :param T: int, time horizon
    :param methods: list, algorithms to use
    :param param_dic: dict, parameters associated to each algorithm (see main for formatting)
    :param labels: list, labels for the curves
    :param colors: list, colors for the curves
    :param doplot: boolean, plot the curves or not
    :return: dict, regrets, quantiles, means, stds of final regrets for each methods
    """
    P = np.random.uniform(0, 1, size=n_arms*n_expe).reshape(n_expe, n_arms)
    models = [BetaBernoulliMAB(p) for p in P]
    mean_regret, all_regrets, final_regrets, quantiles, means, std = storeRegret(models, methods, param_dic, n_expe, T)
    if doplot:
        plotRegret(labels, mean_regret, colors, 'Binary rewards')
    return {'all_regrets': all_regrets, 'quantiles': quantiles, 'means': means, 'std': std}


def gaussian_expe(n_expe, n_arms, T, methods, param_dic, labels, colors, doplot=True):
    """
    Compute regrets for a given set of algorithms (methods) over t=1,...,T and for n_expe number of independent
    experiments. Here we deal with n_arms Gaussian Bandits with Gaussian prior
    :param n_expe: int, number of experiments
    :param n_arms: int, number of arms
    :param T: int, time horizon
    :param methods: list, algorithms to use
    :param param_dic: dict, parameters associated to each algorithm (see main for formatting)
    :param labels: list, labels for the curves
    :param colors: list, colors for the curves
    :param doplot: boolean, plot the curves or not
    :return: dict, regrets, quantiles, means, stds of final regrets for each methods
    """
    mu = np.random.normal(0, 1, size=n_expe*n_arms).reshape(n_expe, n_arms)
    sigma = np.ones(n_arms*n_expe).reshape(n_expe, n_arms)
    P = [[[m[i], s[i]] for i in range(n_arms)] for m, s in zip(mu, sigma)]
    models = [GaussianMAB(p) for p in P]
    mean_regret, all_regrets, final_regrets, quantiles, means, std = storeRegret(models, methods, param_dic, n_expe, T)
    if doplot:
        plotRegret(labels, mean_regret, colors, 'Gaussian rewards')
    return {'all_regrets': all_regrets, 'quantiles': quantiles, 'means': means, 'std': std}


def finite_expe(methods, labels, colors, param_dic, prior, q, R, theta, N, T):
    """
    Compute regrets for a given set of algorithms (methods) over t=1,...,T and for n_expe number of independent
    experiments. Here we deal with Finite Set Problems
    :param methods: list, algorithms to use
    :param labels: list, labels for the curves
    :param colors: list, colors for the curves
    :param param_dic: dict, parameters associated to each algorithm (see main for formatting)
    :param prior: np.array, prior distribution on theta
    :param q: np.array, L*K*N array with the probability of each outcome knowing theta
    :param R: np.array, mapping between outcomes and rewards
    :param theta: float, true theta
    :param N: int, number of possible rewards
    :param T: int, time horizon
    """
    nb_arms, nb_rewards = q.shape[1:3]
    p2 = [[R, q[theta, i, :]] for i in range(q.shape[1])]
    check_MAB = GenericMAB(['F'] * nb_arms, p2)
    plt.figure(1)
    for i, m in enumerate(methods):
        c = cmap[i] if not colors else colors[i]
        r = check_MAB.MC_regret(m, N, T, param_dic)
        plt.plot(r, label=labels[i], c=c)
    p = [[np.arange(nb_rewards), q[theta, i, :]] for i in range(nb_arms)]
    my_MAB = FiniteSets(['F'] * nb_arms, p, q, prior, R)
    regret_IDS = my_MAB.MC_regret('IDS', N, T, param_dic)
    plt.plot(regret_IDS, label='IDS', c='cyan'); plt.ylabel('Cumulative Regret'); plt.xlabel('Time horizon')
    plt.grid(color='grey', linestyle='--', linewidth=0.5); plt.legend(); plt.show()


def LinMAB_expe(n_expe, n_features, n_arms, T, methods, param_dic, labels, colors, doplot=True, movieLens=False):
    """
    Compute regrets for a given set of algorithms (methods) over t=1,...,T and for n_expe number of independent
    experiments. Here we deal with n_arms Linear Gaussian Bandits with multivariate Gaussian prior
    :param n_expe: int, number of experiments
    :param n_features: int, dimension of feature vectors
    :param n_arms: int, number of arms
    :param T: int, time horizon
    :param methods: list, algorithms to use
    :param param_dic: dict, parameters associated to each algorithm (see main for formatting)
    :param labels: list, labels for the curves
    :param colors: list, colors for the curves
    :param doplot: boolean, plot the curves or not
    :param movieLens: boolean, if True uses ColdStartMovieLensModel otherwise PaperLinModel
    :return: dict, regrets, quantiles, means, stds of final regrets for each methods
    """
    if movieLens:
        models = [LinMAB(ColdStartMovieLensModel()) for _ in range(n_expe)]
        log = True
    else:
        u = 1 / np.sqrt(5)
        models = [LinMAB(PaperLinModel(u, n_features, n_arms, sigma=10)) for _ in range(n_expe)]
        log = False
    mean_regret, all_regrets, final_regrets, quantiles, means, std = storeRegret(models, methods, param_dic, n_expe, T)
    if doplot:
        plotRegret(labels, mean_regret, colors, 'Linear Gaussian Model', log=log)
    return {'all_regrets': all_regrets, 'quantiles': quantiles, 'means': means, 'std': std}