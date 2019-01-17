""" Packages import """
from MAB import GenericMAB
from BernoulliMAB import BetaBernoulliMAB
from GaussianMAB import GaussianMAB
from FiniteSetsMAB import FiniteSets
from LinMAB import *
from utils import *
import matplotlib.pyplot as plt


def bernoulli_expe(n_expe, n_arms, T, methods, param_dic, labels, colors, doplot=True):
    """

    :param n_expe: int, number of experiments
    :param n_arms: int, number of arms
    :param T: int, time horizon
    :param methods:
    :param param_dic:
    :param labels:
    :param colors:
    :param doplot:
    :return:
    """
    P = np.random.uniform(0, 1, size=n_arms*n_expe).reshape(n_expe, n_arms)
    models = [BetaBernoulliMAB(p) for p in P]
    mean_regret, all_regrets, final_regrets, quantiles, means, std = storeRegret(models, methods, param_dic, n_expe, T)
    if doplot:
        plotRegret(labels, mean_regret, colors, 'Binary rewards')
    return {'all_regrets': all_regrets, 'quantiles': quantiles, 'means': means, 'std': std}


def gaussian_expe(n_expe, n_arms, T, methods, param_dic, labels, colors, doplot=True):
    """

    :param n_expe:
    :param n_arms:
    :param T:
    :param methods:
    :param param_dic:
    :param labels:
    :param colors:
    :param doplot:
    :return:
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

    :param methods:
    :param labels:
    :param colors:
    :param param_dic:
    :param prior:
    :param q:
    :param R:
    :param theta:
    :param N:
    :param T:
    :return:
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

    :param n_expe:
    :param n_features:
    :param n_arms:
    :param T:
    :param methods:
    :param param_dic:
    :param labels:
    :param colors:
    :param doplot:
    :return:
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
        plotRegret(labels, mean_regret, colors, 'Gaussian rewards', log=log)
    return {'all_regrets': all_regrets, 'quantiles': quantiles, 'means': means, 'std': std}