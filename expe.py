""" Packages import """
from MAB import GenericMAB
from BernoulliMAB import BetaBernoulliMAB
from GaussianMAB import GaussianMAB
from FiniteSetsMAB import FiniteSets
from LinMAB import *
from utils import *
import matplotlib.pyplot as plt


def bernoulli_expe(n_expe, n_arms, T, methods, param_dic, labels, colors, doplot=True, frequentist=False):
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
    if frequentist is False:
        P = np.random.uniform(0, 1, size=n_arms*n_expe).reshape(n_expe, n_arms)
        models = [BetaBernoulliMAB(p) for p in P]
    else:
        p = frequentist
        models = [BetaBernoulliMAB(p)]*n_expe
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


def Finite_Bernoulli(n_expe, nb_arms, T, M, colors, doplot=False):
    theta = np.random.uniform(0, 1, size=nb_arms*n_expe).reshape(n_expe, nb_arms)
    true_param = [[[np.array([0, 1]), np.array([theta[i, j], 1-theta[i, j]])] for j in range(nb_arms)] for i
                  in range(n_expe)]
    prior, q, R = build_bernoulli_finite_set(M, nb_arms)
    all_regrets = np.empty((n_expe, T))
    for i in tqdm(range(n_expe)):
        my_MAB = FiniteSets(['F']*nb_arms, true_param[i], q, prior, R)
        all_regrets[i] = my_MAB.regret(my_MAB.IDS(T)[0], T)
    mean_regret = all_regrets.mean(axis=0).reshape((1, T))
    quantiles = {'Finite IDS': np.quantile(mean_regret, np.arange(0, 1, 21))}
    means = {'Finite IDS': mean_regret[-1]}
    std = {'Finite IDS': all_regrets.std(axis=0).reshape((1, T))}
    if doplot:
        plotRegret(['IDS with fixed parameter sample'], mean_regret, colors, 'Binary rewards')
    return {'all_regrets': all_regrets, 'quantiles': quantiles, 'means': means, 'std': std}