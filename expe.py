""" Packages import """
from MAB import GenericMAB
from BernoulliMAB import BetaBernoulliMAB
from GaussianMAB import GaussianMAB
from FiniteSetsMAB import FiniteSets
from LinMAB import *
from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import inspect


def bernoulli_expe(n_expe, n_arms, T, methods, param_dic, labels, colors, doplot=True):
    P = np.repeat(np.random.uniform(size=n_arms), n_expe).reshape(n_expe, n_arms)
    models = [BetaBernoulliMAB(p) for p in P]
    mean_regret, all_regrets, final_regrets, quantiles, means, std = storeRegret(models, methods, param_dic, n_expe, T)
    if doplot:
        plotRegret(labels, mean_regret, colors, 'Binary rewards')
    return {'all_regrets': all_regrets, 'quantiles': quantiles, 'means': means, 'std': std}


def gaussian_expe(n_expe, n_arms, T, methods, param_dic, labels, colors, doplot=True):
    mu = np.repeat(np.random.normal(0, 1, size=n_arms), n_expe).reshape(n_expe, n_arms)
    sigma = np.repeat(np.ones(n_arms), n_expe).reshape(n_expe, n_arms)
    P = [[[m[i], s[i]] for i in range(n_arms)] for m, s in zip(mu, sigma)]
    models = [GaussianMAB(p) for p in P]
    mean_regret, all_regrets, final_regrets, quantiles, means, std = storeRegret(models, methods, param_dic, n_expe, T)
    if doplot:
        plotRegret(labels, mean_regret, colors, 'Gaussian rewards')
    return {'all_regrets': all_regrets, 'quantiles': quantiles, 'means': means, 'std': std}


def finite_expe(methods, labels, colors, param_dic, prior, q, R, theta, N, T):
    nb_arms, nb_rewards = q.shape[1:3]
    p2 = [[R, q[theta, i, :]] for i in range(q.shape[1])]
    check_MAB = GenericMAB(['F'] * nb_arms, p2)
    plt.figure(1)
    for i, m in enumerate(methods):
        r = check_MAB.MC_regret(m, N, T, param_dic)
        plt.plot(r, label=labels[i], colors=colors[i])
    p = [[np.arange(nb_rewards), q[theta, i, :]] for i in range(nb_arms)]
    my_MAB = FiniteSets(['F'] * nb_arms, p, q, prior, R)
    regret_IDS = my_MAB.MC_regret('IDS', N, T, param_dic)
    plt.plot(regret_IDS, label='IDS', colors='cyan'); plt.ylabel('Cumulative Regret'); plt.xlabel('Time horizon')
    plt.grid(color='grey', linestyle='--', linewidth=0.5); plt.legend(); plt.show()


def LinMAB_expe(n_expe, n_features, n_arms, T, methods, param_dic, labels, colors, doplot=True):
    u = 1 / np.sqrt(5)
    models = [LinMAB(PaperLinModel(u, n_features, n_arms, sigma=10)) for _ in range(n_expe)]
    mean_regret, all_regrets, final_regrets, quantiles, means, std = storeRegret(models, methods, param_dic, n_expe, T)
    if doplot:
        plotRegret(labels, mean_regret, colors, 'Gaussian rewards')
    return {'all_regrets': all_regrets, 'quantiles': quantiles, 'means': means, 'std': std}

    #
    # all_regrets, final_regrets = np.zeros((len(methods), n_expe, T)), np.zeros((len(methods), n_expe))
    # q, quantiles, means, std = np.linspace(0,1,21), {}, {}, {}
    # for j in tqdm(range(n_expe)):
    #     model = PaperLinModel(u, n_features, n_arms, sigma=10)
    #     lMAB = LinMAB(model)
    #     for i, m in enumerate(methods):
    #         alg = lMAB.__getattribute__(m)
    #         args = inspect.getfullargspec(alg)[0][2:]
    #         args = [T] + [param_dic[m][i] for i in args]
    #         reward, arm_sequence = alg(*args)
    #         all_regrets[i, j] = model.best_arm_reward() - reward

    # for j, m in enumerate(methods):
    #     for i in range(n_expe):
    #         final_regrets[j, i] = all_regrets[j, i].sum()
    #     quantiles[m], means[m], std[m] = np.quantile(final_regrets[j, :], q), final_regrets[j, :].mean(), final_regrets[j, :].std()
    # mean_regret = [np.array([np.mean(all_regrets[i, :, t]) for t in range(T)]).cumsum() for i in range(len(methods))]
    # if doplot:
    #     plotRegret(methods, mean_regret, 'Linear-Gaussian Model')
    # return {'all_regrets': all_regrets, 'quantiles': quantiles, 'means': means, 'std': std}
