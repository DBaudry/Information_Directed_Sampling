""" Packages import """
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from tqdm import tqdm
import inspect

cmap = {0: 'black', 1: 'blue', 2: 'yellow', 3: 'green', 4: 'red', 5:'grey', 6:'purple', 7:'brown', 8:'pink', 9:'cyan'}

def rd_argmax(vector):
    """
    Compute random among eligible maximum indices
    :param vector: np.array
    :return: int, random index among eligible maximum indices
    """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return rd.choice(indices)


def display_results(delta, g, ratio, p_star):
    """
    Display quantities of interest in IDS algorithm
    """
    print('delta {}'.format(delta))
    print('g {}'.format(g))
    print('ratio : {}'.format(ratio))
    print('p_star {}'.format(p_star))


def plotRegret(labels, mean_regret, colors, title):
    """
    Plot Bayesian regret
    :param labels: list, list of labels for the different curves
    :param mean_regret: np.array, averaged regrets from t=1 to T for all methods
    :param colors: list, list of colors for the different curves
    :param title: string, plot's title
    """
    for i, l in enumerate(labels):
        c = cmap[i] if not colors else colors[i]
        plt.plot(mean_regret[i], c=c, label=l)
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.title(title)
    plt.ylabel('Cumulative regret')
    plt.xlabel('Time period')
    plt.legend()
    plt.show()


def storeRegret(models, methods, param_dic, n_expe, T):
    all_regrets = np.zeros((len(methods), n_expe, T))
    final_regrets = np.zeros((len(methods), n_expe))
    q, quantiles, means, std = np.linspace(0, 1, 21), {}, {}, {}
    for j in tqdm(range(n_expe)):
        model = models[j]
        for i, m in enumerate(methods):
            alg = model.__getattribute__(m)
            args = inspect.getfullargspec(alg)[0][2:]
            args = [T]+[param_dic[m][i] for i in args]
            all_regrets[i, j, :] = model.regret(alg(*args)[0], T)
        print({m: round(all_regrets[i, j, -1], 1) for i, m in enumerate(methods)})
        print({m + '_bar': np.mean(all_regrets[i, :(j + 1), -1]) for i, m in enumerate(methods)})

    for j, m in enumerate(methods):
        for i in range(n_expe):
            final_regrets[j, i] = all_regrets[j, i, -1]
            quantiles[m], means[m], std[m] = np.quantile(final_regrets[j, :], q), final_regrets[j,:].mean(), final_regrets[j, :].std()
    mean_regret = all_regrets.mean(axis=1)
    return mean_regret, all_regrets, final_regrets, quantiles, means, std


def build_finite(L, K, N):
    """
    Build automatically a finite bandit environment
    :param L: int, number of possible values for theta
    :param K: int, number of arms
    :param N: int, number of possible rewards
    :return: np.arrays, parameters required for launching an experiment with a finite bandit
    (prior, q values and R function)
    """
    R = np.linspace(0., 1., N)
    q = np.random.uniform(size=(L, K, N))
    for i in range(q.shape[0]):
        q[i] = np.apply_along_axis(lambda x: x / x.sum(), 1, q[i])
    p = np.random.uniform(0, 1, L)
    p = p / p.sum()
    return p, q, R
