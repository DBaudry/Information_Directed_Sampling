""" Packages import """
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from tqdm import tqdm
import inspect

cmap = {0: 'black', 1: 'blue', 2: 'yellow', 3: 'green', 4: 'red', 5:'grey', 6:'purple', 7:'brown', 8:'pink', 9:'cyan'}

mapping_methods_labels = {'KG': 'KG',
                          'Approx_KG_star': 'Approximate KG*',
                          'KG_star': 'KG*',
                          'IDS': 'Exact IDS',
                          'TS': 'Thompson Sampling',
                          'BayesUCB': 'Bayes UCB',
                          'UCB_Tuned': 'Tuned UCB',
                          'LinUCB': 'Linear UCB',
                          'MOSS': 'MOSS',
                          'GPUCB': 'GP-UCB',
                          'Tuned_GPUCB': 'Tuned GP-UCB',
                          'VIDS_approx': 'Grid V-IDS',
                          'VIDS_sample': 'Sample V-IDS',
                          'IDS_approx': 'Grid IDS',
                          'IDS_sample': 'Sample IDS',
                          'UCB1': 'UCB1'}

mapping_methods_colors = {'KG': 'yellow',
                          'Approx_KG_star':'orchid',
                          'KG_star': 'orchid',
                          'IDS':'chartreuse',
                          'TS': 'blue',
                          'BayesUCB': 'cyan',
                          'UCB_Tuned': 'red',
                          'LinUCB': 'yellow',
                          'MOSS': 'black',
                          'GPUCB': 'black',
                          'Tuned_GPUCB': 'red',
                          'VIDS_approx': 'purple',
                          'VIDS_sample': 'green',
                          'IDS_approx': 'chartreuse',
                          'IDS_sample': 'orange',
                          'UCB1': 'darkred'}


def labelColor(methods):
    """
    Map methods to labels and colors for regret curves
    :param methods: list, list of methods
    :return: lists, labels and vectors
    """
    labels = [mapping_methods_labels[m] for m in methods]
    colors = [mapping_methods_colors[m] for m in methods]
    return labels, colors


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


def plotRegret(labels, mean_regret, colors, title, log=False):
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
        if log:
            plt.yscale('log')
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
    results = {'mean_regret': mean_regret, 'all_regrets': all_regrets,
               'final_regrets': final_regrets, 'quantiles': quantiles,
               'means': means, 'std': std}
    if models[0].store_IDS:
        IDS_res = [m.IDS_results for m in models]
        results['IDS_results'] = IDS_res
    return results


def plot_IDS_results(T, n_expe, results):
    delta = np.empty((T, n_expe))
    g = np.empty((T, n_expe))
    IR = np.empty((T, n_expe))
    for i, r in enumerate(results):
        r['policy'] = np.asarray(r['policy'])
        delta[:r['policy'].shape[0], i] = (np.asarray(r['delta'])*r['policy']).sum(axis=1)
        g[:r['policy'].shape[0], i] = (np.asarray(r['g'])*r['policy']).sum(axis=1)
        IR[:r['policy'].shape[0], i] = np.asarray(r['IR'])
    x = np.arange(1, T+1)
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
    plt.xlabel('Time horizon')
    ax1.plot(x, delta.mean(axis=1))
    ax1.fill_between(x, np.quantile(delta, 0.05, axis=1), np.quantile(delta, 0.95, axis=1), color='lightblue')
    ax2.plot(x, g.mean(axis=1))
    ax2.fill_between(x, np.quantile(g, 0.05, axis=1), np.quantile(g, 0.95, axis=1), color='lightblue')
    ax3.plot(x, IR.mean(axis=1), color='r')
    ax3.fill_between(x, np.quantile(IR, 0.05, axis=1), np.quantile(IR, 0.95, axis=1), color='#FFA07A')
    ax1.set_title('Average Delta')
    ax2.set_title('Average Information Gain')
    ax3.set_title('Average Information Ratio')
    plt.show()


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


def build_bernoulli_finite_set(L, K):
    r = np.array([0, 1])
    p = np.ones(L)/L
    q = np.empty((L, K, 2))
    q[:, :, 0] = np.random.uniform(size=L*K).reshape((L, K))
    q[:, :, 1] = 1-q[:, :, 0]
    return p, q, r
