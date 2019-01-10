# File containing different experiments that we can run

# Importation
import numpy as np
from MAB import GenericMAB
from BernoulliMAB import BetaBernoulliMAB
from GaussianMAB import GaussianMAB
from FiniteSetsMAB import FiniteSets
from LinMAB import *
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

np.random.seed(42)

default_param = {
    'UCB1': 0.2,
    'BayesUCB': (1, 1),
    'MOSS': 0.2,
    'ExploreCommit': 50,
    'IDS_approx': 1000,
    'GPUCB' : 0.99
}


def beta_bernoulli_expe(n_expe, n_arms, T, methods=['KG', 'KG2'], param=default_param,
                        doplot=True):
    # 'UCB_Tuned', 'Bayes_UCB', 'KG', 'TS', 'Approx_KG*', 'UCB1', 'MOSS', 'IDS_approx'
    all_regrets = np.zeros((len(methods), n_expe, T))
    res = {}
    for j in tqdm(range(n_expe)):
        p = np.random.uniform(size=n_arms)
        my_mab = BetaBernoulliMAB(p)
        #res['UCB1'] = my_mab.regret(my_mab.UCB1(T, rho=param['UCB1'])[0], T)
        #res['TS'] = my_mab.regret(my_mab.TS(T)[0], T)
        #res['UCB_Tuned'] = my_mab.regret(my_mab.UCB_Tuned(T)[0], T)
        #res['Bayes_UCB'] = my_mab.regret(my_mab.BayesUCB(T, p1=param['BayesUCB'][0], p2=param['BayesUCB'][1])[0], T)
        #res['MOSS'] = my_mab.regret(my_mab.MOSS(T, rho=param['MOSS'])[0], T)
        res['KG'] = my_mab.regret(my_mab.KG(T)[0], T)
        res['KG2'] = my_mab.regret(my_mab.KG2(T)[0], T)
        #res['Approx_KG*'] = my_mab.regret(my_mab.Approx_KG_star(T)[0], T)
        #res['IDS_approx'] = my_mab.regret(my_mab.IDS_approx(T, N_steps=param['IDS_approx'])[0], T)
        for i, m in enumerate(methods):
            all_regrets[i, j] = res[m]
    MC_regret = all_regrets.sum(axis=1)/n_expe
    if doplot:
        for i in range(len(methods)):
            plt.plot(MC_regret[i], label=methods[i])
        plt.ylabel('Cumulative Regret')
        plt.xlabel('Rounds')
        plt.legend()
        plt.show()
    return MC_regret


def build_finite(L, K, N):
    """
    Building automatically a finite bandit environment
    :param L: Number of possible values for theta
    :param K: Number of arms
    :param N: Number of possible rewards
    :return: Parameters required for launching an experiment with a finite bandit (prior, q values and R function)
    """
    R = np.linspace(0., 1., N)
    q = np.random.uniform(size=(L, K, N))
    for i in range(q.shape[0]):
        q[i] = np.apply_along_axis(lambda x: x / x.sum(), 1, q[i])
        # For a given theta and a given arm, the sum over the reward should be one
    p = np.random.uniform(0, 1, L)  # In case of a random prior
    p = p / p.sum()
    return p, q, R


def build_finite_deterministic():
    """
    Building a given finite MAB with 2 possible values for theta, 5 arms and eleven different rewards
    :return: Parameters required for launching an experiment with a finite bandit (prior, q values and R function)
    """
    R = np.linspace(0., 1., 11)
    q = np.random.uniform(size=(2, 5, 11))
    for i in range(q.shape[0]):
        q[i] = np.apply_along_axis(lambda x: x / x.sum(), 1, q[i])
    p = np.array([0.35, 0.65])
    return p, q, R


def check_finite(prior, q, R, theta, N, T):
    nb_arms = q.shape[1]
    nb_rewards = q.shape[2]
    method = ['F'] * nb_arms
    param = [[np.arange(nb_rewards), q[theta, i, :]] for i in range(nb_arms)]
    my_MAB = FiniteSets(method, param, q, prior, R)
    print('prior: ', prior)
    print('Reward: ', R)
    print('Theta_a: ', my_MAB.Ta)
    param2 = [[R, q[theta, i, :]] for i in range(q.shape[1])]
    check_MAB = GenericMAB(method, param2)
    regret_IDS = my_MAB.MC_regret(method='IDS', N=N, T=T)
    plt.plot(regret_IDS, label='IDS')
    plt.plot(check_MAB.MC_regret(method='UCB1', N=N, T=T, param=0.2), label='UCB1')
    plt.plot(check_MAB.MC_regret(method='TS', N=N, T=T), label='TS')
    plt.ylabel('Cumulative Regret')
    plt.xlabel('Rounds')
    plt.legend()
    plt.show()


def sanity_check_expe():
    p1 = [0.05, 0.4, 0.7, 0.90]
    p2 = [0.25, 0.27, 0.32, 0.40, 0.42]
    N1 = 50
    plt.figure(1)
    for i, p in enumerate([p1, p2]):
        my_MAB = BetaBernoulliMAB(p)
        print(my_MAB.MAB)
        print(my_MAB.Cp)
        plt.subplot(121 + i)
        plt.plot(my_MAB.MC_regret(method='UCB1', N=N1, T=1000, param=0.2), label='UCB1')
        plt.plot(my_MAB.MC_regret(method='Random', N=N1, T=1000, param=0.), label='Random')
        plt.plot(my_MAB.MC_regret(method='TS', N=N1, T=1000), label='TS')
        plt.plot(my_MAB.Cp * np.log(np.arange(1, 1001)))
        plt.ylabel('Cumulative Regret')
        plt.xlabel('Rounds')
        plt.legend()
    plt.show()

##### Gaussian test ######

def check_gaussian(n_expe, n_arms, T, methods=['TS', 'KG*', 'IDS_approx'], param=default_param,
                   doplot=True):
    # 'UCB1', 'GPUCB', 'Tuned_GPUCB', 'BayesUCB', 'KG', 'KG*'
    all_regrets = np.zeros((len(methods), n_expe, T))
    res = {}
    for j in tqdm(range(n_expe)):
        mu = np.random.normal(0, 1, n_arms)
        sigma = [1] * len(mu)
        #print(mu, sigma)
        p = []
        for i in range(len(mu)):
            p.append([mu[i], sigma[i]])
        my_mab = GaussianMAB(p)
        res['TS'] = my_mab.regret(my_mab.TS(T)[0], T)
        #res['BayesUCB'] = my_mab.regret(my_mab.BayesUCB(T, p1=param['BayesUCB'][0], p2=param['BayesUCB'][0])[0], T)
        #res['UCB1'] = my_mab.regret(my_mab.UCB1(T, rho=param['UCB1'])[0], T)
        #res['GPUCB'] = my_mab.regret(my_mab.GPUCB(T)[0], T)
        #res['Tuned_GPUCB'] = my_mab.regret(my_mab.Tuned_GPUCB(T, c = param['GPUCB'])[0], T)
        #res['KG'] = my_mab.regret(my_mab.KG(T)[0], T)
        res['KG*'] = my_mab.regret(my_mab.KG_star(T)[0], T)
        #res['IDS'] = my_mab.regret(my_mab.IDS(T)[0], T)
        res['IDS_approx'] = my_mab.regret(my_mab.IDS_approx(T, N_steps=param['IDS_approx'])[0], T)
        #res['IDS'] = my_mab.regret(my_mab.IDS(T)[0], T)
        for i, m in enumerate(methods):
            all_regrets[i, j] = res[m]
    MC_regret = all_regrets.sum(axis=1)/n_expe
    if doplot:
        for i in range(len(methods)):
            plt.plot(MC_regret[i], label=methods[i])
        plt.ylabel('Cumulative Regret')
        plt.xlabel('Rounds')
        plt.legend()
        plt.show()
    return MC_regret

def approxIntegral():
    mu = np.random.uniform(-100, 100, 9)
    sigma = np.random.uniform(0, 100, 9)
    nb_arms = 9

    def joint_cdf(x):
        result = 1.
        for a in range(nb_arms):
            result = result * norm.cdf(x, mu[a], sigma[a])
        return result

    def dp_star(x, a):
        return joint_cdf(x) / norm.cdf(x, mu[a], sigma[a])*norm.pdf(x, mu[a], sigma[a])

    Y, X = [[] for _ in range(nb_arms)], [[] for _ in range(nb_arms)]
    for a in tqdm(range(nb_arms), desc='Computing dp_star for all actions'):
        x_sup = np.max([np.max(mu)+3*np.max(sigma), mu[a]+3*sigma[a]])
        x_inf = np.min([np.min(mu)-3*np.max(sigma), mu[a]-3*sigma[a]])
        X0 = np.linspace(x_sup, x_inf, 100)
        Y0 = np.array([dp_star(x, a) for x in X0])
        I_a = np.arange(100) if np.max(Y0) < 10e-10 else np.where(Y0 >= 10e-10)
        X1 = np.linspace(X0[I_a][0], X0[I_a][-1], 1000)
        Y[a] = np.array([dp_star(x, a) for x in X1])
        X[a] = X1

    plt.figure(1)
    for a in range(nb_arms):
        plt.subplot(3, 3, a+1)
        plt.plot(X[a], Y[a], label='action '+str(a))
        plt.legend()
    plt.show()


def LinearGaussianMAB(n_expe, n_features, n_arms, T, plot=True, plotMAB=False):
    methods = ['LinUCB', 'Tuned_GPUCB', 'GPUCB', 'TS', 'BayesUCB', 'IDS']
    u = 1 / np.sqrt(5)
    regret = np.zeros((len(methods), n_expe, T))
    for n in tqdm(range(n_expe)):
        random_state = np.random.randint(0, 312414)
        model = PaperLinModel(u, n_features, n_arms, sigma=10, random_state=random_state)
        if plotMAB:
            model.rewards_plot()
        lMAB = LinMAB(model)
        for i, m in enumerate(methods):
            alg = lMAB.__getattribute__(m)
            reward, arm_sequence = alg(T)
            regret[i, n, :] = model.best_arm_reward() - reward
    mean_regret = [np.array([np.mean(regret[i, :, t]) for t in range(T)]).cumsum() for i in range(len(methods))]
    if plot:
        for i, m in enumerate(methods):
            plt.plot(mean_regret[i], label=m)
        plt.legend()
        plt.show()

LinearGaussianMAB(10, 5, 25, 250)