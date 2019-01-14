# File containing different experiments that we can run

# Importation
from MAB import GenericMAB
from BernoulliMAB import BetaBernoulliMAB
from GaussianMAB import GaussianMAB
from FiniteSetsMAB import FiniteSets
from LinMAB import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import inspect
from itertools import product

def plotRegret(methods, mean_regret, title):
    for i in range(len(methods)):
        plt.plot(mean_regret[i], label=methods[i])
    plt.title(title)
    plt.ylabel('Cumulative regret')
    plt.xlabel('Time period')
    plt.legend()
    plt.show()


def beta_bernoulli_expe(n_expe, n_arms, T, methods, param_dic, doplot=True):
    all_regrets, final_regrets = np.zeros((len(methods), n_expe, T)), np.zeros((len(methods), n_expe))
    q, quantiles, means, std = np.linspace(0,1,21), {}, {}, {}
    for j in tqdm(range(n_expe)):
        p = np.random.uniform(size=n_arms)
        my_mab = BetaBernoulliMAB(p)
        for i, m in enumerate(methods):
            alg = my_mab.__getattribute__(m)
            args = inspect.getfullargspec(alg)[0][2:]
            args = [T]+[param_dic[m][i] for i in args]
            all_regrets[i, j] = my_mab.regret(alg(*args)[0], T)
    for j, m in enumerate(methods):
        for i in range(n_expe):
            final_regrets[j, i] = all_regrets[j, i, -1]
            quantiles[m], means[m], std[m] = np.quantile(final_regrets[j, :], q), final_regrets[j,:].mean(), final_regrets[j, :].std()
    mean_regret = all_regrets.mean(axis=1)
    if doplot:
        plotRegret(methods, mean_regret, 'Binary rewards')
    return {'all_regrets': all_regrets, 'quantiles': quantiles, 'means': means, 'std': std}


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

    
##### Gaussian test ######

def check_gaussian(n_expe, n_arms, T, methods, param_dic, doplot=True):
    all_regrets = np.zeros((len(methods), n_expe, T))
    final_regrets = np.zeros((len(methods), n_expe))
    q, quantiles, means, std = np.linspace(0, 1, 21), {}, {}, {}
    for j in tqdm(range(n_expe)):
        mu, sigma, p = np.random.normal(0, 1, n_arms), np.ones(n_arms), []
        for i in range(len(mu)):
            p.append([mu[i], sigma[i]])
        my_mab = GaussianMAB(p)
        for i, m in enumerate(methods):
            alg = my_mab.__getattribute__(m)
            args = inspect.getfullargspec(alg)[0][2:]
            args = [T]+[param_dic[m][i] for i in args]
            all_regrets[i, j] = my_mab.regret(alg(*args)[0], T)
    for j, m in enumerate(methods):
        for i in range(n_expe):
            final_regrets[j, i] = all_regrets[j, i, -1]
            quantiles[m], means[m], std[m] = np.quantile(final_regrets[j, :], q), final_regrets[j,:].mean(), final_regrets[j, :].std()
    mean_regret = all_regrets.mean(axis=1)
    if doplot:
        plotRegret(methods, mean_regret, 'Gaussian rewards')
    return {'all_regrets': all_regrets, 'quantiles': quantiles, 'means': means, 'std': std}


##### Linear bandit test ######

def check_linearGaussianMAB(n_expe, n_features, n_arms, T, methods, param_dic, doplot=True, plotMAB=False):
    u = 1 / np.sqrt(5)
    all_regrets, final_regrets = np.zeros((len(methods), n_expe, T)), np.zeros((len(methods), n_expe))
    q, quantiles, means, std = np.linspace(0,1,21), {}, {}, {}
    for n in tqdm(range(n_expe)):
        random_state = np.random.randint(0, 312414)
        model = PaperLinModel(u, n_features, n_arms, sigma=10, random_state=random_state)
        if plotMAB:
            model.rewards_plot()
        lMAB = LinMAB(model)
        for i, m in enumerate(methods):
            alg = lMAB.__getattribute__(m)
            args = inspect.getfullargspec(alg)[0][2:]
            args = [T] + [param_dic[m][i] for i in args]
            reward, arm_sequence = alg(*args)
            all_regrets[i, n, :] = model.best_arm_reward() - reward
    for j, m in enumerate(methods):
        for i in range(n_expe):
            final_regrets[j, i] = all_regrets[j, i].sum()
        quantiles[m], means[m], std[m] = np.quantile(final_regrets[j, :], q), final_regrets[j, :].mean(), final_regrets[j, :].std()
    mean_regret = [np.array([np.mean(all_regrets[i, :, t]) for t in range(T)]).cumsum() for i in range(len(methods))]
    if doplot:
        plotRegret(methods, mean_regret, 'Linear-Gaussian Model')
    return {'all_regrets': all_regrets, 'quantiles': quantiles, 'means': means, 'std': std}



#### Influence of prior ####

def PriorInfLinMAB(n_expe, n_features, n_arms, T, methods, param_dic, doplot=True, nrow=1, ncol=2):
    u, rs = 1 / np.sqrt(5), np.random.randint(0, 312414, n_expe)
    L, S, q = [], [1, 5, 10, 20, 40], np.linspace(0,1,21) #:#np.linspace(1, 20, 20):
    regret, quantiles = np.zeros((len(S), len(methods), n_expe, T)),  np.zeros((len(methods), len(S), len(q)))
    for j, s in tqdm(enumerate(S), total=len(S), desc='Iterating over prior'):
        for n in tqdm(range(n_expe), '  Iterating over XPs'):
            random_state = rs[n]
            model = PaperLinModel(u, n_features, n_arms, sigma=10, random_state=random_state)
            lMAB = LinMAB(model, s=s)
            for i, m in enumerate(methods):
                alg = lMAB.__getattribute__(m)
                args = inspect.getfullargspec(alg)[0][2:]
                args = [T] + [param_dic[m][i] for i in args]
                reward, arm_sequence = alg(*args)
                regret[j, i, n, :] = model.best_arm_reward() - reward
    if doplot:
        plt.figure(1)
        for i, m in enumerate(methods):
            plt.subplot(nrow, ncol, i+1)
            plt.title('Method '+m)
            for j, s in enumerate(S):
                quantiles[i, j, :] = np.quantile([regret[j, i, n, :].sum() for n in range(n_expe)], q)
                plt.plot(np.array([np.mean(regret[j, i, :, t]) for t in range(T)]).cumsum(),
                         label=r'$\Sigma='+str(s)+' \cdot I_'+str(n_features)+'$')
                plt.ylabel('Cumulative regret')
                plt.xlabel('Time period')
                plt.legend()
        plt.show()
    return quantiles


def PriorInfGaussian(n_expe, n_arms, T, methods, param_dic, doplot=True, nrow=1, ncol=2):
    rs = np.random.randint(0, 312414, n_expe)
    L, S, q = [], [1, 5, 10, 20, 40], np.linspace(0,1,21)
    regret, quantiles = np.zeros((len(S), len(methods), n_expe, T)),  np.zeros((len(methods), len(S), len(q)))
    for j, s in tqdm(enumerate(S), total=len(S), desc='Iterating over prior'):
        for n in tqdm(range(n_expe), '  Iterating over XPs'):
            np.random.seed(rs[n])
            mu, sigma, p = np.random.normal(0, 1, n_arms), np.ones(n_arms), []
            for i in range(len(mu)):
                p.append([mu[i], sigma[i]])
            my_mab = GaussianMAB(p, s=s)
            for i, m in enumerate(methods):
                alg = my_mab.__getattribute__(m)
                args = inspect.getfullargspec(alg)[0][2:]
                args = [T] + [param_dic[m][i] for i in args]
                regret[j, i, n, :] = my_mab.regret(alg(*args)[0], T)
    if doplot:
        plt.figure(1)
        for i, m in enumerate(methods):
            plt.subplot(nrow, ncol, i+1)
            plt.title('Method '+m)
            for j, s in enumerate(S):
                quantiles[i, j, :] = np.quantile([regret[j, i, n, -1] for n in range(n_expe)], q)
                plt.plot(np.array([np.mean(regret[j, i, :, t]) for t in range(T)]), label=r'$\sigma='+str(s)+'$')
                plt.ylabel('Cumulative regret')
                plt.xlabel('Time period')
                plt.legend()
        plt.show()
    return quantiles

def PriorInfBer(n_expe, n_arms, T, N_steps=1000, doplot=True, legend=False):
    L, beta_1, beta_2, q = [], range(2, 11), range(2, 11), np.linspace(0,1,21)
    regret, quantiles = np.zeros((len(beta_1), len(beta_2), n_expe, T)), np.zeros((len(beta_1), len(beta_2), len(q)))
    for b in tqdm(product(beta_1, beta_2), total=len(beta_1)*len(beta_2), desc='Iterating over prior'):
        b1, b2 = b[0], b[1]
        beta_1_, beta_2_ = np.array([b1]*n_arms), np.array([b2]*n_arms)
        for n in tqdm(range(n_expe), '  Iterating over XPs'):
            p = np.random.uniform(size=n_arms)
            my_mab = BetaBernoulliMAB(p)
            regret[beta_1.index(b1), beta_2.index(b2), n, :] = my_mab.regret(my_mab.IDS_approx(T=T, N_steps=N_steps, beta1=beta_1_, beta2=beta_2_, display_results=False)[0], T)
    if doplot:
        plt.figure(1)
        b1_prev, j = -1, 0
        for b in product(beta_1, beta_2):
            b1, b2 = b[0], b[1]
            if b1 != b1_prev:
                plt.subplot(3, 3, j+1)
                j += 1
                b1_prev = b1
            quantiles[beta_1.index(b1), beta_2.index(b2), :] = np.quantile([regret[beta_1.index(b1), beta_2.index(b2), n, -1] for n in range(n_expe)], q)
            plt.plot(np.array([np.mean(regret[beta_1.index(b1), beta_2.index(b2), :, t]) for t in range(T)]), label=r'$\beta_2='+str(b2)+'$')
            plt.title(r'$\beta_1='+str(b1)+'$')
            plt.ylabel('Cumulative regret')
            plt.xlabel('Time period')
        if legend:
            plt.legend()
        plt.show()
    return quantiles


def GridSearchGaussian():
    rs = np.random.randint(0, 312414, n_expe)
    L, S, q = [], [1, 5, 10, 20, 40], np.linspace(0, 1, 21)
    regret, quantiles = np.zeros((len(S), len(methods), n_expe, T)), np.zeros((len(methods), len(S), len(q)))
    for j, s in tqdm(enumerate(S), total=len(S), desc='Iterating over prior'):
        for n in tqdm(range(n_expe), '  Iterating over XPs'):
            np.random.seed(rs[n])
            mu, sigma, p = np.random.normal(0, 1, n_arms), np.ones(n_arms), []
            for i in range(len(mu)):
                p.append([mu[i], sigma[i]])
            my_mab = GaussianMAB(p, s=s)
            for i, m in enumerate(methods):
                alg = my_mab.__getattribute__(m)
                args = inspect.getfullargspec(alg)[0][2:]
                args = [T] + [param_dic[m][i] for i in args]
                regret[j, i, n, :] = my_mab.regret(alg(*args)[0], T)
    if doplot:
        plt.figure(1)
        for i, m in enumerate(methods):
            plt.subplot(nrow, ncol, i + 1)
            plt.title('Method ' + m)
            for j, s in enumerate(S):
                quantiles[i, j, :] = np.quantile([regret[j, i, n, -1] for n in range(n_expe)], q)
                plt.plot(np.array([np.mean(regret[j, i, :, t]) for t in range(T)]), label=r'$\sigma=' + str(s) + '$')
                plt.ylabel('Cumulative regret')
                plt.xlabel('Time period')
                plt.legend()
        plt.show()
    return quantiles
