""" Packages import """
from MAB import *
import utils
from scipy.stats import beta
from copy import copy

class BetaBernoulliMAB(GenericMAB):
    """
    Bernoulli Bandit Problem
    """
    def __init__(self, p):
        """
        Initialization
        :param p: np.array, true probabilities of success for each arm
        """
        # Initialization of arms from GenericMAB
        super().__init__(methods=['B']*len(p), p=p)
        # Complexity
        self.Cp = sum([(self.mu_max-x)/self.kl(x, self.mu_max) for x in self.means if x != self.mu_max])
        # Parameters used for stop learning policy
        self.flag = False
        self.optimal_arm = None
        self.threshold = 0.99

    @staticmethod
    def kl(x, y):
        """
        Implementation of the Kullback-Leibler divergence for two Bernoulli distributions (B(x),B(y))
        :param x: float
        :param y: float
        :return: float, KL(B(x), B(y))
        """
        return x * np.log(x/y) + (1-x) * np.log((1-x)/(1-y))

    def init_prior(self, a0=1, a1=1):
        """
        Init Beta prior
        :param a0: int, multiplicative factor for alpha
        :param a1: int, multiplicative factor for beta
        :return: np.arrays, prior values (alpha, beta) for each earm
        """
        beta1 = a0 * np.ones(self.nb_arms).astype(int)
        beta2 = a1 * np.ones(self.nb_arms).astype(int)
        return beta1, beta2

    def TS(self, T):
        """
        Implementation of Thomson Sampling (TS) algorithm for Bernoulli Bandit Problems with beta prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        theta = np.zeros(self.nb_arms)
        for t in range(T):
            for k in range(self.nb_arms):
                theta[k] = np.random.beta(Sa[k]+1, Na[k]-Sa[k]+1) if Na[k] >= 1 else np.random.uniform()
            arm = rd_argmax(theta)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence

    def BayesUCB(self, T, p1, p2, c=0):
        """
        Implementation of Bayesian Upper Confidence Bounds (BayesUCB) algorithm for Bernoulli Bandit Problems
        with beta prior
        :param T: int, time horizon
        :param p1: float, first parameter of the Beta prior probability distribution
        :param p2: float, second parameter of the Beta prior probability distribution
        :param c: float, parameter for the quantiles. Default value c=0
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        quantiles = np.zeros(self.nb_arms)
        for t in range(T):
            for k in range(self.nb_arms):
                if Na[k] >= 1:
                    quantiles[k] = beta.ppf(1-1/((t+1)*np.log(T)**c), Sa[k] + p1, p2 + Na[k] - Sa[k])
                else:
                    quantiles[k] = beta.ppf(1-1/((t+1)*np.log(T)**c), p1, p2)
            arm = rd_argmax(quantiles)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence

    def IR_approx(self, N, b1, b2, X, f, F, G, g):
        """
        Compute delta and g for Bernoulli Bandit Problems with beta prior
        :param N: int, number of points to take in the [0,1] interval
        :param b1: np.array, first parameter of the beta distribution for each arm
        :param b2: np.array, second parameter of the beta distribution for each arm
        :param X: np.array, grid on [0, 1]
        :param f: np.array, pdf of each arm
        :param F: np.array, cdf of each arm
        :param G: np.array, G function of each arm as defined in Russo & Van Roy (algorithm 2 page 241)
        :param g: np.array, information gain
        :return: np.arrays, the two components of the Information ratio delta and g with probability p*
        of choosing optimal arm and maap defined as M(a|a') in Russo & Van Roy's paper
        """
        maap = np.zeros((self.nb_arms, self.nb_arms))
        p_star = np.zeros(self.nb_arms)
        prod_F1 = np.ones((self.nb_arms, self.nb_arms, N))
        for a in range(self.nb_arms):
            for ap in range(self.nb_arms):
                for app in range(self.nb_arms):
                    if a != app and app != ap:
                        prod_F1[a, ap] *= F[app]
                prod_F1[a, ap] *= f[a]/N
        for a in range(self.nb_arms):
            p_star[a] = (prod_F1[a, a]).sum()
            for ap in range(self.nb_arms):
                if a != ap:
                    maap[ap, a] = (prod_F1[a, ap]*G[ap]).sum()/p_star[a]
                else:
                    maap[a, a] = (prod_F1[a, a]*X).sum()/p_star[a]
        rho_star = np.inner(np.diag(maap), p_star)
        delta = rho_star - b1/(b1+b2)
        for arm in range(self.nb_arms):
            sum_log = maap[arm]*np.log(maap[arm]*(b1+b2)/b1) + (1-maap[arm])*np.log((1-maap[arm])*(b1+b2)/b2)
            g[arm] = np.inner(p_star, sum_log)
        return delta, g, p_star, maap

    @staticmethod
    def gamma_function(c):
        """
        Computation of gamma coefficient for beta distribution
        :param c: int
        :return: list, gamma(i) for i from 0 to c
        """
        l = np.ones(c+1)
        for i in range(c):
            l[i+1] = (i+1)*l[i]
        return l

    def init_approx(self, N, beta1, beta2):
        """
        Initialization of quantities of interest for IDS_approx algorithm
        :param N: int, number of points to take in the [0,1] interval
        :param beta1: np.array, alpha values for each arm with posterior Beta(alpha, beta)
        :param beta2: np.array, beta values for each arm with posterior Beta(alpha, beta)
        :return: np.arrays, initialization of the arrays for the approximation of the integrals in IDS_approx
        """
        fact = self.gamma_function(int((beta1+beta2).max()))
        B = np.array([fact[beta1[i]-1]*fact[beta2[i]-1]/fact[beta1[i]+beta2[i]-1] for i in range(self.nb_arms)])
        X = np.linspace(1 / N, 1., N)
        f = np.array([X ** (beta1[i] - 1) * (1. - X) ** (beta2[i] - 1) / B[i] for i in range(self.nb_arms)])
        F = (f / N).cumsum(axis=1)
        G = (f * X / N).cumsum(axis=1)
        maap, p_star = np.zeros((self.nb_arms, self.nb_arms)), np.zeros(self.nb_arms)
        prod_F1, g = np.ones((self.nb_arms, self.nb_arms, N)), np.zeros(self.nb_arms)
        return X, f, F, G, B, maap, p_star, prod_F1, g

    @staticmethod
    def update_approx(arm, y, beta, X, f, F, G, B):
        """
        Update of all functions with recursion formula. These formulas are derived using the properties of the beta
        distribution: the pdf and cdf of beta(a, b) can be used to compute the cdf and pdf of beta(a+1, b) and
        beta(a, b+1)
        :param arm: int, arm chose
        :param y: float, reward obtained after pulling the arm
        :param beta: np.array, alpha and beta values of the arm's posterior distribution
        :param X: np.array, grid on [0, 1]
        :param f: np.array, pdf of each arm
        :param F: np.array, cdf of each arm
        :param G: np.array, G function of each arm as defined in Russo & Van Roy (algorithm 2 page 241)
        :param B: np.array, normalisation coefficient
        :return: np.arrays, updates of arrays f, F, G and B
        """
        adjust = beta[0] * y + beta[1] * (1 - y)
        sign_F_update = 1. if y == 0 else -1.
        f[arm] = (X * y + (1 - X) * (1 - y)) * beta.sum() / adjust * f[arm]
        G[arm] = beta[0] / beta.sum() * (F[arm] - X ** beta[0] * (1. - X) ** beta[1] / beta[0] / B[arm])
        F[arm] = F[arm] + sign_F_update * X ** beta[0] * (1. - X) ** beta[1] / adjust / B[arm]
        B[arm] = B[arm] * adjust / beta.sum()
        return f, F, G, B

    def IDS_approx(self, T, N, display_results=False):
        """
        Implementation of the Information Directed Sampling with approximation of integrals using a grid on [0, 1]
        for Bernoulli Bandit Problems with beta prior
        :param T: int, time horizon
        :param N: int, number of points to take in the [0,1] interval
        :param display_results: boolean, if True displayed. Defaut False
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        beta1, beta2 = self.init_prior()
        X, f, F, G, B, maap, p_star, prod_F1, g = self.init_approx(N, beta1, beta2)
        for t in range(T):
            if not self.flag:
                if np.max(p_star) > self.threshold:
                    # stop learning policy
                    self.flag = True
                    self.optimal_arm = np.argmax(p_star)
                    arm = self.optimal_arm
                else:
                    delta, g, p_star, maap = self.IR_approx(N, beta1, beta2, X, f, F, G, g)
                    arm = self.IDSAction(delta, g)
            else:
                arm = self.optimal_arm
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            prev_beta = np.array([copy(beta1[arm]), copy(beta2[arm])])
            # Posterior update
            beta1[arm], beta2[arm] = beta1[arm] + reward[t], beta2[arm] + 1-reward[t]
            if display_results:
                utils.display_results(delta, g, delta**2/g, p_star)
            f, F, G, B = self.update_approx(arm, reward[t], prev_beta, X, f, F, G, B)
        return reward, arm_sequence

    def KG(self, T):
        """
        Implementation of Knowledge Gradient algorithm for Bernoulli Bandit Problems with beta prior
        as described in Ryzhov et al. (2010) 'The knowledge gradient algorithm for a general class of online
        learning problems
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        v = np.zeros(self.nb_arms)
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                mu = Sa / Na
                c = np.array([max([mu[i] for i in range(self.nb_arms) if i != arm]) for arm in range(self.nb_arms)])
                for arm in range(self.nb_arms):
                    if mu[arm] <= c[arm] < (Sa[arm]+1)/(Na[arm]+1):
                        v[arm] = mu[arm] * ((Sa[arm]+1)/(Na[arm]+1) - c[arm])
                    elif Sa[arm]/(Na[arm]+1) < c[arm] < mu[arm]:
                        v[arm] = (1-mu[arm])*(c[arm]-Sa[arm]/(Na[arm]+1))
                    else:
                        v[arm] = 0
                arm = rd_argmax(mu + (T-t)*v)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence

    def Approx_KG_star(self, T):
        """
        Implementation of Optimized Knowledge Gradient algorithm for Bernoulli Bandit Problems with beta prior
        as described in Kaminski (2015) 'Refined knowledge-gradient policy for learning probabilities'
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        m = np.zeros(self.nb_arms)
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                mu = Sa / Na
                c = np.array([max([mu[i] for i in range(self.nb_arms) if i != arm]) for arm in range(self.nb_arms)])
                for arm in range(self.nb_arms):
                    if c[arm] >= mu[arm]:
                        ta = Na[arm] * (c[arm]-mu[arm]) / (1-c[arm]+10e-9)
                        m[arm] = np.nan_to_num(mu[arm]**ta/ta)
                    else:
                        ta = Na[arm] * (mu[arm]-c[arm]) / (c[arm]+10e-9)
                        m[arm] = ((1-mu[arm])**ta)/ta
                arm = rd_argmax(mu + (T-t)*m)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence

    def computeIDS(self, Maap, p_a, thetas, M, VIDS=False):
        """
        Implementation of SampleIR (algorithm 4 in Russo & Van Roy, p. 242) applied for Bernoulli Bandits with
        beta prior. Here integrals are no more approximated using a grid on [0, 1] but in sampling thetas according
        to their respective posterior distributions.
        :param Maap: np.array, M(a|a') as defined in Russo & Van Roy's paper
        :param p_a: np.array, probability p* of choosing each arm supposing the latter is the optimal one
        :param thetas: np.array, posterior samples
        :param M: int, number of samples
        :param VIDS: boolean, if True choose arm which delta**2/v quantity
        :return: int, np.array, arm chose and p*
        """
        mu, theta_hat = np.mean(thetas, axis=1), np.argmax(thetas, axis=0)
        for a in range(self.nb_arms):
            mu[a] = np.mean(thetas[a])
            for ap in range(self.nb_arms):
                t = thetas[ap, np.where(theta_hat == a)]
                Maap[ap, a] = np.nan_to_num(np.mean(t))
                if ap == a:
                    p_a[a] = t.shape[1]/M
        if np.max(p_a) >= self.threshold:
            # Stop learning policy
            self.optimal_arm = np.argmax(p_a)
            arm = self.optimal_arm
        else:
            rho_star = sum([p_a[a] * Maap[a, a] for a in range(self.nb_arms)])
            delta = rho_star - mu
            if VIDS:
                v = np.array([sum([p_a[ap] * (Maap[a, ap] - mu[a]) ** 2 for ap in range(self.nb_arms)])
                              for a in range(self.nb_arms)])
                arm = rd_argmax(-delta ** 2 / v)
            else:
                g = np.array([sum([p_a[ap] * (Maap[a, ap] * np.log(Maap[a, ap]/mu[a]+1e-10) +
                                              (1-Maap[a, ap]) * np.log((1-Maap[a, ap])/(1-mu[a])+1e-10))
                                   for ap in range(self.nb_arms)]) for a in range(self.nb_arms)])
                arm = self.IDSAction(delta, g)
        return arm, p_a

    def IDS_sample(self, T, M=10000, VIDS=False):
        """
        Implementation of the Information Directed Sampling with approximation of integrals using MC sampling
        for Bernoulli Bandit Problems with beta prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :param VIDS: boolean, if True choose arm which delta**2/v quantity. Default: False
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        beta1, beta2 = self.init_prior()
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        Maap, p_a = np.zeros((self.nb_arms, self.nb_arms)), np.zeros(self.nb_arms)
        thetas = np.array([np.random.beta(beta1[arm], beta2[arm], M) for arm in range(self.nb_arms)])
        for t in range(T):
            if not self.flag:
                if np.max(p_a) >= self.threshold:
                    # Stop learning policy
                    self.flag = True
                    arm = self.optimal_arm
                else:
                    arm, p_a = self.computeIDS(Maap, p_a, thetas, M, VIDS)
            else:
                arm = self.optimal_arm
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            beta1[arm] += reward[t]
            beta2[arm] += 1-reward[t]
            thetas[arm] = np.random.beta(beta1[arm], beta2[arm], M)
        return reward, arm_sequence


    def VIDS_sample(self, T, M=10000, VIDS=True):
        """
        Implementation of the V-IDS with approximation of integrals using MC sampling
        for Bernoulli Bandit Problems with beta prior
        :param T: int, time horizon
        :param M: int, number of samples
        :param VIDS: boolean, if True choose arm which delta**2/v quantity. Default: True
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        beta1, beta2 = self.init_prior()
        Maap, p_a = np.zeros((self.nb_arms, self.nb_arms)), np.zeros(self.nb_arms)
        thetas = np.array([np.random.beta(beta1[arm], beta2[arm], M) for arm in range(self.nb_arms)])
        for t in range(T):
            if not self.flag:
                if np.max(p_a) >= self.threshold:
                    # Stop learning policy
                    self.flag = True
                    arm = self.optimal_arm
                else:
                    arm, p_a = self.computeIDS(Maap, p_a, thetas, M, VIDS)
            else:
                arm = self.optimal_arm
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            # Posterior update
            beta1[arm] += reward[t]
            beta2[arm] += 1-reward[t]
            thetas[arm] = np.random.beta(beta1[arm], beta2[arm], M)
        return reward, arm_sequence
