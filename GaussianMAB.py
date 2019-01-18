""" Packages import """
from MAB import *
from scipy.stats import norm


class GaussianMAB(GenericMAB):
    """
    Gaussian Bandit Problem
    """
    def __init__(self, p):
        """
        Initialization
        :param p: np.array, true values of (mu, sigma) for each arm with mean sampled from N(mu, sigma)
        """
        # Initialization of arms from GenericMAB
        super().__init__(methods=['G']*len(p), p=p)
        # Parameters used for stop learning policy
        self.flag = False
        self.optimal_arm = None
        self.threshold = 0.99

    def init_prior(self, mu0=0, s0=1):
        """
        Init Gaussian prior
        :param mu0: float, multiplicative factor for mean
        :param s0: float, multiplicative factor for variance
        :return: np.arrays, prior values (mu, sigma) for each earm
        """
        mu = mu0 * np.ones(self.nb_arms)
        sigma = s0 * np.ones(self.nb_arms)
        return mu, sigma

    def update_posterior(self, arm, r, sigma, mu):
        """
        Update posterior mean and std for the chose arm only
        :param arm: int, arm chose
        :param r: float, associated reward
        :param sigma: np.array, posterior stds
        :param mu: np.array, posterior means
        :return: np.arrays, updated means and stds
        """
        eta = self.MAB[arm].eta
        mu[arm] = (eta ** 2 * mu[arm] + r * sigma[arm] ** 2) / (eta ** 2 + sigma[arm] ** 2)
        sigma[arm] = np.sqrt((eta * sigma[arm]) ** 2 / (eta ** 2 + sigma[arm] ** 2))
        return mu, sigma

    def TS(self, T):
        """
        Implementation of Thomson Sampling (TS) algorithm for Gaussian Bandit Problems with normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        mu, sigma = self.init_prior()
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                theta = np.array([np.random.normal(mu[arm], sigma[arm]) for arm in range(self.nb_arms)])
                arm = rd_argmax(theta)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            mu, sigma = self.update_posterior(arm, reward[t], sigma, mu)
        return reward, arm_sequence

    def BayesUCB(self, T):
        """
        Implementation of Bayesian Upper Confidence Bounds (BayesUCB) algorithm for Gaussian Bandit Problems
        with normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        mu, sigma = self.init_prior()
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                arm = rd_argmax(mu + sigma * norm.ppf(t/(t+1)))
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            mu, sigma = self.update_posterior(arm, reward[t], sigma, mu)
        return reward, arm_sequence


    def GPUCB(self, T):
        """
        Implementation of GPUCB, Srinivas (2010) 'Gaussian Process Optimization in the Bandit Setting: No Regret and
        Experimental Design' for Gaussian Bandit Problems with normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        mu, sigma = self.init_prior()
        for t in range(T):
            beta = 2 * np.log(self.nb_arms * ((t+1) * np.pi) ** 2 / 6 / 0.1)
            arm = rd_argmax(mu + sigma*np.sqrt(beta))
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            mu, sigma = self.update_posterior(arm, reward[t], sigma, mu)
        return reward, arm_sequence

    def Tuned_GPUCB(self, T, c=0.9):
        """
        Implementation of Tuned GPUCB described in Russo & Van Roy's paper of study for Gaussian Bandit Problems
        with normal prior
        :param T: int, time horizon
        :param c: float, tunable parameter. Default 0.9
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        mu, sigma = self.init_prior()
        for t in range(T):
            arm = rd_argmax(mu + sigma*np.sqrt(c*np.log(t+1)))
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            mu, sigma = self.update_posterior(arm, reward[t], sigma, mu)
        return reward, arm_sequence

    def kgf(self, x):
        """
        :param x: np.array
        :return: np.array, f(x) as defined in Ryzhov et al. (2010) 'The knowledge gradient algorithm for
        a general class of online learning problems'
        """
        return norm.cdf(x) * x + norm.pdf(x)

    def KG(self, T):
        """
        Implementation of Knowledge Gradient algorithm for Gaussian Bandit Problems with normal prior
        as described in Ryzhov et al. (2010) 'The knowledge gradient algorithm for a general class of online
        learning problems'
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        mu, sigma = self.init_prior()
        eta = np.array([self.MAB[arm].eta for arm in range(self.nb_arms)])
        for t in range(T):
            delta_t = np.array(
                [mu[arm] - np.max(list(mu)[:arm] + list(mu)[arm+1:]) for arm in range(self.nb_arms)])
            sigma_next = np.sqrt(((sigma*eta)**2)/(sigma**2+eta**2))
            s_t = np.sqrt(sigma**2-sigma_next**2)
            v = s_t * self.kgf(-np.absolute(delta_t / (s_t + 10e-9)))
            arm = rd_argmax(mu + (T - t) * v)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            mu[arm] = (eta[arm] ** 2 * mu[arm] + reward[t] * sigma[arm] ** 2) / (eta[arm] ** 2 + sigma[arm] ** 2)
            sigma[arm] = sigma_next[arm]
        return reward, arm_sequence

    def KG_star(self, T):
        """
        Implementation of Optimized Knowledge Gradient algorithm for Bernoulli Bandit Problems with normal prior
        as described in Kaminski (2015) 'Refined knowledge-gradient policy for learning probabilities'
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        mu, sigma = self.init_prior()
        eta = np.array([self.MAB[arm].eta for arm in range(self.nb_arms)])
        for t in range(T):
            delta_t = np.array(
                [mu[i] - np.max(list(mu)[:i] + list(mu)[i + 1:]) for i in range(self.nb_arms)])
            r = (delta_t / sigma) ** 2
            m_lower = eta / (4 * sigma ** 2) * (-1 + r + np.sqrt(1 + 6 * r + r ** 2))
            m_higher = eta / (4 * sigma ** 2) * (1 + r + np.sqrt(1 + 10 * r + r ** 2))
            m_star = np.zeros(self.nb_arms)
            for arm in range(self.nb_arms):
                if T - t <= m_lower[arm]:
                    m_star[arm] = T - t
                elif (delta_t[arm] == 0) or (m_higher[arm] <= 1):
                    m_star[arm] = 1
                else:
                    m_star[arm] = np.ceil(0.5 * ((m_lower + m_higher)[arm])).astype(int)  # approximation
            s_m = np.sqrt((m_star + 1) * sigma ** 2 / ((eta / sigma) ** 2 + m_star + 1))
            v_m = s_m * self.kgf(-np.absolute(delta_t / (s_m + 10e-9)))
            arm = rd_argmax(mu - np.max(mu) + (T-t)*v_m)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            sigma_next = np.sqrt(((sigma * eta) ** 2) / (sigma ** 2 + eta ** 2))
            mu[arm] = (eta[arm] ** 2 * mu[arm] + reward[t] * sigma[arm] ** 2) / (eta[arm] ** 2 + sigma[arm] ** 2)
            sigma[arm] = sigma_next[arm]
        return reward, arm_sequence

    def IR_approx(self, mu, sigma, X, f, F, N, p_star, maap):
        """
        Computation of delta and v for Gaussian Bandit Problems with normal prior
        :param mu: float, mean for Gaussian prior probability distribution
        :param sigma: float, std for Gaussian prior probability distribution
        :param X: np.array, grid on a specified interval
        :param f: np.array, normal pdf
        :param F: np.array, normal cdf
        :param N: int, number of points to take in the range of X
        :param p_star: np.array, probability  p* of choosing optimal arm
        :param maap: np.array, M(a|a') as defined in Russo & Van Roy's paper
        :return: np.arrays, the two components of the V-IDS criteria delta and v with probability p*
        of choosing optimal arm and maap defined as M(a|a') in Russo & Van Roy's paper
        """
        maap = np.zeros((self.nb_arms, self.nb_arms))
        prod_F1 = np.ones((self.nb_arms, self.nb_arms, N))
        for a in range(self.nb_arms):
            for ap in range(self.nb_arms):
                for app in range(self.nb_arms):
                    if a != app and app != ap:
                        prod_F1[a, ap] *= F[app]
                prod_F1[a, ap] *= f[a] / N
        for a in range(self.nb_arms):
            p_star[a] = (prod_F1[a, a]).sum() * (np.max(X)-np.min(X))
            for ap in range(self.nb_arms):
                if a != ap:
                    maap[ap, a] = mu[ap] - sigma[ap]**2 * (prod_F1[a, ap] * f[ap]).sum() / p_star[a]
                else:
                    maap[a, a] = (prod_F1[a, a] * X).sum() / p_star[a]
        rho_star = np.inner(np.diag(maap), p_star)
        delta = rho_star - mu
        v = np.zeros(self.nb_arms)
        for arm in range(self.nb_arms):
            v[arm] = np.inner(p_star, (maap[arm]-mu[arm])**2)
        return delta, v, p_star, maap

    def init_approx(self, N, mu, sigma, rg=10.):
        """
        Initialization of quantities of interest for VIDS_approx algorithm
        :param N: int, number of points to take in the range of X
        :param mu: np.array, posterior means
        :param sigma: np.array, posterior stds
        :param rg: float, size/2 of the interval for computing integrals
        :return: np.arrays, initialization of the arrays for the approximation of the integrals in IDS_approx
        """
        X = np.linspace(-rg, rg, N)
        mu0, sigma0 = mu[0], sigma[0]
        f = np.repeat(norm.pdf(X, mu0, sigma0), self.nb_arms, axis=0).reshape((N, self.nb_arms)).T
        F = np.repeat(norm.cdf(X, mu0, sigma0), self.nb_arms, axis=0).reshape((N, self.nb_arms)).T
        return X, f, F

    @staticmethod
    def update_approx(arm, m, s, X, f, F):
        """
        Update pdf and cdf only for the chosen arm
        :param arm: int, chosen arm
        :param m: float, posterior mean of the arm
        :param s: float, posterior std of the arm
        :param X: np.array, grid on a specified interval
        :param f: np.array, normal pdf
        :param F: np.array, normal cdf
        :return: np.arrays, updated f and F
        """
        f[arm] = norm.pdf(X, m, s)
        F[arm] = norm.cdf(X, m, s)
        return f, F

    def VIDS_approx(self, T, rg=10., N=10000):
        """
        Implementation of the V-IDS algorithm with approximation of integrals using a grid on [-10, 10]
        for Gaussian Bandit Problems with normal prior
        :param T: int, time horizon
        :param rg: float, size/2 of the interval for computing integrals
        :param N: int, number of points to take in the specified interval
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        mu, sigma = self.init_prior()
        X, f, F = self.init_approx(N, mu, sigma, rg)
        p_star = np.zeros(self.nb_arms)
        maap = np.zeros((self.nb_arms, self.nb_arms))
        for t in range(T):
            if not self.flag:
                if np.max(p_star) > self.threshold:
                    # Stop learning policy
                    self.flag = True
                    self.optimal_arm = np.argmax(p_star)
                    arm = self.optimal_arm
                else:
                    delta, v, p_star, maap = self.IR_approx(mu, sigma, X, f, F, N, p_star, maap)
                    arm = self.IDSAction(delta, v)
                    # arm = rd_argmax(-delta**2/v)
            else:
                arm = self.optimal_arm
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            mu, sigma = self.update_posterior(arm, reward[t], sigma, mu)
            f, F = self.update_approx(arm, mu[arm], sigma[arm], X, f, F)
        return reward, arm_sequence

    def computeVIDS(self, Maap, p_a, thetas, M):
        """
        Implementation of SampleIR (algorithm 4 in Russo & Van Roy, p. 242) applied for Gaussian Bandits with
        normal prior. Here integrals are no more approximated using a grid on [-10, 10] but in sampling thetas according
        to their respective posterior distributions.
        :param Maap: np.array, M(a|a') as defined in Russo & Van Roy's paper
        :param p_a: np.array, probability p* of choosing each arm supposing the latter is the optimal one
        :param thetas: np.array, posterior samples
        :param M: int, number of samples
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
            self.optimal_arm = np.argmax(p_a)
            arm = self.optimal_arm
        else:
            rho_star = sum([p_a[a] * Maap[a, a] for a in range(self.nb_arms)])
            delta = rho_star - mu
            v = np.array([sum([p_a[ap] * (Maap[a, ap] - mu[a]) ** 2 for ap in range(self.nb_arms)]) for a in range(self.nb_arms)])
            arm = self.IDSAction(delta, v)
            # arm = rd_argmax(-delta ** 2 / v)
        return arm, p_a

    def VIDS_sample(self, T, M=10000):
        """
        Implementation of V-IDS with approximation of integrals using MC sampling for Gaussian Bandit Problems
        with normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        mu, sigma = self.init_prior()
        reward, arm_sequence = np.zeros(T), np.zeros(T)
        Maap, p_a = np.zeros((self.nb_arms, self.nb_arms)), np.zeros(self.nb_arms)
        thetas = np.array([np.random.normal(mu[arm], sigma[arm], M) for arm in range(self.nb_arms)])
        for t in range(T):
            if not self.flag:
                if np.max(p_a) >= self.threshold:
                    # Stop learning policy
                    self.flag = True
                    arm = self.optimal_arm
                else:
                    arm, p_a = self.computeVIDS(Maap, p_a, thetas, M)
            else:
                arm = self.optimal_arm
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            mu, sigma = self.update_posterior(arm, reward[t], sigma, mu)
            thetas[arm] = np.random.normal(mu[arm], sigma[arm], M)
        return reward, arm_sequence,
