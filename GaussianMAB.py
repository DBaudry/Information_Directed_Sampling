from MAB import *


class GaussianMAB(GenericMAB):
    """

    """
    def __init__(self, p):
        super().__init__(method=['G']*len(p), param=p)
        self.flag = False
        self.optimal_arm = None
        self.threshold = 0.99

    def init_prior(self, s0=1):
        ''' Init prior for Gaussian prior '''
        mu = np.zeros(self.nb_arms)
        sigma = s0 * np.ones(self.nb_arms)
        return mu, sigma

    def TS(self, T):
        """
        Implementation of the Thomson Sampling algorithm for Gaussian bandits
        :param T: number of rounds
        :return: Reward obtained by the policy and sequence of the arms chose

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
            eta = self.MAB[arm].eta
            mu[arm] = (eta ** 2 * mu[arm] + reward[t] * sigma[arm] ** 2) / (eta ** 2 + sigma[arm] ** 2)
            sigma[arm] = np.sqrt((eta * sigma[arm]) ** 2 / (eta ** 2 + sigma[arm] ** 2))
        return reward, arm_sequence

    def BayesUCB(self, T, p1, p2, c=0):
        """
        BayesUCB implementation in the case of a N(p1,p2) prior on the theta parameters
        for a GaussianMAB.
        Implementation of On Bayesian Upper Confidence Bounds for Bandit Problems, Kaufman & al,
        from http://proceedings.mlr.press/v22/kaufmann12/kaufmann12.pdf
        :param T: number of rounds
        :param p1: First parameter of the normal prior probability distribution
        :param p2: Second parameter of the normal prior probability distribution
        :param c: Parameter for the quantiles. Default value c=0
        :return: Reward obtained by the policy and sequence of the arms chose
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        mu, sigma = self.init_prior()
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                arm = rd_argmax(mu + sigma * norm.ppf(t/(t+1)))
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            eta = self.MAB[arm].eta
            mu[arm] = (eta ** 2 * mu[arm] + reward[t] * sigma[arm] ** 2) / (eta ** 2 + sigma[arm] ** 2)
            sigma[arm] = np.sqrt((eta * sigma[arm]) ** 2 / (eta ** 2 + sigma[arm] ** 2))
        return reward, arm_sequence


    def GPUCB(self, T):
        """
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        mu, sigma = self.init_prior()
        for t in range(T):
            beta = 2 * np.log(self.nb_arms * ((t+1) * np.pi) ** 2 / 6 / 0.1)
            arm = rd_argmax(mu + sigma*np.sqrt(beta))
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            eta = self.MAB[arm].eta
            mu[arm] = (eta ** 2 * mu[arm] + reward[t] * sigma[arm] ** 2) / (eta ** 2 + sigma[arm] ** 2)
            sigma[arm] = np.sqrt((eta * sigma[arm]) ** 2 / (eta ** 2 + sigma[arm] ** 2))
        return reward, arm_sequence

    def Tuned_GPUCB(self, T, c=0.9):
        """
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        mu, sigma = self.init_prior()
        for t in range(T):
            arm = rd_argmax(mu + sigma*np.sqrt(c*np.log(t+1)))
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            eta = self.MAB[arm].eta
            mu[arm] = (eta ** 2 * mu[arm] + reward[t] * sigma[arm] ** 2) / (eta ** 2 + sigma[arm] ** 2)
            sigma[arm] = np.sqrt((eta * sigma[arm]) ** 2 / (eta ** 2 + sigma[arm] ** 2))
        return reward, arm_sequence

    def kgf(self, x):
        return norm.cdf(x) * x + norm.pdf(x)

    def KG(self, T):
        """
        Implementation of Knowledge Gradient algorithm
        :param T: number of rounds
        :return: Reward obtained by the policy and sequence of the arms chosen
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
        return np.array(reward), np.array(arm_sequence)

    def KG_star(self, T):
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

    def IR_approx(self, mu, sigma, X, f, F, N):
        """
        Implementation of the Information Ratio for bernoulli bandits with beta prior
        :param b1: np.array, first parameter of the beta distribution for each arm
        :param b2: np.array, second parameter of the beta distribution for each arm
        :return: the two components of the Information ration delta and g
        """
        assert type(mu) == np.ndarray, "b1 type should be an np.array"
        assert type(sigma) == np.ndarray, "b2 type should be an np.array"
        maap = np.zeros((self.nb_arms, self.nb_arms))
        p_star = np.zeros(self.nb_arms)
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

    def init_approx(self, N):
        """
        :param N: number of points to take in the [0,1] interval
        :return: Initialisation of the arrays for the approximation of the integrals in IDS
        The initialization is made for uniform prior (equivalent to beta(1,1))
        """
        X = np.linspace(-10, 10., N)
        f = np.repeat(norm.pdf(X, 0, 1), self.nb_arms, axis=0).reshape((N, self.nb_arms)).T
        F = np.repeat(norm.cdf(X, 0, 1), self.nb_arms, axis=0).reshape((N, self.nb_arms)).T
        return X, f, F

    def update_approx(self, arm, m, s, X, f, F, N):
        """
        Update all functions with recursion formula. These formula are all derived
        using the properties of the beta distribution: the pdf and cdf of beta(a, b)
         can be used to compute the cdf and pdf of beta(a+1, b) and beta(a, b+1)
        """
        f[arm] = norm.pdf(X, m, s)
        F[arm] = norm.cdf(X, m, s)

        return f, F

    def IDS_approx(self, T, N_steps=10000):
        """
        Implementation of the Information Directed Sampling with approximation of integrals
        :param T: number of rounds
        :return: Reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        X, f, F = self.init_approx(N_steps)
        mu, sigma = self.init_prior(s0=1)
        p_star = np.zeros(self.nb_arms)
        for t in tqdm(range(T)):
            if not self.flag:
                if np.max(p_star) > self.threshold:
                    self.flag = True
                    self.optimal_arm = np.argmax(p_star)
                    arm = self.optimal_arm
                else:
                    delta, v, p_star, maap = self.IR_approx(mu, sigma, X, f, F, N_steps)
                    arm = rd_argmax(-delta**2/v)
            else:
                arm = self.optimal_arm
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            eta = self.MAB[arm].eta
            mu[arm] = (eta ** 2 * mu[arm] + reward[t] * sigma[arm] ** 2) / (eta ** 2 + sigma[arm] ** 2)
            sigma[arm] = np.sqrt((eta * sigma[arm]) ** 2 / (eta ** 2 + sigma[arm] ** 2))
            f, F = self.update_approx(arm, mu[arm], sigma[arm], X, f, F, N_steps)
        return reward, arm_sequence

    def computeIDS(self, Maap, p_a,thetas, M):
        mu = np.mean(thetas, axis=1)
        theta_hat = np.argmax(thetas, axis=0)
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
            arm = rd_argmax(-delta ** 2 / v)
        return arm, p_a

    def IDS_sample(self, T, M=10000):
        eta = 1
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        mu, sigma = self.init_prior(s0=1)
        reward, arm_sequence = np.zeros(T), np.zeros(T)
        Maap, p_a = np.zeros((self.nb_arms, self.nb_arms)), np.zeros(self.nb_arms)
        thetas = np.array([np.random.normal(mu[arm], sigma[arm], M) for arm in range(self.nb_arms)])
        for t in range(T):
            if not self.flag:
                if np.max(p_a) >= self.threshold:
                    self.flag = True
                    arm = self.optimal_arm
                else:
                    arm, p_a = self.computeIDS(Maap, p_a, thetas, M)
            else:
                arm = self.optimal_arm
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            mu[arm] = (eta ** 2 * mu[arm] + reward[t] * sigma[arm] ** 2) / (eta ** 2 + sigma[arm] ** 2)
            sigma[arm] = np.sqrt((eta * sigma[arm]) ** 2 / (eta ** 2 + sigma[arm] ** 2))
            thetas[arm] = np.random.normal(mu[arm], sigma[arm], M)
        return reward, arm_sequence