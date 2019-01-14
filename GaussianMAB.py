from MAB import *


class GaussianMAB(GenericMAB):
    """

    """
    def __init__(self, p, s=10):
        super().__init__(method=['G']*len(p), param=p)
        self.flag = False
        self.optimal_arm = None
        self.threshold = 0.99
        self.s = s

    def init_prior(self):
        ''' Init prior for Gaussian prior '''
        mu, sigma = np.zeros(self.nb_arms), self.s * np.ones(self.nb_arms)
        return mu, sigma

    def TS(self, T):
        """
        Implementation of the Thomson Sampling algorithm for Gaussian bandits
        :param T: number of rounds
        :return: Reward obtained by the policy and sequence of the arms chose
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        mu, S = np.zeros(self.nb_arms), np.zeros(self.nb_arms)
        alpha = 0.5
        n_bar = max(2, 3-np.ceil(2*alpha))
        for t in range(T):
            if t < self.nb_arms * n_bar:
                arm = t % self.nb_arms
            else:
                for arm in range(self.nb_arms):
                    S[arm] = sum([r**2 for r in reward[np.where(arm_sequence==arm)]]) - Sa[arm]**2/Na[arm]
                    mu[arm] = Sa[arm]/Na[arm] + np.sqrt(S[arm]/(Na[arm]*(Na[arm]+2*alpha-1))) * np.random.standard_t(Na[arm]+2*alpha-1,1)
                arm = rd_argmax(mu)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
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
        quantiles = np.zeros(self.nb_arms)
        S, mu = np.zeros(self.nb_arms), np.zeros(self.nb_arms)
        for n in range(T):
            for arm in range(self.nb_arms):
                if Na[arm] >= 2:
                    S[arm] = (sum([r ** 2 for r in reward[np.where(arm_sequence == arm)]]) - Sa[arm] ** 2 / Na[arm]) / (Na[arm]-1)
                    quantiles[arm] = Sa[arm]/Na[arm] + np.sqrt(S[arm]/Na[arm]) * t.ppf(1-1/(n+1), Na[arm]-1)
                    arm = rd_argmax(quantiles)
                else:
                    arm = n % self.nb_arms
            self.update_lists(n, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence

    def GPUCB(self, T):
        """
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        S, mu = np.zeros(self.nb_arms), np.zeros(self.nb_arms)
        alpha = 0.5
        for t in range(T):
            if t < self.nb_arms*2:
                arm = t % self.nb_arms
            else:
                for arm in range(self.nb_arms):
                    S[arm] = sum([r ** 2 for r in reward[np.where(arm_sequence == arm)]]) - Sa[arm] ** 2 / Na[arm]
                    mu[arm] = Sa[arm] / Na[arm] + np.sqrt(
                        S[arm] / (Na[arm] * (Na[arm] + 2 * alpha - 1))) * np.random.standard_t(Na[arm] + 2 * alpha - 1, 1)
                beta = 2 * np.log(self.nb_arms * (t*np.pi)**2 / 6 / 0.1)
                arm = rd_argmax(mu + np.sqrt(beta * S/(Na * (Na+2*alpha-1))))
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence

    def Tuned_GPUCB(self, T, c=0.9):
        """
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        S, mu = np.zeros(self.nb_arms), np.zeros(self.nb_arms)
        alpha = 0.5
        for t in range(T):
            if t < self.nb_arms*2:
                arm = t % self.nb_arms
            else:
                for arm in range(self.nb_arms):
                    S[arm] = sum([r ** 2 for r in reward[np.where(arm_sequence == arm)]]) - Sa[arm] ** 2 / Na[arm]
                    mu[arm] = Sa[arm] / Na[arm] + np.sqrt(
                        S[arm] / (Na[arm] * (Na[arm] + 2 * alpha - 1))) * np.random.standard_t(Na[arm] + 2 * alpha - 1, 1)
                arm = rd_argmax(mu + np.sqrt(c*np.log(t) / (Na * (Na + 2 * alpha - 1))))
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
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

    def norm_pdf(self, X, m, s):
        return 1/s/np.sqrt(2*np.pi) * np.exp(-0.5*((X-m)/s)**2)

    def norm_cdf(self, X, m, s, N):
        pdf = self.norm_pdf(X, m, s)
        return pdf.cumsum()/N*20

    def init_approx(self, N):
        """
        :param N: number of points to take in the [0,1] interval
        :return: Initialisation of the arrays for the approximation of the integrals in IDS
        The initialization is made for uniform prior (equivalent to beta(1,1))
        """
        X = np.linspace(-10, 10., N)
        f = np.repeat(self.norm_pdf(X, 0, 1), self.nb_arms, axis=0).reshape((N, self.nb_arms)).T
        F = np.repeat(self.norm_cdf(X, 0, 1, N), self.nb_arms, axis=0).reshape((N, self.nb_arms)).T
        return X, f, F

    def update_approx(self, arm, m, s, X, f, F, N):
        """
        Update all functions with recursion formula. These formula are all derived
        using the properties of the beta distribution: the pdf and cdf of beta(a, b)
         can be used to compute the cdf and pdf of beta(a+1, b) and beta(a, b+1)
        """
        f[arm] = self.norm_pdf(X, m, s)
        F[arm] = self.norm_cdf(X, m, s, N)
        return f, F

    def IDS_approx(self, T, N_steps=10000):
        """
        Implementation of the Information Directed Sampling with approximation of integrals
        :param T: number of rounds
        :return: Reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        X, f, F = self.init_approx(N_steps)
        mu, sigma = self.init_prior()
        p_star = np.zeros(self.nb_arms)
        for t in range(T):
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