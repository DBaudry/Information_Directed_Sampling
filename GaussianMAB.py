from MAB import *


class GaussianMAB(GenericMAB):
    """

    """
    def __init__(self, p):
        super().__init__(method=['G']*len(p), param=p)

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

    def GPUCB(self, T, c):
        """
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        mu, sigma = np.zeros(self.nb_arms), np.ones(self.nb_arms)
        for t in range(T):
            arm = rd_argmax(mu + c*np.log(t+1)*sigma)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            eta = self.MAB[arm].eta
            mu[arm] = (eta ** 2 * mu[arm] + reward[t] * sigma[arm] ** 2) / (eta ** 2 + sigma[arm] ** 2)
            sigma[arm] = (eta * sigma[arm]) ** 2 / (eta ** 2 + sigma[arm] ** 2)
        return reward, arm_sequence

    @staticmethod
    def kl(x, y):
        """
        Implementation of the Kullback-Leibler divergence
        """
        return x * np.log(x/y) + (1-x) * np.log((1-x)/(1-y))

    def kl_inf(self, x, y):
        """
        Implementation of the Kullback-Leibler divergence introduced by Burnetas and Katehakis (1996) for
        non-binary rewards in [0,1]
        """
        return np.min([self.kl(x, z) for z in self.means if z > y])


    def IR(self, mu, sigma):
        """
        Implementation of the Information Ratio for gaussian bandits with gaussian prior
        :param b1: list, first parameter of the beta distribution for each arm
        :param b2: list, second parameter of the beta distribution for each arm
        :return: the two components of the Information ration delta and g
        """
        def joint_cdf(x):
            result = 1.
            for a in range(self.nb_arms):
                result = result*norm.cdf(x, mu[a], sigma[a])
            return result

        def dp_star(x, a):
            return norm.pdf(x, mu[a], sigma[a])*joint_cdf(x)/norm.cdf(x, mu[a], sigma[a])

        def p_star(a):
            x_sup = np.max([np.max(mu) + 5 * np.max(sigma), mu[a] + 5 * sigma[a]])
            x_inf = np.min([np.min(mu) - 5 * np.max(sigma), mu[a] - 5 * sigma[a]])
            return integrate.quad(lambda x: dp_star(x, a), x_inf, x_sup, epsabs=1e-2)[0]

        def MAA(a, p):
            x_sup = np.max([np.max(mu) + 5 * np.max(sigma), mu[a] + 5 * sigma[a]])
            x_inf = np.min([np.min(mu) - 5 * np.max(sigma), mu[a] - 5 * sigma[a]])
            return integrate.quad(lambda x: x*dp_star(x, a), x_inf, x_sup, epsabs=1e-2)[0]/p[a]

        def MAAP(ap, a, p):
            x_sup = np.max([np.max(mu) + 5 * np.max(sigma), np.max([mu[a], mu[ap]]) + 5 * np.max([sigma[a], sigma[ap]])])
            x_inf = np.min([np.min(mu) - 5 * np.max(sigma), np.min([mu[a], mu[ap]]) - 5 * np.max([sigma[a], sigma[ap]])])
            return mu[ap]-(sigma[ap]**2)/p[a]*integrate.quad(
                lambda x: dp_star(x, a)*norm.pdf(x, mu[ap], sigma[ap])/norm.cdf(x, mu[ap], sigma[ap]),
                x_inf, x_sup, epsabs=1e-2)[0]

        def v(a, p, M):
            vp = p*((M[a]-mu[a])**2)
            return vp.sum()

        ps = np.array([p_star(a) for a in range(self.nb_arms)])
        ma = np.array([MAA(a, ps) for a in range(self.nb_arms)])
        maap = np.array([[MAAP(a, ap, ps) for a in range(self.nb_arms)] for ap in range(self.nb_arms)])
        rho = (ps*ma).sum()
        delta = rho-mu
        v = np.array([v(a, ps, maap.T) for a in range(self.nb_arms)])
        return delta, v

    def IDS(self, T):
        """
        Implementation of the Information Directed Sampling for Gaussian bandits
        :param T: number of rounds
        :return: Reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        mu = np.zeros(self.nb_arms)
        sigma = np.ones(self.nb_arms)
        for t in range(T):
            delta, v = self.IR(mu, sigma)
            arm = self.IDSAction(delta, v)
            eta = self.MAB[arm].eta
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            mu[arm] = (eta ** 2 * mu[arm] + reward[t] * sigma[arm] ** 2) / (eta ** 2 + sigma[arm] ** 2)
            sigma[arm] = (eta * sigma[arm]) ** 2 / (eta ** 2 + sigma[arm] ** 2)
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
        mu = np.zeros(self.nb_arms)
        sigma = np.ones(self.nb_arms)
        for t in range(T):
            if t < 2*self.nb_arms+1:
                arm = t % self.nb_arms
                self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
                mu[arm] = Sa[arm] / Na[arm]
                sigma[arm] = 1/Na[arm]*(sum([r ** 2 for r in reward[np.where(arm_sequence == arm)]]) - Sa[arm] ** 2 / Na[arm])
            else:
                delta = np.array(
                    [mu[i] - np.max(list(mu)[:i] + list(mu)[i + 1:]) for i in range(self.nb_arms)])
                v = sigma * self.kgf(-np.absolute(delta / (sigma + 10e-9)))
                arm = rd_argmax(mu + (T - t) * v)
                self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
                eta = self.MAB[arm].eta
                mu[arm] = (eta ** 2 * mu[arm] + reward[t] * sigma[arm] ** 2) / (eta ** 2 + sigma[arm] ** 2)
                sigma[arm] = (eta * sigma[arm]) ** 2 / (eta ** 2 + sigma[arm] ** 2)
        return np.array(reward), np.array(arm_sequence)

    def multi_simul_KG(self, M, arm, sigma, mu, delta):
        eta = self.MAB[arm].eta
        v = 0
        mu_, sigma_, delta_ = mu[arm], sigma[arm], delta[arm]
        for k in range(M):
            delta_ = mu_ - np.max([mu[i] for i in range(self.nb_arms) if i != arm])
            v += sigma_ * self.kgf(-np.absolute(delta_ / (sigma_ + 10e-9)))
            y = self.MAB[arm].sample()
            mu_ = (eta ** 2 * mu_ + y * sigma_ ** 2) / (eta ** 2 + sigma_ ** 2)
            sigma_ = (eta * sigma_) ** 2 / (eta ** 2 + sigma_ ** 2)
        return v/M


    def KG_star(self, T, lbda=100):
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        mu = np.zeros(self.nb_arms)
        sigma = np.ones(self.nb_arms)
        for t in range(T):
            if t < 2*self.nb_arms+1:
                arm = t % self.nb_arms
                self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
                mu[arm] = Sa[arm] / Na[arm]
                sigma[arm] = 1/Na[arm]*(sum([r ** 2 for r in reward[np.where(arm_sequence == arm)]]) - Sa[arm] ** 2 / Na[arm])
            else:
                delta = np.array(
                    [mu[i] - np.max(list(mu)[:i] + list(mu)[i + 1:]) for i in range(self.nb_arms)])
                r = (delta / (sigma + 10e-9)) ** 2
                m_lower = lbda / (4 * sigma ** 2 + 10e-9) * (-1 + r + np.sqrt(1 + 6 * r + r ** 2))
                m_higher = lbda / (4 * sigma ** 2 + 10e-9) * (1 + r + np.sqrt(1 + 10 * r + r ** 2))
                v = np.zeros(self.nb_arms)
                for arm in range(self.nb_arms):
                    if T - t <= m_lower[arm]:
                        M = T - t
                    elif (delta[arm] == 0) or (m_higher[arm] <= 1):
                        M = 1
                    else:
                        M = np.ceil(0.5 * ((m_lower + m_higher)[arm])).astype(int)  # approximation
                    v[arm] = self.multi_simul_KG(M, arm, sigma, mu, delta)
                arm = rd_argmax(mu + (T-t)*v)
                self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
                eta = self.MAB[arm].eta
                mu[arm] = (eta ** 2 * mu[arm] + reward[t] * sigma[arm] ** 2) / (eta ** 2 + sigma[arm] ** 2)
                sigma[arm] = (eta * sigma[arm]) ** 2 / (eta ** 2 + sigma[arm] ** 2)
        return np.array(reward), np.array(arm_sequence)
