from MAB import *


class BetaBernoulliMAB(GenericMAB):
    def __init__(self, p):
        super().__init__(method=['B']*len(p), param=p)
        self.Cp = sum([(self.mu_max-x)/self.kl(x, self.mu_max) for x in self.means if x != self.mu_max])
        self.flag = False
        self.optimal_arm = None
        self.threshold = 0.99

    @staticmethod
    def kl(x, y):
        """
        Implementation of the Kullback-Leibler divergence for two Bernoulli distributions (B(x),B(y))
        """
        return x * np.log(x/y) + (1-x) * np.log((1-x)/(1-y))

    def TS(self, T):
        """
        Implementation of the Thomson Sampling algorithm
        :param T: number of rounds
        :return: Reward obtained by the policy and sequence of the chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        theta = np.zeros(self.nb_arms)
        for t in range(T):
            for k in range(self.nb_arms):
                if Na[k] >= 1:
                    theta[k] = np.random.beta(Sa[k]+1, Na[k]-Sa[k]+1)
                else:
                    theta[k] = np.random.uniform()
            arm = rd_argmax(theta)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence

    def BayesUCB(self, T, p1, p2, c=0):
        """
        BayesUCB implementation in the case of a Beta(p1, p2) prior on the theta parameters
        for a BinomialMAB.
        Implementation of On Bayesian Upper Confidence Bounds for Bandit Problems, Kaufman & al,
        from http://proceedings.mlr.press/v22/kaufmann12/kaufmann12.pdf
        :param T: number of rounds
        :param p1: First parameter of the Beta prior probability distribution
        :param p2: Second parameter of the Beta prior probability distribution
        :param c: Parameter for the quantiles. Default value c=0
        :return: Reward obtained by the policy and sequence of the arms choosed
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

    def IR(self, b1, b2):
        """
        Implementation of the Information Ratio for bernoulli bandits with beta prior
        :param b1: np.array, first parameter of the beta distribution for each arm
        :param b2: np.array, second parameter of the beta distribution for each arm
        :return: the two components of the Information ration delta and g
        """
        assert type(b1) == np.ndarray, "b1 type should be an np.array"
        assert type(b2) == np.ndarray, "b2 type should be an np.array"

        def joint_cdf(x):
            result = 1.
            for i in range(self.nb_arms):
                result *= beta.cdf(x, b1[i], b2[i])
            return result

        def G(x, a):
            return b1[a]/(b1[a]+b2[a])*beta.cdf(x, b1[a]+1, b2[a])

        def dp_star(x, a):
            return beta.pdf(x, b1[a], b2[a])*joint_cdf(x)/beta.cdf(x, b1[a], b2[a])

        def p_star(a):
            return integrate.quad(lambda x: dp_star(x, a), 0., 1., epsabs=1e-2)[0]  # return a tuple (value, UB error)

        def MAA(a, p):
            return integrate.quad(lambda x: x*dp_star(x, a), 0., 1., epsabs=1e-2)[0]/p[a]

        def MAAP(ap, a, p):
            return integrate.quad(lambda x: dp_star(x, a)*G(x, ap)/beta.cdf(x, b1[ap], b2[ap]), 0., 1., epsabs=1e-2)[0]\
                   / p[a]

        def g(a, p, M, ma_value):
            gp = p*(M[a]*np.log(M[a]*(b1+b2)/b1)+(1-M[a])*np.log((1-M[a])*(b1+b2)/b2))
            gp[a] = ma_value
            return gp.sum()

        ps = np.array([p_star(a) for a in range(self.nb_arms)])
        ma = np.array([MAA(a, ps) for a in range(self.nb_arms)])
        maap = np.array([[MAAP(a, ap, ps) for ap in range(self.nb_arms)] for a in range(self.nb_arms)])
        np.fill_diagonal(maap, 0, wrap=False)
        rho = (ps*ma).sum()
        delta = rho-b1/(b1+b2)
        g = np.array([g(a, ps, maap, ma[a]) for a in range(self.nb_arms)])
        return delta, g

    def IDS(self, T):
        """
        Implementation of the Information Directed Sampling for Beta-Bernoulli bandits
        :param T: number of rounds
        :return: Reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        beta_1 = np.ones(self.nb_arms)
        beta_2 = np.ones(self.nb_arms)
        for t in range(T):
            delta, g = self.IR(beta_1, beta_2)
            arm = self.IDSAction(delta, g)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            beta_1[arm] += reward[t]
            beta_2[arm] += 1-reward[t]
        return reward, arm_sequence

    def IR_approx(self, N, b1, b2, X, f, F, G):
        """
        Implementation of the Information Ratio for bernoulli bandits with beta prior
        :param b1: np.array, first parameter of the beta distribution for each arm
        :param b2: np.array, second parameter of the beta distribution for each arm
        :return: the two components of the Information ration delta and g
        """
        assert type(b1) == np.ndarray, "b1 type should be an np.array"
        assert type(b2) == np.ndarray, "b2 type should be an np.array"
        maap = np.zeros((self.nb_arms, self.nb_arms))
        p_star = np.zeros(self.nb_arms)
        prod_F1 = np.ones((self.nb_arms, self.nb_arms, N))
        for a in range(self.nb_arms):
            for ap in range(self.nb_arms):
                for app in range(self.nb_arms):
                    if a != app and app != ap:
                        prod_F1[a, ap] = prod_F1[a, ap]*F[app]
                prod_F1[a, ap] *= f[a]/N
                p_star[a] = (prod_F1[a, a]).sum()

        for a in range(self.nb_arms):
            for ap in range(self.nb_arms):
                if a != ap:
                    maap[ap, a] = (prod_F1[a, ap]*G[ap]).sum()/p_star[a]
                else:
                    maap[a, a] = (prod_F1[a, a]*X).sum()/p_star[a]
            p_star[a] = (prod_F1[a, a]).sum()
        for a in range(self.nb_arms):
                for ap in range(self.nb_arms):
                    if a != ap:
                        maap[ap, a] = (prod_F1[a, ap]*G[ap]).sum()/p_star[a]
                    else:
                        maap[a, a] = (prod_F1[a, a]*X).sum()/p_star[a]
        rho_star = np.inner(np.diag(maap), p_star)
        delta = rho_star - b1/(b1+b2)
        g = np.zeros(self.nb_arms)
        for arm in range(self.nb_arms):
            sum_log = maap[arm]*np.log(maap[arm]*(b1+b2)/b1) + (1-maap[arm])*np.log((1-maap[arm])*(b1+b2)/b2)
            g[arm] = np.inner(p_star, sum_log)
        return delta, g, p_star, maap

    def init_approx(self, N):
        """
        :param N: number of points to take in the [0,1] interval
        :return: Initialisation of the arrays for the approximation of the integrals in IDS
        The initialization is made for uniform prior (equivalent to beta(1,1))
        """
        X = np.linspace(1/N, 1., N)
        f = np.ones((self.nb_arms, N))
        F = np.repeat(X, self.nb_arms, axis=0).reshape((N, self.nb_arms)).T
        G = F**2/2
        B = np.ones(self.nb_arms)
        return X, f, F, G, B

    def update_approx(self, arm, y, beta, X, f, F, G, B):
        """
        Update all functions with recursion formula. These formula are all derived
        using the properties of the beta distribution: the pdf and cdf of beta(a, b)
         can be used to compute the cdf and pdf of beta(a+1, b) and beta(a, b+1)
        """
        adjust = beta[0]*y+beta[1]*(1-y)
        sign_F_update = 1. if y == 0 else -1.
        f[arm] = (X*y+(1-X)*(1-y))*beta.sum()/adjust*f[arm]
        G[arm] = beta[0]/beta.sum()*(F[arm]-X**beta[0]*(1.-X)**beta[1]/beta[0]/B[arm])
        F[arm] = F[arm] + sign_F_update*X**beta[0]*(1.-X)**beta[1]/adjust/B[arm]
        B[arm] = B[arm]*adjust/beta.sum()
        return f, F, G, B

    def IDS_approx(self, T, N_steps, display_results = False):
        """
        Implementation of the Information Directed Sampling with approximation of integrals
        :param T: number of rounds
        :return: Reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        X, f, F, G, B = self.init_approx(N_steps)
        beta_1 = np.ones(self.nb_arms)
        beta_2 = np.ones(self.nb_arms)
        p_star = np.zeros(self.nb_arms)
        for t in range(T):
            if not self.flag:
                if np.max(p_star) > self.threshold:
                    self.flag = True
                    self.optimal_arm = np.argmax(p_star)
                    arm = self.optimal_arm
                else:
                    delta, g, p_star, maap = self.IR_approx(N_steps, beta_1, beta_2, X, f, F, G)
                    # arm = rd_argmax(-delta**2/g)
                    arm = self.IDSAction(delta, g)
                    # print('chosen arm: {}'.format(arm))
                    # print('IDS action : {}'.format(self.IDSAction0(delta, g)))
            else:
                arm = self.optimal_arm
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            prev_beta = np.array([copy.copy(beta_1[arm]), copy.copy(beta_2[arm])])
            beta_1[arm] += reward[t]
            beta_2[arm] += 1-reward[t]
            # print(t)
            # print(Sa/Na)
            # print(Na)
            # print(delta)
            # print(g)
            # print('ratio : {}'.format(delta**2/g))
            # print(p_star)
            # print(maap)
            f, F, G, B = self.update_approx(arm, reward[t], prev_beta, X, f, F, G, B)
        if display_results:
            res = {'delta': delta, 'g': g, 'p_star': p_star, 'maap': maap }
            print(res)
        return reward, arm_sequence


    def KG(self, T):
        """
        Implementation of Knowledge Gradient algorithm
        :param T: number of rounds
        :return: Reward obtained by the policy and sequence of the chosen arms
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
