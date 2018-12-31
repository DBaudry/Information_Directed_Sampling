# Importations
import numpy as np
import arms
from tqdm import tqdm
from utils import rd_argmax
import random
from scipy.stats import beta, norm
import scipy.integrate as integrate
import copy

class GenericMAB:
    def __init__(self, method, param):
        self.MAB = self.generate_arms(method, param)
        self.nb_arms = len(self.MAB)
        self.means = [el.mean for el in self.MAB]
        self.mu_max = np.max(self.means)

    @staticmethod
    def generate_arms(meth, par):
        """
        Method for generating different arms
        :param meth: Probability distribution used for the arm
        :param par: Parameters for the probability distribution of all the different arms
        :return:
        """
        arms_list = list()
        for i, m in enumerate(meth):
            print(i, m, par[i])
            p = par[i]
            if m == 'B':
                arms_list.append(arms.ArmBernoulli(p, random_state=np.random.randint(1, 312414)))
            elif m == 'beta':
                arms_list.append(arms.ArmBeta(a=p[0], b=p[1], random_state=np.random.randint(1, 312414)))
            elif m == 'F':
                arms_list.append(arms.ArmFinite(X=p[0], P=p[1], random_state=np.random.randint(1, 312414)))
            elif m == 'Exp':
                arms_list.append(arms.ArmExp(L=p[0], B=p[1], random_state=np.random.randint(1, 312414)))
            elif m == 'G':
                arms_list.append(arms.ArmGaussian(mu=p[0], sigma=p[1], eta=p[2], random_state=np.random.randint(1, 312414)))
            else:
                raise NameError('This method is not implemented, available methods are defined in generate_arms method')
        return arms_list

    def regret(self, reward, T):
        """
        Computing the regret of a single experiment.
        :param reward: The array of reward the policy was able to receive by selecting the different actions
        :param T: Number of rounds
        :return: Cumulated regret for a single experiment
        """
        return self.mu_max * np.arange(1, T+1) - np.cumsum(reward)

    def MC_regret(self, method, N, T, param=0.2):
        """
        Monte Carlo method for approximating the expectation of the regret.
        :param method: Method used (UCB, Thomson Sampling, etc..)
        TODO: pourquoi param=0.2 et au-dessus c'est marqué method used ? il correspond a quoi ce parametre?
        """
        MC_regret = np.zeros(T)
        for _ in tqdm(range(N), desc='Computing ' + str(N) + ' simulations'):
            if method == 'RandomPolicy':
                MC_regret += self.regret(self.RandomPolicy(T)[0], T)
            elif method == 'UCB1':
                MC_regret += self.regret(self.UCB1(T, param)[0], T)
            elif method == 'TS':
                MC_regret += self.regret(self.TS(T)[0], T)
            elif method == 'IDS':
                MC_regret += self.regret(self.IDS(T)[0], T)
            elif method == 'MOSS':
                MC_regret += self.regret(self.MOSS(T, param)[0], T)
            elif method == 'KG':
                MC_regret += self.regret(self.KG(T)[0], T)
            elif method == 'KG*':
                MC_regret += self.regret(self.KG_star(T)[0], T)
            elif method == 'ExploreCommit':
                MC_regret += self.regret(self.ExploreCommit(m=param, T=T)[0], T)
            elif method == 'BayesUCB':
                MC_regret += self.regret(self.BayesUCB(T=T, a=1., b=1., c=0.)[0], T)
            elif method == 'IDS_approx':
                MC_regret += self.regret(self.IDS_approx(T=T, N_steps=1000)[0], T)
            else:
                raise NotImplementedError
        return MC_regret / N

    def init_lists(self, T):
        """
        :param T: number of rounds
        :return: - Sa: Cumulated reward of arm a
                 - Na: number of pull of arm a
                 - reward: array of reward
                 - Arms chosen: array of length T containing the arm choosed at each step
        """
        return np.zeros(self.nb_arms), np.zeros(self.nb_arms), np.zeros(T), np.zeros(T)

    def update_lists(self, t, arm, Sa, Na, reward, arm_sequence):
        """
        Update all the parameters of interest after choosing the correct arm
        :param t: current round
        :param arm: arm choosen at this round
        :param Sa:  Cumulated reward array
        :param Na:  Number of pull of arm a
        :param reward: array of reward, reward[t] is filled
        :param arm_sequence: array of the selected arms, arm_sequence[t] is filled
        :return: Nothing but update the parameters of interest
        """
        Na[arm] += 1  # Updating the number of times the arm "arm" was selected
        arm_sequence[t] = arm  # the arm "arm" was selected at round t
        new_reward = self.MAB[arm].sample() # obtaining the new reward by pulling the arm a
        reward[t] = new_reward  # adding this new reward to the array of rewards
        Sa[arm] += new_reward  # updating the cumulative reward of the arm "arm"

    def RandomPolicy(self, T):
        """
        Implementation of a random policy consisting in randomly choosing one of the available arms.
        Only useful for checking that the behavior of the different policies is normal
        :param T:  Number of rounds
        :return: Reward obtained by the policy and sequence of the arms choosed
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        for t in range(T):
            arm = random.randint(0, self.nb_arms-1)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence

    def ExploreCommit(self, m, T):
        """
        Explore-then-Commit algorithm as presented in Bandits Algorithms, Lattimore, Chapter 6
        :param m: Number of rounds before choosing the best action
        :param T: Number of steps
        :return: Reward obtained by the policy and sequence of the arms choosed
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        for t in range(m*self.nb_arms):
            arm = t % self.nb_arms
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        arm = np.argmax(Sa/Na)
        for t in range(m*self.nb_arms + 1, T):
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence

    def UCB1(self, T, rho):
        """
        Implementation of the UCB1 algorithm
        :param T: Number of rounds
        :param rho: Parameter for balancing between exploration and exploitation
        :return: Reward obtained by the policy and sequence of the arms choosed
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                arm = rd_argmax(Sa/Na+rho*np.sqrt(np.log(t+1)/2/Na))
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence

    def MOSS(self, T, rho):
        """
        Implementation of the Minimax Optimal Strategy in the Stochastic case (MOSS).
        Further details for the algorithm can be found in the chapter 9 of Bandits Algorithm (Tor Lattimore et al.)
        :param T: Number of rounds
        :param rho: Parameter for balancing between exploration and exploitation
        :return: Reward obtained by the policy and sequence of the arms choosed
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                root_term = np.array(list(map(lambda x: max(x, 1), T / (self.nb_arms * Na))))
                arm = rd_argmax(Sa / Na + rho * np.sqrt(4 / Na * np.log(root_term)))
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence

    def TS(self, T):
        """
        Implementation of the Thomson Sampling algorithm
        :param T: number of rounds
        :return: Reward obtained by the policy and sequence of the arms choosed
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
            Sa[arm] += np.random.binomial(1, reward[t])-reward[t]
        return reward, arm_sequence

    def IDSAction(self, delta, g):
        Q = np.zeros((self.nb_arms, self.nb_arms))
        IR = np.zeros((self.nb_arms, self.nb_arms))
        for a in range(self.nb_arms):
            for ap in range(self.nb_arms):
                da, dap = delta[a], delta[ap]
                ga, gap = g[a], g[ap]
                q1 = -dap/(da-dap)
                q2 = -1.
                if ga != gap:
                    q2 = q1-2*gap/(ga-gap)
                if 0 <= q1 <= 1:
                    Q[a, ap] = q1
                elif 0 <= q2 <= 1:
                    Q[a, ap] = q2
                elif da**2/ga > dap**2/gap:
                    Q[a, ap] = 1
                else:
                    Q[a, ap] = 0
                IR[a, ap] = (Q[a, ap]*(da-dap)+dap)**2/(Q[a, ap]*(ga-gap)+gap)
        amin = rd_argmax(-IR.reshape(self.nb_arms*self.nb_arms))
        a, ap = amin//self.nb_arms, amin % self.nb_arms
        b = np.random.binomial(1, Q[a, ap])
        return int(b*a+(1-b)*ap)

    def IDS(self, T):
        raise NotImplementedError

    def BayesUCB(self, T, a, b, c=0.):
        raise NotImplementedError


class BetaBernoulliMAB(GenericMAB):
    """
    TODO: checker avec DODO mais pour moi c'est bien une distribution Bernoulli sur les bras avec un prior Beta dans
    le cas bayésien
    """
    def __init__(self, p):
        super().__init__(method=['B']*len(p), param=p)
        self.Cp = sum([(self.mu_max-x)/self.kl(x, self.mu_max) for x in self.means if x != self.mu_max])

    def TS(self, T):
        """
        Implementation of the Thomson Sampling algorithm
        :param T: number of rounds
        :return: Reward obtained by the policy and sequence of the arms choosed
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

    def BayesUCB(self, T, a, b, c=0):
        """
        BayesUCB implementation in the case of a Beta(a,b) prior on the theta parameters
        for a BinomialMAB.
        Implementation of On Bayesian Upper Confidence Bounds for Bandit Problems, Kaufman & al,
        from http://proceedings.mlr.press/v22/kaufmann12/kaufmann12.pdf
        :param T: number of rounds
        :param a: First parameter of the Beta prior probability distribution
        :param b: Second parameter of the Beta prior probability distribution
        :param c: Parameter for the quantiles. Default value c=0
        :return: Reward obtained by the policy and sequence of the arms choosed
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        quantiles = np.zeros(self.nb_arms)
        for t in range(T):
            for k in range(self.nb_arms):
                if Na[k] >= 1:
                    quantiles[k] = beta.ppf(1-1/(t*np.log(T)**c), Sa[k] + a, b + Na[k] - Sa[k])
                else:
                    quantiles[k] = beta.ppf(1-1/(t*np.log(T)**c), a, b)
            arm = rd_argmax(quantiles)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence

    @staticmethod
    def kl(x, y):
        """
        Implementation of the Kullback-Leibler divergence for two Bernoulli distributions (B(x),B(y))
        """
        return x * np.log(x/y) + (1-x) * np.log((1-x)/(1-y))

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
            # TODO: pourquoi 1e-2 (Yo)
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

    def IR_approx(self, N, b1, b2, X, f, F, F_bar, G):
        """
        Implementation of the Information Ratio for bernoulli bandits with beta prior
        :param b1: np.array, first parameter of the beta distribution for each arm
        :param b2: np.array, second parameter of the beta distribution for each arm
        :return: the two components of the Information ration delta and g
        """
        assert type(b1) == np.ndarray, "b1 type should be an np.array"
        assert type(b2) == np.ndarray, "b2 type should be an np.array"
        maap = np.zeros((self.nb_arms, self.nb_arms))
        dp_star = np.apply_along_axis(lambda x: x*F_bar, 1, f/F)/(N+1)
        dp_star[:, 0] = np.zeros(self.nb_arms)
        p_star = dp_star.sum(axis=1)
        ma = (X*dp_star).sum(axis=1)/p_star
        for a in range(self.nb_arms):
            for ap in range(self.nb_arms):
                if a != ap:
                    joint_density = dp_star[a]*G[ap]/F[ap]
                    joint_density[0] = 0.
                    maap[ap, a] = joint_density.sum()/p_star[a]
                else:
                    maap[ap, a] = ma[a]
        rho_star = np.inner(ma, p_star)
        delta = rho_star - b1/(b1+b2)
        g = np.zeros(self.nb_arms)
        for arm in range(self.nb_arms):
            sum_log = maap[arm]*np.log(maap[arm]*(b1+b2)/b1) + (1-maap[arm])*np.log((1-maap[arm])*(b1+b2)/b2)
            g[arm] = np.inner(p_star, sum_log)
        return delta, g

    def init_approx(self, N):
        """
        :param N: number of points to take in the [0,1] interval
        :return: Initialisation of the arrays for the approximation of the integrals in IDS
        The initialization is made for uniform prior (equivalent to beta(1,1))
        """
        X = np.linspace(0., 1., N+1)
        f = np.ones((self.nb_arms, N+1))
        F = np.repeat(X, self.nb_arms, axis=0).reshape((N+1, self.nb_arms)).T
        G = F**2/2
        F_bar = X**self.nb_arms
        B = np.ones(self.nb_arms)
        return X, f, F, F_bar, G, B

    def update_approx(self, arm, y, beta, X, f, F, F_bar, G, B):
        adjust = beta[0]*y+beta[1]*(1-y)
        sign_F_update = 1. if y == 0 else -1.
        F_bar=F_bar/F[arm]
        f[arm] = (X*y+(1-X)*(1-y))*beta.sum()/adjust*f[arm]
        G[arm] = beta[0]/beta.sum()*(F[arm]-X**beta[0]*(1.-X)**beta[1]/beta[0]/B[arm])
        F_bar[0] = 0
        F[arm] = F[arm] + sign_F_update*X**beta[0]*(1.-X)**beta[1]/adjust/B[arm]
        F_bar = F_bar*F[arm]
        B[arm] = B[arm]*adjust/beta.sum()
        return f, F, F_bar, G, B

    def IDS_approx(self, T, N_steps):
        """
        Implementation of the Information Directed Sampling with approximation of integrals
        :param T: number of rounds
        :return: Reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        X, f, F, F_bar, G, B = self.init_approx(N_steps)
        beta_1 = np.ones(self.nb_arms)
        beta_2 = np.ones(self.nb_arms)
        for t in range(T):
            delta, g = self.IR_approx(N_steps, beta_1, beta_2, X, f, F, F_bar, G)
            arm = self.IDSAction(delta, g)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            prev_beta = np.array([copy.copy(beta_1[arm]), copy.copy(beta_2[arm])])
            beta_1[arm] += reward[t]
            beta_2[arm] += 1-reward[t]
            f, F, F_bar, G, B = self.update_approx(arm, reward[t], prev_beta, X, f, F, F_bar, G, B)
        return reward, arm_sequence

    def KG(self, T):
        """
        Implementation of Knowledge Gradient algorithm
        :param T: number of rounds
        :return: Reward obtained by the policy and sequence of the arms choosed
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                mu = Sa/Na
                Vt = max(mu)
                v = np.zeros(self.nb_arms)
                for a in range(self.nb_arms):
                    mu_up = (Sa[a]+1)/(Na[a]+1)
                    mu_down = Sa[a]/(Na[a]+1)
                    if Vt != mu[a]:
                        if mu_up <= Vt:
                            v[a] = 0
                        else:
                            v[a] = mu[a]*(mu_up-Vt)
                    else:
                        v[a] = Vt*mu_up+(1-Vt)*max([mu_down, max([mu[ap] for ap in range(self.nb_arms) if ap != a])])-Vt
                arm = rd_argmax(Sa / Na + (T - t) * v)

            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence


class FiniteSets(GenericMAB):
    def __init__(self, method, param, q_theta, prior, R):
        """
        theta in [1,L], Y in [1,N], A in [1,K]
        K is the number of arms in our algorithm and is denoted nb_arms
        :param method: list with the types for each arm
        :param param: list with the parameters for each arm
        :param q_theta: L*K*N array with the probability of each outcome knowing theta
        :param R: mapping between outcomes and rewards
        """
        super().__init__(method, param)
        self.means = [(R*p[1]).sum() for p in param]
        self.mu_max = max(self.means)
        self.q_theta = q_theta
        self.prior = prior
        self.R = R
        self.nb_arms = q_theta.shape[1]
        self.L = q_theta.shape[0]
        self.N = q_theta.shape[2]
        self.Ta = self.get_theta_a()

    def get_theta_a(self):
        """
        :return: list of length K containing the lists of theta for which action a in [1,K] is optimal
        """
        Ta = [[] for _ in range(self.nb_arms)]
        for theta in range(self.L):
            a_theta = rd_argmax(np.dot(self.q_theta[theta], self.R))
            Ta[a_theta].append(theta)
        return Ta

    def get_pa_star(self):
        """
        :return: array of shape K
         For a given prior, the probabilities that action a in [1,K] is the optimal action
        """
        pa = np.zeros(self.nb_arms)
        for a_star in range(self.nb_arms):
            for x in self.Ta[a_star]:
                pa[a_star] += self.prior[x]
        return pa

    def get_py(self):
        """
        :return: array of shape (K,N)
        Probability of outcome Y while pulling arm A for a given prior
        """
        PY = np.zeros((self.nb_arms, self.N))
        for a in range(self.nb_arms):
            PY[a] = self.q_theta[:, a, :].T @ self.prior
        return PY

    def get_joint_ay(self):
        """
        :return: Array of shape (K,K,N)
        Joint distribution of the outcome and the optimal arm while pulling arm a
        """
        P_ay = np.zeros((self.nb_arms, self.nb_arms, self.N))
        for a_star in range(self.nb_arms):
            for theta in self.Ta[a_star]:
                P_ay[:, a_star, :] += self.q_theta[theta] * self.prior[theta]
        return P_ay

    def get_R_star(self, joint_P):
        '''
        :return: Optimal expected reward for a given prior
        '''
        R = 0
        for a in range(self.nb_arms):
            for y in range(self.N):
                R += joint_P[a, a, y]*self.R[y]
        return R

    def IR(self):
        pa = self.get_pa_star()
        py = self.get_py()
        joint = self.get_joint_ay()
        R_star = self.get_R_star(joint)
        g = np.zeros(self.nb_arms)
        delta = np.zeros(self.nb_arms)+R_star
        for a in range(self.nb_arms):
            for y in range(self.N):
                for a_star in range(self.nb_arms):
                    if joint[a, a_star, y]>0:
                        g[a] += joint[a, a_star, y]*np.log(joint[a, a_star, y]/pa[a_star]/py[a, y])
                for x in range(self.L):
                    delta[a] -= self.prior[x]*self.q_theta[x, a, y]*self.R[y]
        return delta, g

    def update_prior(self, a, y):
        for x in range(self.L):
            self.prior[x] *= self.q_theta[x, a, y]
        self.prior = self.prior/self.prior.sum()

    def IDS(self, T):
        """
        Implementation of the Information Directed Sampling for Finite sets
        :param T: number of rounds
        :return: Reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, Y, arm_sequence = self.init_lists(T)
        reward = np.zeros(T)
        for t in range(T):
            delta, g = self.IR()
            arm = self.IDSAction(delta, g)
            self.update_lists(t, arm, Sa, Na, Y, arm_sequence)
            reward[t] = self.R[int(Y[t])]
            self.update_prior(arm, int(Y[t]))
        return reward, arm_sequence


class GaussianMAB(GenericMAB):
    """
    TODO: BayesUCB to adapt for gaussian bandits
    """
    def __init__(self, p):
        super().__init__(method=['G']*len(p), param=p)

    def TS(self, T):
        """
        Implementation of the Thomson Sampling algorithm
        :param T: number of rounds
        :return: Reward obtained by the policy and sequence of the arms choosed
        """
        # Sa, Na, reward, arm_sequence = self.init_lists(T)
        # mu, S = np.zeros(self.nb_arms), np.zeros(self.nb_arms)
        # n_bar = np.max(2, 3-np.ceil(2*alpha))
        # for t in range(T):
        #     if t < self.nb_arms * n_bar:
        #         arm = t % self.nb_arms
        #     else:
        #         for k in range(self.nb_arms):
        #             S[k] = sum([r**2 for r in reward[np.where(arm_sequence==k)]]) - Sa[k]**2/Na[k]
        #             mu[k] = Sa[k]/Na[k] + np.sqrt(S[k]/(Na[k]*(Na[k]+2*alpha-1))) * np.random.standard_t(Na[k]+2*alpha-1,1)
        #         arm = rd_argmax(mu)
        #     self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        mu, sigma = np.zeros(self.nb_arms), np.ones(self.nb_arms)
        for t in range(T):
            if t < 2*self.nb_arms+1:
                arm = t % self.nb_arms
                self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
                mu[arm] = Sa[arm] / Na[arm]
                sigma[arm] = 1/Na[arm]*(sum([r ** 2 for r in reward[np.where(arm_sequence == arm)]]) - Sa[arm] ** 2 / Na[arm])
            else:
                arm = rd_argmax(mu)
                eta = self.MAB[arm].eta
                self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
                mu[arm] = (eta**2 * mu[arm] + reward[t] * sigma[arm]**2) / (eta**2 + sigma[arm]**2)
                sigma[arm] = (eta*sigma[arm])**2 / (eta**2 + sigma[arm]**2)
        return reward, arm_sequence


    def BayesUCB(self, T, a, b, c=0):
        """
        BayesUCB implementation in the case of a Beta(a,b) prior on the theta parameters
        for a BinomialMAB.
        Implementation of On Bayesian Upper Confidence Bounds for Bandit Problems, Kaufman & al,
        from http://proceedings.mlr.press/v22/kaufmann12/kaufmann12.pdf
        :param T: number of rounds
        :param a: First parameter of the Beta prior probability distribution
        :param b: Second parameter of the Beta prior probability distribution
        :param c: Parameter for the quantiles. Default value c=0
        :return: Reward obtained by the policy and sequence of the arms choosed
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        quantiles = np.zeros(self.nb_arms)
        for t in range(T):
            for k in range(self.nb_arms):
                if Na[k] >= 1:
                    quantiles[k] = beta.ppf(1-1/(t*np.log(T)**c), Sa[k] + a, b + Na[k] - Sa[k])
                else:
                    quantiles[k] = beta.ppf(1-1/(t*np.log(T)**c), a, b)
            arm = rd_argmax(quantiles)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
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
