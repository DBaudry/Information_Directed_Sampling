# Importations
import numpy as np
import arms
from tqdm import tqdm
from utils import rd_argmax
import random
from scipy.stats import beta, norm
import scipy.integrate as integrate


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
            elif method == 'ExploreCommit':
                MC_regret += self.regret(self.ExploreCommit(m=param, T=T)[0], T)
            elif method == 'BayesUCB':
                MC_regret += self.regret(self.BayesUCB(T=T, a=1., b=1., c=0.)[0], T)
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

    def update_lists(self, t, arm, Sa, Na, reward, arm_sequence, tsNoBinary=False):
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
        # obtaining the new reward by pulling the arm a
        new_reward = np.random.binomial(1, self.MAB[arm].sample(), 1)[0] * 1.0 if tsNoBinary else self.MAB[arm].sample()
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
            arm = np.argmax(theta)
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
        Implementation of the Kullback-Leibler divergence
        """
        return x * np.log(x/y) + (1-x) * np.log((1-x)/(1-y))

    def IR(self, b1, b2):
        """
        Implementation of the Information Ratio for bernoulli bandits with beta prior
        :param b1: list, first parameter of the beta distribution for each arm
        :param b2: list, second parameter of the beta distribution for each arm
        :return: the two components of the Information ration delta and g
        """
        def joint_cdf(x):
            result = 1.
            for i in range(self.nb_arms):
                result = result*beta.cdf(x, b1[i], b2[i])
            return result

        def G(x, a):
            return b1[a]/(b1[a]+b2[a])*beta.cdf(x, b1[a]+1, b2[a])

        def dp_star(x, a):
            return beta.pdf(x, b1[a], b2[a])*joint_cdf(x)/beta.cdf(x, b1[a], b2[a])

        def p_star(a):
            return integrate.quad(lambda x: dp_star(x, a), 0., 1., epsabs=1e-2)[0]  # result is a tuple (value, UB error)

        def MAA(a, p):
            return integrate.quad(lambda x: x*dp_star(x, a), 0., 1., epsabs=1e-2)[0]/p[a]

        def MAAP(ap, a, p):
            return integrate.quad(lambda x: dp_star(x, a)*G(x, ap)/beta.cdf(x, b1[ap], b2[ap]), 0., 1., epsabs=1e-2)[0]/p[a]

        def g(a, p, M):
            gp = p*(M[a]*np.log(M[a]*(b1+b2)/b1)+(1-M[a])*np.log((1-M[a])*(b1+b2)/b2))
            return gp.sum()

        ps = np.array([p_star(a) for a in range(self.nb_arms)])
        ma = np.array([MAA(a, ps) for a in range(self.nb_arms)])
        maap = np.array([[MAAP(a, ap, ps) for a in range(self.nb_arms)] for ap in range(self.nb_arms)])
        rho = (ps*ma).sum()
        delta = rho-b1/(b1+b2)
        g = np.array([g(a, ps, maap.T) for a in range(self.nb_arms)])
        return delta, g

    def IDS(self, T):
        """
        Implementation of the Information Directed Sampling for Beta-Bernoulli bandits
        :param T: number of rounds
        :return: Reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        beta_1 = np.zeros(self.nb_arms)+1.
        beta_2 = np.zeros(self.nb_arms)+1.
        for t in range(T):
            delta, g = self.IR(beta_1, beta_2)
            arm = self.IDSAction(delta, g)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            beta_1[arm] += reward[t]
            beta_2[arm] += 1-reward[t]
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
        '''
        theta in [1,L], Y in [1,N], A in [1,K]
        :param method: list with the types for each arm
        :param param: list with the parameters for each arm
        :param q_theta: L*K*N array with the probability of each outcome knowing theta
        :param R: mapping between outcomes and rewards
        '''
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
        '''
        :return: list of length K containing the lists of theta for which action a in [1,K] is optimal
        '''
        Ta = [[] for a in range(self.nb_arms)]
        for theta in range(self.L):
            a_theta = rd_argmax(np.dot(self.q_theta[theta, :, :], self.R))
            Ta[a_theta].append(theta)
        return Ta

    def get_pa_star(self):
        '''
        :return: array of shape K
         For a given prior, the probabilities that action a in [1,K] is the optimal action
        '''
        pa = np.zeros(self.nb_arms)
        for a_star in range(self.nb_arms):
            for x in self.Ta[a_star]:
                pa[a_star] += self.prior[x]
        return pa

    def get_py(self):
        '''
        :return: array of shape (K,N)
        Probability of outcome Y while pulling arm A for a given prior
        '''
        PY = np.zeros((self.nb_arms, self.N))
        for a in range(self.nb_arms):
            for y in range(self.N):
                for x in range(self.L):
                    PY[a, y] += self.prior[x]*self.q_theta[x, a, y]
        return PY

    def get_joint_ay(self):
        '''
        :return: Array of shape (K,K,N)
        Joint distribution of the outcome and the optimal arm while pulling arm a
        '''
        P_ay = np.zeros((self.nb_arms, self.nb_arms, self.N))
        for a in range(self.nb_arms):
            for a_star in range(self.nb_arms):
                for y in range(self.N):
                    for x in self.Ta[a_star]:
                        P_ay[a, a_star, y] += self.q_theta[x, a, y]*self.prior[x]
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


class GaussianMAP(GenericMAB):
    """
    TODO: BayesUCB to adapt for gaussian bandits
    """
    def __init__(self, p):
        super().__init__(method=['G']*len(p), param=p)
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
            arm = np.argmax(theta)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence, tsNoBinary=True)
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
        sigma = np.zeros(self.nb_arms)+1.
        for t in range(T):
            delta, v = self.IR(mu, sigma)
            arm = self.IDSAction(delta, v)
            eta = self.MAB[arm].eta
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            mu[arm] = (mu[arm]/sigma[arm]**2 + reward[t]/eta**2)/(1/(sigma[arm]**2) + eta**(-2))
            sigma[arm] = (1/(sigma[arm]**2) + eta**(-2))**(-1)
        return reward, arm_sequence

    def KG(self, T):
        """
        Implementation of Knowledge Gradient algorithm
        :param T: number of rounds
        :return: Reward obtained by the policy and sequence of the arms chosen
        """
        def kgf(x):
            return norm.cdf(x)*x+norm.pdf(x)

        Sa, Na, reward, arm_sequence = self.init_lists(T)
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                rmse = np.sqrt(
                    1 / Na * np.array([np.sum((reward[np.where(arm_sequence == arm)] - (Sa / Na)[arm]) ** 2)
                                       for arm in range(self.nb_arms)]))
                x = np.array(
                    [(Sa / Na)[i] - np.max(list(Sa / Na)[:i] + list(Sa / Na)[i + 1:]) for i in range(self.nb_arms)])
                v = rmse * kgf(-np.absolute(x / (rmse + 10e-9)))
                arm = rd_argmax(Sa / Na + (T - t) * v)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return np.array(reward), np.array(arm_sequence)

    def multi_simul_KG(self, M, arm_sequence, arm, reward, Na, Sa, std, delta, T, t):
        reward_temp = list(reward[np.where(arm_sequence == arm)])
        Na_temp, Sa_temp, v_temp = Na[arm], Sa[arm], 0
        for k in range(M):
            r_temp = self.MAB[arm].sample()
            reward_temp.append(r_temp)
            Na_temp += 1
            Sa_temp += r_temp
            std[arm] = np.std(reward_temp)
            delta[arm] = Sa_temp / Na_temp - np.max([(Sa / Na)[i] for i in range(self.nb_arms) if i != arm])
            v_temp += Sa_temp / Na_temp + (T - t) * (std[arm] * self.kgf(-np.absolute(delta[arm] / (std[arm] + 10e-9))))
        return std[arm], delta[arm], v_temp / M

    def KG_star(self, T, lbda=1):
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                std = np.array([np.std(reward[np.where(arm_sequence == arm)]) for arm in range(self.nb_arms)])
                delta = np.array(
                    [(Sa / Na)[i] - np.max(list(Sa / Na)[:i] + list(Sa / Na)[i + 1:]) for i in range(self.nb_arms)])
                r = (delta / (std + 10e-9)) ** 2
                m_lower = lbda / (4 * std ** 2 + 10e-9) * (-1 + r + np.sqrt(1 + 6 * r + r ** 2))
                m_higher = lbda / (4 * std ** 2 + 10e-9) * (1 + r + np.sqrt(1 + 10 * r + r ** 2))
                v = np.zeros(self.nb_arms)
                for arm in range(self.nb_arms):
                    if T - t <= m_lower[arm]:
                        M = T - t
                    elif (delta[arm] == 0) or (m_higher[arm] <= 1):
                        M = 1
                    else:
                        M = np.ceil(0.5 * ((m_lower + m_higher)[arm])).astype(int)  # approximation
                    std[arm], delta[arm], v[arm] = self.multi_simul_KG(M, arm_sequence, arm, reward, Na, Sa, std, delta,
                                                                       T, t)
                arm = rd_argmax(v)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return np.array(reward), np.array(arm_sequence)
