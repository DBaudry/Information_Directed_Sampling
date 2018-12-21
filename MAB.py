# Importations
import numpy as np
import arms
from tqdm import tqdm
from utils import rd_argmax
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

    def MC_regret(self, method, N, T, rho=0.2):
        """
        Monte Carlo method for approximating the expectation of the regret.
        :param method: Method used (UCB, Thomson Sampling, etc..)
        :param N: Number of independent experiments used for the Monte Carlo
        :param T: Number of rounds for each experiment
        :param rho: Useful parameter for the UCB policy
        :return: Averaged regret over N independent experiments
        """
        MC_regret = np.zeros(T)
        for _ in tqdm(range(N), desc='Computing ' + str(N) + ' simulations'):
            if method == 'UCB1':
                MC_regret += self.regret(self.UCB1(T, rho)[0], T)
            elif method == 'TS':
                MC_regret += self.regret(self.TS(T)[0], T)
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
        :return:
        """
        Na[arm] += 1  # Updating the number of times the arm "arm" was selected
        arm_sequence[t] = arm  # the arm "arm" was selected at round t
        new_reward = self.MAB[arm].sample()  # obtaining the new reward by pulling the arm a
        reward[t] = new_reward  # adding this new reward to the array of rewards
        Sa[arm] += new_reward  # updating the cumulated reward of the arm "arm"

    def ExploreCommit(self, m, T):
        """
        Explore-then-Commit algorithm as presented in Bandits Algorithms, Lattimore, Chapter 6
        :param m: Number of rounds before choosing the best action
        :param T: Number of steps
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
                arm = rd_argmax(Sa / Na + rho * np.sqrt(4/Na * np.log(max(1, T/(self.nb_arms * Na)))))
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence

    def kgf(self, x):
        """
        :param x: float
        :return: kgf(x) used to select the best arm with KG algorithm (see below)
        """
        return norm.cdf(x) * x + norm.pdf(x)

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
                rmse = np.sqrt(
                    1 / Na * np.array([np.sum((reward[np.where(arm_sequence == arm)] - (Sa / Na)[arm]) ** 2) for arm
                                       in range(self.nb_arms)]))
                x = np.array(
                    [(Sa / Na)[i] - np.max(list(Sa / Na)[:i] + list(Sa / Na)[i + 1:]) for i in range(self.nb_arms)])
                v = rmse * self.kgf(-np.absolute(x / (rmse + 10e-9)))
                # print(v)
                print(Sa / Na + (T - t) * v)
                arm = np.argmax(Sa / Na + (T - t) * v)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return np.array(reward), np.array(arm_sequence)

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

    def IDSAction(self,delta,g):
        Q = np.zeros((self.nb_arms, self.nb_arms))
        for a in range(self.nb_arms):
            for ap in range(self.nb_arms):
                da, dap = delta[a], delta[ap]
                ga, gap = g[a], g[ap]
                q1 = dap/(da+dap)
                q2 = -1.
                if ga != gap:
                    q2 = q1+2*gap/(ga-gap)
                if 0 <= q1 <= 1:
                    Q[a, ap] = q1
                elif 0 <= q2 <= 1:
                    Q[a, ap] = q2
                elif da**2/ga > dap**2/gap:
                    Q[a, ap] = 1
                else:
                    Q[a, ap] = 0
        amin = np.argmin(Q.reshape(self.nb_arms*self.nb_arms))
        a, ap = amin//self.nb_arms, amin % self.nb_arms
        b = np.random.binomial(1, Q[a, ap])
        return int(b*a+(1-b)*ap)


class BetaBernoulliMAB(GenericMAB):
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
            arm = np.argmax(quantiles)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence

    @staticmethod
    def kl(x, y):
        """
        Implementation of the Kullback-Leibler divergence
        """
        return x * np.log(x/y) + (1-x) * np.log((1-x)/(1-y))

    def IR(self, b1, b2):

        def joint_cdf(x):
            result = 1
            for i in range(self.nb_arms):
                result = result*beta.cdf(x, b1[i], b2[i])
            return result

        def G(x, a):
            return b1[a]/(b1[a]+b2[a])*beta.cdf(x, b1[a]+1, b2[a])

        def dp_star(x, a):
            return beta.pdf(x, b1[a], b2[a])*joint_cdf(x)/beta.cdf(x, b1[a], b2[a])

        def p_star(a):
            return integrate.quad(lambda x: dp_star(x, a), 0., 1.)[0]  # result is a tuple (value, UB error)

        def MAA(a, p):
            return integrate.quad(lambda x: x*dp_star(x, a), 0., 1.)[0]/p[a]

        def MAAP(a, ap, p):
            return integrate.quad(lambda x: dp_star(x, a)*G(x, ap)/beta.cdf(x, b1[ap], b2[ap]), 0., 1.)[0]/p[a]

        def g(a, p, M):
            gp = p*(M[a]*np.log(M[a]*(b1+b2)/b1)+(1-M[a])*np.log((1-M[a])*(b1+b2)/b2))
            return gp.sum()

        # To be completed
        # ps = np.array([p_star(a) for a in range(self.nb_arms)])
        # Ma = np.array([MAA(a, ps) for a in range(self.nb_arms)])
        # Map = np.array([[MAAP(a, ap, ps) for a in range(self.nb_arms)] for ap in range(self.nb_arms)])
        # rho = (ps*Ma).sum()
        # delta = rho-b1/(b1+b2)
        # g = np.array([g(a, ps, Map) for a in range(self.nb_arms)])
        # return delta, g


class FiniteMAB(GenericMAB):
    def __init__(self, method, param, theta_space):
        super().__init__(method, param)
        self.theta = theta_space