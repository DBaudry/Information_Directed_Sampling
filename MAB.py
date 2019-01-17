""" Packages import """
import numpy as np
import arms
from tqdm import tqdm
from utils import rd_argmax
import random
import inspect

mapping = {'B': arms.ArmBernoulli, 'beta': arms.ArmBeta, 'F': arms.ArmFinite, 'G': arms.ArmGaussian}


class GenericMAB:
    """
    Generic class for arms that defines general methods
    """

    def __init__(self, methods, p):
        """
        Initialization of the arms
        :param methods: string, probability distribution of each arm
        :param p: np.array or list, parameters of the probability distribution of each arm
        """
        self.MAB = self.generate_arms(methods, p)
        self.nb_arms = len(self.MAB)
        self.means = [el.mean for el in self.MAB]
        self.mu_max = np.max(self.means)

    @staticmethod
    def generate_arms(methods, p):
        """
        Method for generating different arms
        :param methods: string, probability distribution of each arm
        :param p: np.array or list, parameters of the probability distribution of each arm
        :return: list of class objects, list of arms
        """
        arms_list = list()
        for i, m in enumerate(methods):
            args = [p[i]] + [[np.random.randint(1, 312414)]]
            args = sum(args, []) if type(p[i]) == list else args
            try:
                alg = mapping[m]
                arms_list.append(alg(*args))
            except Exception:
                raise NotImplementedError
        return arms_list

    def regret(self, reward, T):
        """
        Compute the regret of a single experiment
        :param reward: np.array, the array of reward obtained from the policy up to time T
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """
        return self.mu_max * np.arange(1, T + 1) - np.cumsum(reward)

    def MC_regret(self, method, N, T, param_dic):
        """
        Implementation of Monte Carlo method to approximate the expectation of the regret
        :param method: string, method used (UCB, Thomson Sampling, etc..)
        :param N: int, number of independent Monte Carlo simulation
        :param T: int, time horizon
        :param param_dic: dict, parameters for the different methods, can be the value of rho for UCB model or an int
        corresponding to the number of rounds of exploration for the ExploreCommit method
        """
        mc_regret = np.zeros(T)
        try:
            alg = self.__getattribute__(method)
            args = inspect.getfullargspec(alg)[0][2:]
            args = [T] + [param_dic[method][i] for i in args]
            for _ in tqdm(range(N), desc='Computing ' + str(N) + ' simulations'):
                mc_regret += self.regret(alg(*args)[0], T)
        except Exception:
            raise NotImplementedError
        return mc_regret / N

    def init_lists(self, T):
        """
        Initialization of quantities of interest used for all methods
        :param T: int, time horizon
        :return: - Sa: np.array, cumulative reward of arm a
                 - Na: np.array, number of times a has been pulled
                 - reward: np.array, rewards
                 - arm_sequence: np.array, arm chose at each step
        """
        Sa, Na, reward, arm_sequence = np.zeros(self.nb_arms), np.zeros(self.nb_arms), np.zeros(T), np.zeros(T)
        return Sa, Na, reward, arm_sequence

    def update_lists(self, t, arm, Sa, Na, reward, arm_sequence):
        """
        Update all the parameters of interest after choosing the correct arm
        :param t: int, current time/round
        :param arm: int, arm chose at this round
        :param Sa:  np.array, cumulative reward array up to time t-1
        :param Na:  np.array, number of times arm has been pulled up to time t-1
        :param reward: np.array, rewards obtained with the policy up to time t-1
        :param arm_sequence: np.array, arm chose at each step up to time t-1
        """
        Na[arm], arm_sequence[t], new_reward = Na[arm] + 1, arm, self.MAB[arm].sample()
        reward[t], Sa[arm] = new_reward, Sa[arm] + new_reward

    def RandomPolicy(self, T):
        """
        Implementation of a random policy consisting in randomly choosing one of the available arms. Only useful
        for checking that the behavior of the different policies is normal
        :param T:  int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        for t in range(T):
            arm = random.randint(0, self.nb_arms - 1)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence

    def ExploreCommit(self, T, m):
        """
        Implementation of Explore-then-Commit algorithm
        :param T: int, time horizon
        :param m: int, number of rounds before choosing the best action
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        for t in range(m * self.nb_arms):
            arm = t % self.nb_arms
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        arm = np.argmax(Sa / Na)
        for t in range(m * self.nb_arms + 1, T):
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence

    def UCB1(self, T, rho):
        """
        Implementation of UCB1 algorithm
        :param T: int, time horizon
        :param rho: float, parameter for balancing between exploration and exploitation
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                arm = rd_argmax(Sa / Na + rho * np.sqrt(np.log(t + 1) / 2 / Na))
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence

    def UCB_Tuned(self, T):
        """
        Implementation of UCB-tuned algorithm
        :param T: int, time horizon
        :param rho: float, parameter for balancing between exploration and exploitation
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        S, m = np.zeros(self.nb_arms), np.zeros(self.nb_arms)
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                for arm in range(self.nb_arms):
                    S[arm] = sum([r ** 2 for r in reward[np.where(arm_sequence == arm)]]) / Na[arm] - (
                            Sa[arm] / Na[arm]) ** 2
                    m[arm] = min(0.25, S[arm] + np.sqrt(2 * np.log(t + 1) / Na[arm]))
                arm = rd_argmax(Sa / Na + np.sqrt(np.log(t + 1) / Na * m))
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return reward, arm_sequence

    def MOSS(self, T, rho):
        """
        Implementation of Minimax Optimal Strategy in the Stochastic case (MOSS).
        :param T: int, time horizon
        :param rho: float, parameter for balancing between exploration and exploitation
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
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
        Implementation of Thomson Sampling algorithm
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        theta = np.zeros(self.nb_arms)
        for t in range(T):
            for k in range(self.nb_arms):
                if Na[k] >= 1:
                    theta[k] = np.random.beta(Sa[k] + 1, Na[k] - Sa[k] + 1)
                else:
                    theta[k] = np.random.uniform()
            arm = rd_argmax(theta)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            Sa[arm] += np.random.binomial(1, reward[t]) - reward[t]
        return reward, arm_sequence

    def IDSAction(self, delta, g):
        """
        Implementation of IDSAction algorithm as defined in Russo & Van Roy, p. 242
        :param delta: np.array, instantaneous regrets
        :param g: np.array, information gains
        :return: int, arm to pull
        """
        Q = np.zeros((self.nb_arms, self.nb_arms))
        IR = np.ones((self.nb_arms, self.nb_arms)) * np.inf
        q = np.linspace(0, 1, 1000)
        for a in range(self.nb_arms - 1):
            for ap in range(a + 1, self.nb_arms):
                if g[a] < 1e-6 or g[ap] < 1e-6:
                    return rd_argmax(-g)
                da, dap, ga, gap = delta[a], delta[ap], g[a], g[ap]
                qaap = q[rd_argmax(-(q * da + (1 - q) * dap) ** 2 / (q * ga + (1 - q) * gap))]
                IR[a, ap] = (qaap * (da - dap) + dap) ** 2 / (qaap * (ga - gap) + gap)
                Q[a, ap] = qaap
        amin = rd_argmax(-IR.reshape(self.nb_arms * self.nb_arms))
        a, ap = amin // self.nb_arms, amin % self.nb_arms
        b = np.random.binomial(1, Q[a, ap])
        arm = int(b * a + (1 - b) * ap)
        return arm
