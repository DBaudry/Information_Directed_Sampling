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
        :param N: Number of independent Monte Carlo simulation
        :param T: Number of rounds for each simulation
        :param param: Parameters for the different methods, can be the value of rho for UCB model or an int
        corresponding to the number of rounds of exploration for the ExploreCommit method
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
        IR = np.ones((self.nb_arms, self.nb_arms)) * np.inf
        for a in range(self.nb_arms-1):
            for ap in range(a+1, self.nb_arms):
                da, dap = delta[a], delta[ap]
                ga, gap = g[a], g[ap]
                q2 = -1.
                if da != dap and ga != gap:
                    q2 = dap/(da-dap)-2*gap/(ga-gap)
                if 0 <= q2 <= 1:
                    Q[a, ap] = q2
                elif da**2/ga > dap**2/gap:
                    Q[a, ap] = 0
                elif da**2/ga == dap**2:
                    Q[a, ap] = np.random.choice([0, 1])
                else:
                    Q[a, ap] = 1
                IR[a, ap] = (Q[a, ap]*(da-dap)+dap)**2/(Q[a, ap]*(ga-gap)+gap)
        amin = rd_argmax(-IR.reshape(self.nb_arms*self.nb_arms))
        a, ap = amin//self.nb_arms, amin % self.nb_arms
        b = np.random.binomial(1, Q[a, ap])
        return int(b*a+(1-b)*ap)

    def IDS(self, T):
        raise NotImplementedError

    def BayesUCB(self, T, a, b, c=0.):
        raise NotImplementedError
