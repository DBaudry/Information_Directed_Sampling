# Importations
import numpy as np
import arms
from tqdm import tqdm

class GenericMAB:
    def __init__(self,method,param):
        self.MAB=self.generate_arms(method,param)
        self.nb_arms=len(self.MAB)
        self.means=[el.mean for el in self.MAB]
        self.mu_max=np.max(self.means)

    @staticmethod
    def generate_arms(meth, par):
        arms_list = list()
        for i,m in enumerate(meth):
            print(i, m, par[i])
            p = par[i]
            if m == 'B':
                arms_list.append(arms.ArmBernoulli(p, random_state = np.random.randint(1, 312414)))
            elif m == 'beta':
                arms_list.append(arms.ArmBeta(a=p[0],b=p[1], random_state = np.random.randint(1, 312414)))
            elif m == 'F':
                arms_list.append(arms.ArmFinite(X=p[0],P=p[1], random_state=np.random.randint(1, 312414)))
            elif m == 'Exp':
                arms_list.append(arms.ArmExp(L=p[0],B=p[1], random_state=np.random.randint(1, 312414)))
            else:
                raise NameError('This method is not implemented, available methods are defined in generate_arms method')
        return arms_list


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
        Na[arm] += 1 # Updating the number of times the arm "arm" was selected
        arm_sequence[t] = arm # the arm "arm" was selected at round t
        new_reward = self.MAB[arm].sample() # obtaining the new reward by pulling the arm a
        reward[t] = new_reward  # adding this new reward to the array of rewards
        Sa[arm] += new_reward # updating the cumulated reward of the arm "arm"

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
        for t in range(m*self.nb_arms + 1,T):
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return np.array(reward), np.array(arm_sequence)


    def UCB1(self, T, rho):
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                arm = np.argmax(Sa/Na+rho*np.sqrt(np.log(t+1)/2/Na))
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return np.array(reward), np.array(arm_sequence)


    def MOSS(self, T, rho):
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                arm = np.argmax(Sa / Na + rho * np.sqrt(4/Na * np.log(max(1,T/(self.nb_arms * Na)))))
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return np.array(reward), np.array(arm_sequence)


    def TS(self,T):
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        theta = np.zeros(self.nb_arms)
        for t in range(T):
            for k in range(self.nb_arms):
                if Na[k] >= 1:
                    theta[k] = np.random.beta(Sa[k]+1,Na[k]-Sa[k]+1)
                else:
                    theta[k] = np.random.uniform()
            arm = np.argmax(theta)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
            Sa[arm] += np.random.binomial(1,reward[t])-reward[t]
        return np.array(reward), np.array(arm_sequence)


    def regret(self, reward, T):
        return self.mu_max * np.arange(1,T+1) - np.cumsum(reward)


    def MC_regret(self,method, N, T,rho=0.2):
        MC_regret = np.zeros(T)
        for _ in tqdm(range(N), desc='Computing ' + str(N) + ' simulations'):
            if method == 'UCB1':
                MC_regret += self.regret(self.UCB1(T,rho)[0],T)
            elif method == 'TS':
                MC_regret += self.regret(self.TS(T)[0],T)
        return MC_regret/N


class BinomialMAB(GenericMAB):
    def __init__(self,p):
        super().__init__(method=['B']*len(p),param=p)
        self.Cp = sum([(self.mu_max-x)/self.kl(x,self.mu_max) for x in self.means if x!=self.mu_max])


    def TS(self,T):
        Sa, Na, reward, arm_sequence = self.init_lists(T)
        theta = np.zeros(self.nb_arms)
        for t in range(T):
            for k in range(self.nb_arms):
                if Na[k] >= 1:
                    theta[k] = np.random.beta(Sa[k]+1,Na[k]-Sa[k]+1)
                else:
                    theta[k] = np.random.uniform()
            arm = np.argmax(theta)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence)
        return np.array(reward), np.array(arm_sequence)

    @staticmethod
    def kl(x,y):
        return x * np.log(x/y) + (1-x) * np.log((1-x)/(1-y))

