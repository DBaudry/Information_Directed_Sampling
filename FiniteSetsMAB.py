from MAB import *


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
        self.flag = False
        self.optimal_arm = None
        self.threshold = 0.999

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
            if pa[a_star] > self.threshold:
                self.flag = True
                self.optimal_arm = a_star
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
        """
        :return: Optimal expected reward for a given prior
        """
        R = 0
        for a in range(self.nb_arms):
            for y in range(self.N):
                R += joint_P[a, a, y] * self.R[y]
        return R

    def get_R(self, PY):
        """
        :param PY: array of shape (K,N), Probability of outcome Y while pulling arm A
        :return: Expected reward for a given prior
        """
        R = np.zeros(self.nb_arms)
        for a in range(self.nb_arms):
                R[a] += PY[a, :] @ self.R
        return R

    def get_g(self, joint, pa, py):
        """
        :param joint: Joint distribution P_a(y,a_star)
        :param pa: Distribution of the optimal action
        :param py: Probability of outcome Y while pulling arm A
        :return: Information Gain
        """
        g = np.zeros(self.nb_arms)
        for a in range(self.nb_arms):
            for a_star in range(self.nb_arms):
                if pa[a_star] > 0.00001:
                    for y in range(self.N):
                        g[a] += joint[a, a_star, y] * np.log(joint[a, a_star, y] / (pa[a_star] * py[a, y]))
                    # print('a:', a, 'y:', y, 'a_star:', a_star)
                    # print('joint[a, a_star, y]:', joint[a, a_star, y])
                    # print('pa[a_star]', pa[a_star])
                    # print('py[a, y]', py[a, y])
        return g

    def IR(self):
        pa = self.get_pa_star()
        py = self.get_py()
        joint = self.get_joint_ay()
        # print('joint dans IR:', joint)
        print('pa dans IR: ', pa)
        print('py dans IR: ', py)
        R_star = self.get_R_star(joint)
        delta = np.zeros(self.nb_arms) + R_star - self.get_R(py)
        g = self.get_g(joint, pa, py)
        return delta, g

    def update_prior(self, a, y):
        for theta in range(self.L):
            self.prior[theta] *= self.q_theta[theta, a, y]
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
            if not self.flag:
                delta, g = self.IR()
                arm = self.IDSAction(delta, g)
                print('delta:', delta, 'T: ', t)
                print('g:', g)
            else:
                arm = self.optimal_arm
            self.update_lists(t, arm, Sa, Na, Y, arm_sequence)
            reward[t] = self.R[int(Y[t])]
            self.update_prior(arm, int(Y[t]))
        return reward, arm_sequence
