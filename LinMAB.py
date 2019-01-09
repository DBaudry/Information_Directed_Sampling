import numpy as np
from utils import rd_argmax

class ArmGaussianLinear(object):
    def __init__(self, random_state=0):
        self.local_random = np.random.RandomState(random_state)

    def reward(self, arm):
        return np.dot(self.features[arm], self.real_theta) + self.eta * self.local_random.normal(0, 1, 1)

    def rewards_plot(self):
        D = np.dot(self.features, self.real_theta)
        d = np.argsort(-D)
        dict = {'Arm '+str(d[i]): round(D[d[i]], 3) for i in range(self.n_actions)}
        print("{" + "\n".join("{}: {}".format(k, v) for k, v in dict.items()) + "}")

    def best_arm_reward(self):
        D = np.dot(self.features, self.real_theta)
        return np.max(D)

    @property
    def n_features(self):
        return self.features.shape[1]

    @property
    def n_actions(self):
        return self.features.shape[0]


class PaperLinModel(ArmGaussianLinear):
    def __init__(self, u, n_features, n_actions, eta=1, sigma=10, random_state=0):
        super(PaperLinModel, self).__init__(random_state=random_state)
        self.eta = eta
        self.features = self.local_random.uniform(-u, u, (n_actions, n_features))
        self.real_theta = self.local_random.multivariate_normal(np.zeros(n_features), sigma*np.eye(n_features))

class LinMAB():
    def __init__(self, model):
        self.model = model
        self.n_a = model.n_actions
        self.d = model.n_features
        self.features = model.features
        self.reward = model.reward
        self.eta = model.eta


    def LinUCB(self, T, lbda=10e-4, alpha=10e-1):
        ''' Implements 'LinearUCB'

        :param T: int
            Time horizon
        :param lbda: float
            Penalization parameter in lasso regression
        :param alpha: float
            Parameter to adjust in the expression of beta for LinearUCB
        '''
        arm_sequence, reward = np.zeros(T), np.zeros(T)
        a_t, A_t, b_t = np.random.randint(0, self.n_a - 1, 1)[0], lbda * np.eye(self.d), np.zeros(self.d)
        r_t = self.reward(a_t)
        for t in range(T):
            A_t += np.outer(self.features[a_t, :], self.features[a_t, :])
            b_t += r_t * self.features[a_t, :]
            inv_A = np.linalg.inv(A_t)
            theta_t = np.dot(inv_A, b_t)
            beta_t = alpha * np.sqrt(np.diagonal(np.dot(np.dot(self.features, inv_A), self.features.T)))
            a_t = np.argmax(np.dot(self.features, theta_t) + beta_t)
            r_t = self.reward(a_t)
            arm_sequence[t], reward[t] = a_t, r_t
        return reward, arm_sequence


    def initIDS(self):
        mu_0 = np.zeros(self.d)
        sigma_0 = np.eye(self.d)
        return mu_0, sigma_0

    def updateIDS(self, a, mu, sigma):
        f, r = self.features[a], self.reward(a)[0]
        s_inv = np.linalg.inv(sigma)
        ffT = np.outer(f, f.T)
        mu_ = np.dot(np.linalg.inv(s_inv + ffT / self.eta**2), np.dot(s_inv, mu) + r * f / self.eta**2)
        sigma_ = np.linalg.inv(s_inv + ffT/self.eta**2)
        return r, mu_, sigma_

    def computeIDS(self, mu_t, sigma_t, M):
        thetas = np.random.multivariate_normal(mu_t, sigma_t, M)
        mu = np.mean(thetas, axis=0)
        theta_hat = np.argmax(np.dot(self.features, thetas.T), axis=0)
        theta_hat_ = [thetas[np.where(theta_hat==a)] for a in range(self.n_a)]
        theta_hat_card = np.array([len(theta_hat_[a]) for a in range(self.n_a)])
        p_a = theta_hat_card/M
        print(p_a)
        #print(sum(p_a))
        mu_a = np.nan_to_num(np.array([np.mean([theta_hat_[a]], axis=1).squeeze() for a in range(self.n_a)]))
        L_hat = np.sum(np.array([p_a[a]*np.outer(mu_a[a], mu_a[a].T) for a in range(self.n_a)]), axis=0)
        rho_star = np.sum(np.array([p_a[a]*np.dot(self.features[a], mu_a[a]) for a in range(self.n_a)]), axis=0)
        v = np.array([np.dot(self.features[a], np.dot(L_hat, self.features[a])) for a in range(self.n_a)])
        delta = np.array([rho_star - np.dot(self.features[a], mu) for a in range(self.n_a)])
        arm = rd_argmax(-delta**2/v)
        return arm

    def IDS(self, T, M=10000):
        mu_t, sigma_t = self.initIDS()
        reward, arm_sequence = np.zeros(T), np.zeros(T)
        for t in range(T):
           a_t = self.computeIDS(mu_t, sigma_t, M)
           r_t, mu_t, sigma_t = self.updateIDS(a_t, mu_t, sigma_t)
           reward[t], arm_sequence[t] = r_t, a_t
        print(arm_sequence)
        return reward, arm_sequence

