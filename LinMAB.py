import numpy as np
from utils import rd_argmax
from scipy.stats import norm

class ArmGaussianLinear(object):
    def __init__(self, random_state=0):
        self.local_random = np.random.RandomState(random_state)

    def reward(self, arm):
        return np.dot(self.features[arm], self.real_theta) + self.local_random.normal(0, self.eta, 1)

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
    def __init__(self, model, s=10):
        self.model = model
        self.n_a = model.n_actions
        self.d = model.n_features
        self.features = model.features
        self.reward = model.reward
        self.eta = model.eta
        self.flag = False
        self.optimal_arm = None
        self.threshold = 0.9
        self.s = s

    def initPrior(self):
        mu_0 = np.zeros(self.d)
        sigma_0 = self.s * np.eye(self.d)
        return mu_0, sigma_0

    def TS(self, T):
        arm_sequence, reward = np.zeros(T), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        for t in range(T):
            theta_t = np.random.multivariate_normal(mu_t, sigma_t, 1).T
            a_t = rd_argmax(np.dot(self.features, theta_t))
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t] = r_t, a_t
        return reward, arm_sequence


    def LinUCB(self, T, lbda=10e-4, alpha=10e-1):
        arm_sequence, reward = np.zeros(T), np.zeros(T)
        a_t, A_t, b_t = np.random.randint(0, self.n_a - 1, 1)[0], lbda * np.eye(self.d), np.zeros(self.d)
        r_t = self.reward(a_t)
        for t in range(T):
            A_t += np.outer(self.features[a_t, :], self.features[a_t, :])
            b_t += r_t * self.features[a_t, :]
            inv_A = np.linalg.inv(A_t)
            theta_t = np.dot(inv_A, b_t)
            beta_t = alpha * np.sqrt(np.diagonal(np.dot(np.dot(self.features, inv_A), self.features.T)))
            a_t = rd_argmax(np.dot(self.features, theta_t) + beta_t)
            r_t = self.reward(a_t)
            arm_sequence[t], reward[t] = a_t, r_t
        return reward, arm_sequence

    def BayesUCB(self, T):
        arm_sequence, reward = np.zeros(T), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        for t in range(T):
            a_t = rd_argmax(np.dot(self.features, mu_t) + norm.ppf(t/(t+1)) *
                            np.sqrt(np.diagonal(np.dot(np.dot(self.features, sigma_t), self.features.T))))
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t] = r_t, a_t
        return reward, arm_sequence


    def GPUCB(self, T):
        arm_sequence, reward = np.zeros(T), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        for t in range(T):
            beta_t = 2 * np.log(self.n_a * ((t+1)*np.pi)**2 / 6 / 0.1)
            a_t = rd_argmax(np.dot(self.features, mu_t) +
                            np.sqrt(beta_t * np.diagonal(np.dot(np.dot(self.features, sigma_t), self.features.T))))
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t] = r_t, a_t
        return reward, arm_sequence

    def Tuned_GPUCB(self, T, c=3.23):
        arm_sequence, reward = np.zeros(T), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        for t in range(T):
            beta_t = c * np.log(t+1)
            #print(np.sqrt(beta_t), norm.ppf(t/(t+1)))
            a_t = rd_argmax(np.dot(self.features, mu_t) +
                            np.sqrt(beta_t*np.diagonal(np.dot(np.dot(self.features, sigma_t), self.features.T))))
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t] = r_t, a_t
        return reward, arm_sequence

    def updatePosterior(self, a, mu, sigma):
        f, r = self.features[a], self.reward(a)[0]
        s_inv = np.linalg.inv(sigma)
        ffT = np.outer(f, f)
        mu_ = np.dot(np.linalg.inv(s_inv + ffT / self.eta**2), np.dot(s_inv, mu) + r * f / self.eta**2)
        sigma_ = np.linalg.inv(s_inv + ffT/self.eta**2)
        return r, mu_, sigma_

    def computeIDS(self, mu_t, sigma_t, M):
        thetas = np.random.multivariate_normal(mu_t, sigma_t, M)
        mu = np.mean(thetas, axis=0)
        theta_hat = np.argmax(np.dot(self.features, thetas.T), axis=0)
        theta_hat_ = [thetas[np.where(theta_hat==a)] for a in range(self.n_a)]
        p_a = np.array([len(theta_hat_[a]) for a in range(self.n_a)])/M
        if np.max(p_a) >= self.threshold:
            self.optimal_arm = np.argmax(p_a)
            arm = self.optimal_arm
        else:
            mu_a = np.nan_to_num(np.array([np.mean([theta_hat_[a]], axis=1).squeeze() for a in range(self.n_a)]))
            L_hat = np.sum(np.array([p_a[a]*np.outer(mu_a[a]-mu, mu_a[a]-mu) for a in range(self.n_a)]), axis=0)
            rho_star = np.sum(np.array([p_a[a]*np.dot(self.features[a], mu_a[a]) for a in range(self.n_a)]), axis=0)
            v = np.array([np.dot(np.dot(self.features[a], L_hat), self.features[a].T) for a in range(self.n_a)])
            delta = np.array([rho_star - np.dot(self.features[a], mu) for a in range(self.n_a)])
            arm = rd_argmax(-delta**2/v)
        return arm, p_a

    def IDS_sample(self, T, M=10000):
        mu_t, sigma_t = self.initPrior()
        reward, arm_sequence = np.zeros(T), np.zeros(T)
        p_a = np.zeros(self.n_a)
        for t in range(T):
           if not self.flag:
               if np.max(p_a) >= self.threshold:
                   self.flag = True
                   a_t = self.optimal_arm
               else:
                   a_t, p_a = self.computeIDS(mu_t, sigma_t, M)
           else:
               a_t = self.optimal_arm
           r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
           reward[t], arm_sequence[t] = r_t, a_t
        return reward, arm_sequence

