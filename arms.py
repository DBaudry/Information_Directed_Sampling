""" Packages import """
import numpy as np


class AbstractArm(object):
    def __init__(self, mean, variance, random_state):
        """
        :param mean: float, expectation of the arm
        :param variance: float, variance of the arm
        :param random_state: int, seed to make experiments reproducible
        """
        self.mean = mean
        self.variance = variance
        self.local_random = np.random.RandomState(random_state)

    def sample(self):
        pass


class ArmBernoulli(AbstractArm):
    def __init__(self, p, random_state=0):
        """
        :param p: float, mean parameter
        :param random_state: int, seed to make experiments reproducible
        """
        self.p = p
        super(ArmBernoulli, self).__init__(mean=p,
                                           variance=p * (1. - p),
                                           random_state=random_state)

    def sample(self):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        return self.local_random.rand(1) < self.p


class ArmBeta(AbstractArm):
    def __init__(self, a, b, random_state=0):
        """
        :param a: int, alpha coefficient in beta distribution
        :param b: int, beta coefficient in beta distribution
        :param random_state: int, seed to make experiments reproducible
        """
        self.a = a
        self.b = b
        super(ArmBeta, self).__init__(mean=a/(a + b),
                                      variance=(a * b)/((a + b) ** 2 * (a + b + 1)),
                                      random_state=random_state)

    def sample(self):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        return self.local_random.beta(self.a, self.b, 1)


class ArmGaussian(AbstractArm):
    def __init__(self, mu, eta, random_state=0):
        """
        :param mu: float, mean parameter in gaussian distribution
        :param eta: float, std parameter in gaussian distribution
        :param random_state: int, seed to make experiments reproducible
        """
        self.mu = mu
        self.eta = eta
        super(ArmGaussian, self).__init__(mean=mu,
                                          variance=eta**2,
                                          random_state=random_state)

    def sample(self):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        return self.local_random.normal(self.mu, self.eta, 1)


class ArmFinite(AbstractArm):
    def __init__(self, X, P, random_state=0):
        """
        :param X: np.array, support of the distribution
        :param P: np.array, associated probabilities
        :param random_state: int, seed to make experiments reproducible
        """
        self.X = X
        self.P = P
        mean = np.sum(X * P)
        super(ArmFinite, self).__init__(mean=mean,
                                        variance=np.sum(X ** 2 * P) - mean ** 2,
                                        random_state=random_state)

    def sample(self):
        """
        Sampling strategy for an arm with a finite support and the associated probability distribution
        :return: float, a sample from the arm
        """
        i = self.local_random.choice(len(self.P), size=1, p=self.P)
        reward = self.X[i]
        return reward
