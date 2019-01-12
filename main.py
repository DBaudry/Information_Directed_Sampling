# Importation
import expe as exp
import numpy as np
from tqdm import tqdm
import MAB as mab
import matplotlib.pyplot as plt
import BernoulliMAB as BM

np.random.seed(1234)
if __name__ == '__main__':
    # p, q, R = exp.build_finite(L=100, K=10, N=20)
    # exp.check_finite(p, q, R, theta=0, N=100, T=1000)
    # b1, b2 = (np.ones(10)*10).astype(int), (np.ones(10)*1).astype(int)
    # exp.beta_bernoulli_expe(T=500, n_expe=100, n_arms=10, n_iter=1, b1=b1, b2=b2, doplot=True)
    b1, b2 = (np.ones(2)*10).astype(int), (np.ones(2)*1).astype(int)
    exp.beta_bernoulli_expe(T=1000, n_expe=100, n_arms=2, n_iter=1, b1=b1, b2=b2, doplot=True)
    # exp.check_gaussian(n_expe=100, n_arms=2, T=1000, doplot=True)
    # exp.LinearGaussianMAB(10, 5, 25, 250)

    # beta_1 = (np.ones(10)*10).astype(int)
    # beta_2 = (np.ones(10)*1).astype(int)
    # p = [0.6, 0.15, 0.97, 0.07, 0.83, 0.37, 0.55, 0.04, 0.47, 0.88]
    # my_mab = BM.BetaBernoulliMAB(p)
    # print(my_mab.IDS_approx(T=1000, N_steps=10000, beta1=beta_1, beta2=beta_2, display_results=True))
