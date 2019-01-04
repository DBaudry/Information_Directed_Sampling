# Importation
import expe as exp
import numpy as np
from tqdm import tqdm
import MAB as mab
import matplotlib.pyplot as plt

np.random.seed(1234)
if __name__ == '__main__':
    p, q, R = exp.build_finite_deterministic()
    p, q, R = exp.build_finite(L=25, K=4, N=3)
    exp.check_finite(p, q, R, theta=0, N=100, T=1000)
    # exp.beta_bernoulli_expe(T=1000, n_expe=10, n_arms=10, doplot=True)
    # exp.beta_bernoulli_expe(T=1000, n_expe=100, n_arms=2, doplot=True)

