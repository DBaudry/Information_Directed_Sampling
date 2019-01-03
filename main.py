# Importation
import expe as exp
import numpy as np
from tqdm import tqdm
import MAB as mab

np.random.seed(123)
if __name__ == '__main__':
    p, q, R = exp.build_finite_deterministic()
    # p, q, R = exp.build_finite(L=25, K=4, N=3)
    exp.check_finite(p, q, R, theta=0, N=1, T=10)
    # exp.beta_bernoulli_expe(T=1000, n_expe=10, n_arms=10, doplot=True)
