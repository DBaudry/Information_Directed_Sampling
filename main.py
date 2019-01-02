# Importation
import expe as exp
import numpy as np
from tqdm import tqdm
import MAB as mab

np.random.seed(123)
if __name__ == '__main__':
    exp.beta_bernoulli_expe(T=1000, n_expe=10, n_arms=10, doplot=True)
