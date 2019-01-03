# Importation
import expe as exp
import numpy as np
from tqdm import tqdm
import MAB as mab
import matplotlib.pyplot as plt

np.random.seed(1234)
if __name__ == '__main__':
    exp.beta_bernoulli_expe(T=1000, n_expe=100, n_arms=2, doplot=True)
    p = [0.25, 0.5, 0.7, 0.98]
    my_mab = mab.BetaBernoulliMAB(p)
    res = my_mab.IDS_approx(1000, 1000, display_results=True)
    regret = my_mab.mu_max*np.arange(1, 1001)-res[0].cumsum()
    print(res[1])
    plt.plot(regret)
    plt.show()