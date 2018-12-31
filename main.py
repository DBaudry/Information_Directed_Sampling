# Importation
import expe as exp
import numpy as np
import MAB as mab

np.random.seed(123)
if __name__ == '__main__':
    exp.check_finite(p, q, R, theta=0, N=10, T=1000)
    p1 = [0.4, 0.5, 0.7, 0.8, 0.90]
    N1 = 1
    my_MAB = mab.BetaBernoulliMAB(p1)
    print(my_MAB.IDS_approx(200, 1000))