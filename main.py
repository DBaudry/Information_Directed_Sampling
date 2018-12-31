# Importation
import expe as exp
import numpy as np
import MAB as mab

np.random.seed(123)
if __name__ == '__main__':
    '''
    R = np.linspace(0., 1., 11)
    q = np.random.uniform(size=(2, 5, 11))
    for i in range(q.shape[0]):
        q[i] = np.apply_along_axis(lambda x : x/x.sum(), 1, q[i])
    p = np.array([0.35, 0.65])
    exp.check_finite(p, q, R, theta=0, N=10, T=1000)
    '''
    p1 = [0.4, 0.5, 0.7, 0.8, 0.90]
    N1 = 1
    my_MAB = mab.BetaBernoulliMAB(p1)
    print(my_MAB.IDS_approx(200, 1000))