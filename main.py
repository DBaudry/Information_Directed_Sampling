# Importation
import expe as exp
import numpy as np

np.random.seed(123)
if __name__ == '__main__':
    p, q, R = exp.build_finite_deterministic()
    exp.check_finite(p, q, R, theta=0, N=10, T=1000)

