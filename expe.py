# File containing different experiments that we can run

# Importation
import numpy as np
import MAB as mab
import matplotlib.pyplot as plt


def comprehension():
    p1 = [0.05, 0.4, 0.7, 0.90]
    N1 = 50
    my_MAB = mab.BetaBernoulliMAB(p1)
    plt.plot(my_MAB.MC_regret(method='UCB1', N=N1, T=1000, param=0.2), label='UCB1')
    plt.plot(my_MAB.MC_regret(method='RandomPolicy', N=N1, T=1000), label='Random')
    plt.plot(my_MAB.MC_regret(method='TS', N=N1, T=1000), label='TS')
    plt.ylabel('Cumulative Regret')
    plt.xlabel('Rounds')
    plt.legend()
    plt.show()
    return 0
 #       plt.subplot(121 + i)
 #       plt.plot(my_MAB.MC_regret(method='UCB1', N=N1, T=1000, rho=0.2), label='UCB1')
 #       plt.plot(my_MAB.MC_regret(method='Random', N=N1, T=1000, rho=0.), label='Random')
 #       plt.plot(my_MAB.MC_regret(method='TS', N=N1, T=1000), label='TS')
 #       plt.plot(my_MAB.Cp * np.log(np.arange(1, 1001)))
 #       plt.ylabel('Cumulative Regret')
 #       plt.xlabel('Rounds')
 #       plt.legend()
 #   plt.show()


def sanity_check_expe():
    p1 = [0.05, 0.4, 0.7, 0.90]
    p2 = [0.25, 0.27, 0.32, 0.40, 0.42]
    N1 = 50
    plt.figure(1)
    for i, p in enumerate([p1, p2]):
        my_MAB = mab.BetaBernoulliMAB(p)
        print(my_MAB.MAB)
        print(my_MAB.Cp)
        plt.subplot(121 + i)
        plt.plot(my_MAB.MC_regret(method='UCB1', N=N1, T=1000, rho=0.2), label='UCB1')
        plt.plot(my_MAB.MC_regret(method='Random', N=N1, T=1000, rho=0.), label='Random')
        plt.plot(my_MAB.MC_regret(method='TS', N=N1, T=1000), label='TS')
        plt.plot(my_MAB.Cp * np.log(np.arange(1, 1001)))
        plt.ylabel('Cumulative Regret')
        plt.xlabel('Rounds')
        plt.legend()
    plt.show()