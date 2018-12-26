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
        plt.plot(my_MAB.MC_regret(method='UCB1', N=N1, T=1000, param=0.2), label='UCB1')
        plt.plot(my_MAB.MC_regret(method='Random', N=N1, T=1000, param=0.), label='Random')
        plt.plot(my_MAB.MC_regret(method='TS', N=N1, T=1000), label='TS')
        plt.plot(my_MAB.Cp * np.log(np.arange(1, 1001)))
        plt.ylabel('Cumulative Regret')
        plt.xlabel('Rounds')
        plt.legend()
    plt.show()

def check_finite(prior, q, R,theta, N, T):
    method = ['F']*q.shape[1]
    param = [[np.arange(q.shape[2]), q[theta, i, :]] for i in range(q.shape[1])]
    my_MAB = mab.FiniteSets(method, param, q, prior, R)
    print(prior)
    print(R)
    print(my_MAB.Ta)
    param2 = [[R, q[theta, i, :]] for i in range(q.shape[1])]
    check_MAB = mab.GenericMAB(method, param2)
    regret_IDS = my_MAB.MC_regret(method='IDS', N=N, T=T, param=0.2)
    plt.plot(regret_IDS, label='IDS')
    plt.plot(check_MAB.MC_regret(method='UCB1', N=N, T=T, param=0.2), label='UCB1')
    plt.plot(check_MAB.MC_regret(method='TS', N=N, T=T), label='TS')
    plt.ylabel('Cumulative Regret')
    plt.xlabel('Rounds')
    plt.legend()
    plt.show()