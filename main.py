import numpy as np
import MAB as mab
import matplotlib.pyplot as plt


p1=[0.05,0.4,0.7,0.90]
p2=[0.25,0.27,0.32,0.40,0.42]
N1=50
plt.figure(1)
for i,p in enumerate([p1,p2]):
    my_MAB=mab.Binomial_MAB(p)
    print(my_MAB.MAB)
    print(my_MAB.Cp)
    plt.subplot(121+i)
    plt.plot(my_MAB.MC_regret(method='UCB1',N=N1,T=1000,rho=0.2),label='UCB1')
    plt.plot(my_MAB.MC_regret(method='UCB1',N=N1,T=1000,rho=0.),label='Naive') #Naive method
    plt.plot(my_MAB.MC_regret(method='TS',N=N1,T=1000),label='TS')
    plt.plot(my_MAB.Cp*np.log(np.arange(1,1001)))
    plt.ylabel('Cumulative Regret')
    plt.xlabel('Rounds')
    plt.legend()
plt.show()

methods=['beta','B','Exp','F','Exp','beta','B']
#methods=['beta','B']
#param=[(0.5,0.5),0.4,(0.2,1.),[np.array([0.,0.4,0.6,0.8]),np.array([1./5.]*5)],(0.3,0.9),(0.4,0.5),0.5]
param=[(0.5,0.8),0.3,(0.2,1.),[np.array([0.2,0.4,0.6,0.8,1.]),np.array([1./5.]*5)],(0.3,0.9),(0.4,0.5),0.4]
#param=[(0.5,0.5),0.5]

my_MAB_2=mab.MAB(methods,param)
print(my_MAB_2.means)
N2=100

plt.figure(2)
plt.plot(my_MAB_2.MC_regret(method='UCB1',N=N2,T=1000,rho=0.2),label='UCB1')
plt.plot(my_MAB_2.MC_regret(method='UCB1',N=N2,T=1000,rho=0.),label='Naive')
plt.plot(my_MAB_2.MC_regret(method='TS',N=N2,T=1000),label='TS')
plt.ylabel('Cumulative Regret')
plt.xlabel('Rounds')
plt.legend()
plt.show()
