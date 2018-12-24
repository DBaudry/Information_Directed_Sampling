# Importation
import expe as exp

if __name__ == '__main__':
    exp.comprehension()



# methods=['beta','B','Exp','F','Exp','beta','B']
# methods=['beta','B']
# param=[(0.5,0.5),0.4,(0.2,1.),[np.array([0.,0.4,0.6,0.8]),np.array([1./5.]*5)],(0.3,0.9),(0.4,0.5),0.5]
# param=[(0.5,0.8),0.3,(0.2,1.),[np.array([0.2,0.4,0.6,0.8,1.]),np.array([1./5.]*5)],(0.3,0.9),(0.4,0.5),0.4]
# param=[(0.5,0.5),0.5]

# my_MAB_2=mab.GenericMAB(methods,param)
# print(my_MAB_2.means)
# N2=100

# plt.figure(2)
# plt.plot(my_MAB_2.MC_regret(method='UCB1',N=N2,T=1000,rho=0.2),label='UCB1')
# plt.plot(my_MAB_2.MC_regret(method='UCB1',N=N2,T=1000,rho=0.),label='Naive')
# plt.plot(my_MAB_2.MC_regret(method='TS',N=N2,T=1000),label='TS')
# plt.ylabel('Cumulative Regret')
# plt.xlabel('Rounds')
# plt.legend()
# plt.show()
