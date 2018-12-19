import numpy as np
from linearmab_models import ToyLinearModel, ColdStartMovieLensModel
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm

random_state = np.random.randint(0, 24532523)
#model = ToyLinearModel(n_features=8,n_actions=20,random_state=random_state,noise=0.1)
model = ColdStartMovieLensModel(random_state=random_state,noise=0.1)

T = 2000
print('best suggestion: '+str(np.argmax(np.dot(model.features, model.real_theta))))

#alg_names=['ColdStart', 'random_action','eps_greedy_action']
alg_names=['ColdStart']
#parameters=[6.,None,0.05]
parameters=[6.]
p=dict(zip(alg_names,parameters))

N=50

for alg_name in alg_names:
    results=model.MC_simul(T,N,alg_name,0.0001,param=p[alg_name])
    plt.figure(1)
    plt.subplot(121)
    plt.plot(results[1], label=alg_name)
    plt.ylabel('d(theta, theta_hat)')
    plt.xlabel('Rounds')
    plt.legend()

    plt.subplot(122)
    plt.plot(results[0], label=alg_name)
    plt.ylabel('Cumulative Regret')
    plt.xlabel('Rounds')
    plt.legend()
plt.show()


########### gridsearch 1 to optimize alpha and lambda: large T and few values ##################
# Idea: show asymptotic properties to calibrate T for further exploration
#       Find a target zone for alpha and lambda for arbitrage good theta/good regret

alpha_list=[1.,5.,10.,50.]
l_list=[0.001,0.005,0.01,0.05,0.1]
N=50
path1=r'C:\Users\dobau\Desktop\MVA\Reinforcement Learning\TP2\TP2_python\results\GridSearch1'

'''
i,j=0,0
for alpha in tqdm(alpha_list,desc='alpha={}'.format(str(i))):
    for l in tqdm(l_list,desc='l={}'.format(str(j))):
        results=model.MC_simul(T,N,'ColdStart',l,param=alpha)
        name='results_l_'+str(l*1000)+'_alpha_'+str(alpha*10)+'.pkl'
        pickle.dump(results, open(os.path.join(path1,name), 'wb'))
        j+=1
    i+=1

available_results=os.listdir(path1)
theta_dist=np.zeros((T,len(available_results)))
regret=np.zeros((T,len(available_results)))

for i,name in enumerate(available_results):
    print(i,name)
    res=pickle.load(open(os.path.join(path1,name), 'rb'))
    regret[:,i]=res[0]
    theta_dist[:,i]=res[1]

print(np.argsort(regret[-1]))
print(np.argsort(theta_dist[-1]))

import pandas as pd
pd.DataFrame(regret[:,[9,17,5,15]]).plot()
plt.show()
pd.DataFrame(theta_dist[:,[9,17,5,15]]).plot()
plt.show()
'''

###################### Gridsearch2: more precise calibration of alpha/lambda ####################################

'''
alpha_list2=np.linspace(3.,7.,9)
l_list2=np.linspace(0.0001,0.01,10)
print(alpha_list2)
print(l_list2)
N2=50
T2=500
'''
path2=r'C:\Users\dobau\Desktop\MVA\Reinforcement Learning\TP2\TP2_python\results\GridSearch2'
'''
i,j=0,0
for alpha in tqdm(alpha_list2,desc='alpha={}'.format(str(i))):
    for l in tqdm(l_list2,desc='l={}'.format(str(j))):
        results=model.MC_simul(T2,N2,'ColdStart',l,param=alpha)
        name='results_l_'+str(l)+'_alpha_'+str(alpha)+'.pkl'
        pickle.dump(results, open(os.path.join(path2,name), 'wb'))
        j+=1
    i+=1

available_results2=os.listdir(path2)
theta_dist2=np.zeros((T2,len(available_results2)))
regret2=np.zeros((T2,len(available_results2)))

for i,name in enumerate(available_results2):
    print(i,name)
    res=pickle.load(open(os.path.join(path2,name), 'rb'))
    regret2[:,i]=res[0]
    theta_dist2[:,i]=res[1]

print(np.argsort(regret2[-1]))
print(np.argsort(theta_dist2[-1]))

print(regret2[-1])
print(theta_dist2[-1])

print(np.min(regret2[-1]),np.max(regret2[-1]))
print(np.min(theta_dist2[-1]),np.max(theta_dist2[-1]))

print(T2*model.best_arm_reward())

import pandas as pd
pd.DataFrame(regret2[:,:]).plot()
plt.show()
pd.DataFrame(theta_dist2[:,:]).plot()
plt.show()
'''