import numpy as np
from tqdm import tqdm

class LinearMABModel(object):
    def __init__(self, random_state=0, noise=0.):
        self.local_random = np.random.RandomState(random_state)
        self.noise = noise

    def reward(self, action):
        assert 0<= action < self.n_actions, "{} not in 0 .. {}".format(action, self.n_actions)
        reward = np.inner(self.features[action], self.real_theta) + self.noise * self.local_random.randn(1)
        return reward

    def best_arm_reward(self):
        D = np.dot(self.features, self.real_theta)
        return np.max(D)

    @property
    def n_features(self):
        return self.features.shape[1]

    @property
    def n_actions(self):
        return self.features.shape[0]

    def MC_simul(self,T,N,method,L,param=None):
        mean_regret=np.zeros(T)
        mean_theta_dist=np.zeros(T)
        for i in tqdm(range(N), desc="Simulating {}".format(method)):
            results=self.run(T,method,L,param)
            mean_regret+=results[1]/N
            mean_theta_dist+=self.theta_dist(results[3])/N
        return mean_regret, mean_theta_dist

    def run(self,T,method,L,param=None):
        actions, reward, regret,all_theta=self.init_random(T,L) #initialize all lists and self.A, self.Z when CS
        for t in range(1,T):
            action=self.__getattribute__(method)(param)
            actions[t]=action
            reward[t]=self.reward(action)
            regret[t]=regret[t-1]+self.best_reward-reward[t]
            all_theta[t]=self.theta
            self.update(action,t,reward,L)
        return actions, regret, reward,all_theta

    def init_random(self,T,L): #Random first draw, initialization of the run
        actions=np.zeros(T,dtype=int)
        reward=np.zeros(T)
        regret=np.zeros(T)
        all_theta=np.zeros((T,self.n_features))
        actions[0]=self.local_random.randint(0,self.n_actions-1)
        reward[0]=self.reward(actions[0])
        self.best_reward=self.best_arm_reward()
        regret[0]=self.best_reward-reward[0]
        self.A=np.outer(self.features[actions[0]],self.features[actions[0]]) #Z^T*Z
        self.Zy=reward[0]*self.features[actions[0]] #Z^T*y
        self.Z_lamb=np.linalg.inv(self.A+L*np.eye(self.n_features))
        self.theta=np.dot(self.Z_lamb,self.Zy)
        all_theta[0]=self.theta
        return actions,reward,regret,all_theta

    def update(self,action,t,reward,L):
        self.A+=np.outer(self.features[action], self.features[action])
        self.Zy+=reward[t]*self.features[action]
        self.Z_lamb=np.linalg.inv(self.A+L*np.eye(self.n_features))
        self.theta=np.dot(self.Z_lamb, self.Zy)

    def ColdStart(self,param):
        beta=np.zeros(self.n_actions)
        exp_reward=np.zeros(self.n_actions)
        alpha=param
        for a in range(self.n_actions):
            phi_a=self.features[a]
            beta[a]=alpha*np.sqrt(np.dot(phi_a,np.dot(self.Z_lamb,phi_a)))
            exp_reward[a]=np.inner(phi_a,self.theta)+beta[a]
        return np.argmax(exp_reward)

    def random_action(self,param):
        return self.local_random.randint(0,self.n_actions)

    def eps_greedy_action(self,param):
        epsilon=param
        p=self.local_random.rand(1)
        if p<=epsilon:
            return self.local_random.randint(0,self.n_actions)
        else:
            return self.ColdStart(param=0)

    def theta_dist(self,theta):
        return np.apply_along_axis(lambda x:np.sum(x-self.real_theta)**2,1,theta)

class ToyLinearModel(LinearMABModel):
    def __init__(self, n_features, n_actions, random_state=0, noise=0.):
        super(ToyLinearModel, self).__init__(random_state=random_state, noise=noise)
        self.features = self.local_random.rand(n_actions, n_features) - 0.5
        self.real_theta = self.local_random.rand(n_features) - 0.5

class ColdStartMovieLensModel(LinearMABModel):
    def __init__(self, random_state=0, noise=0.):
        self.features = np.loadtxt('movielens/Vt.csv', delimiter=',').T
        super(ColdStartMovieLensModel, self).__init__(random_state=random_state, noise=noise)
        self.real_theta = self.local_random.randn(self.n_features)
