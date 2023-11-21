import numpy as np 
import gymnasium as gym
import random
import matplotlib.pyplot as plt
import math

class ContinuousQ():
    def __init__(self, num_buckets, env = gym.make('CartPole-v1', render_mode="human")):
        self.env = env

        self.epsilon = 1
        self.discount = 0.9
        self.alpha = 0.1
        


        DISCRETE_OS_SIZE = [num_buckets]*4
        self.high_space = np.array([env.observation_space.high[0], 5, env.observation_space.high[2], 5])
        self.low_space = np.array([env.observation_space.low[0], -5, env.observation_space.low[2], -5])

        self.bucket_size = (self.high_space - self.low_space)/DISCRETE_OS_SIZE

        self.q_table = np.random.uniform(low=0, high=1, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
       

    def policy(self, current_state):
        random_numb = random.uniform(0,1)
        directions = self.q_table[current_state]

        if random_numb > self.epsilon:
            action = np.argmax(directions)
        else:
            action = random.choice([0,1])
        return action
            
    def state_to_tuple(self, array_state):
        indexed_state = (array_state - self.low_space)/self.bucket_size

        indexed_state = tuple(round(i) for i in indexed_state)
        #print(indexed_state)
        return indexed_state

    def qUpdate(self, state, action, reward, next_state):
        maxQ = max(self.q_table[next_state])
        print(reward)
        self.q_table[state][action] = self.q_table[state][action]*(1-self.alpha) + self.alpha * (reward + self.discount*maxQ)

    def episodeQ(self,max_steps):
        terminate = False
        step = 0
        bonus = 0
        self.env.reset()
        observation, info = self.env.reset()
        state = self.state_to_tuple(observation)
        while not terminate and step < max_steps:

            #Get a great action from policy
            action = self.policy(state)

            #Get next state and reward
            next_state, reward, terminate, _, _ = self.env.step(action)
            #Draw State
            #self.env.render()
            next_state = self.state_to_tuple(next_state)
            #Update q value
            if terminate:
                continue
            self.qUpdate(state, action, reward, next_state)
            if reward==1:
                bonus+=1
            else:
                bonus-=0
        
            state = next_state
            
            self.epsilon=self.epsilon*self.epsilon
            step += 1
        return bonus


    def trainQ(self,episodes):
        reward_over_time = []
        for _ in range(episodes):
            bonus = self.episodeQ(1000)
            reward_over_time.append(bonus)
        self.env.close()
        return reward_over_time

q = ContinuousQ(100,)
x = list(range(0,1000))
results_Q = q.trainQ(1000)


plt.plot(x, results_Q, color='r',label='Q Learning',linewidth=0.7)

plt.title("Q Learning Rewards")
plt.xlabel("Episode #") 
plt.ylabel("Reward Per Episode") 
plt.legend()
plt.show()