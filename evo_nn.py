import numpy as np 
import gymnasium as gym
import random
import matplotlib.pyplot as plt
import math
import nn
from statistics import mean 

architecture = [
    {"input_dim": 4, "output_dim": 8, "activation": "relu"},
   # {"input_dim": 4, "output_dim": 6, "activation": "relu"},
   # {"input_dim": 6, "output_dim": 6, "activation": "relu"},
   # {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 8, "output_dim": 2, "activation": "relu"},
]




def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class EvolutionaryNN():
    def __init__(self, pop_size, env = gym.make('CartPole-v1', render_mode='human')):
        self.env = env
        self.population = [nn.init_layers(architecture) for i in range(pop_size)]
        self.fitness = [0.0]*pop_size
        self.unwrapped = []

    def init_values(self):
        for i in range(len(self.population)):
            #Evaluate fitness of solution
            self.fitness[i] = self.evaluate_fitness(self.population[i])
            #Unwrap weights and biases for crossover
            self.unwrapped.append(self.unwrap(self.population[i]))


    def unwrap(self, params_values):
        print(params_values)
        wrapping = np.array([])
        for key in params_values.keys():
            if 'w' in key or 'b' in key:
                item = params_values[key]
                print(key)
                print(item.shape)
                
                unwrapped_value = np.ndarray.flatten(item)
                np.concatenate((wrapping, unwrapped_value)) 
        print("Done unwrap")
        return wrapping


    "Combine parents to create child"
    def crossover(self, mother_index, father_index):
        mother_gene = self.unwrapped[mother_index]
        father_gene = self.unwrapped[father_index]

        gene_length = len(mother_gene)
        cut_index = random.randint(0,gene_length-1)
        child = mother_gene[:cut_index] + father_gene[cut_index:]
        assert len(child) == len(father_gene)
        return child


    def rewrap(self, unwrapped_child):
        unpacked_params = []                                    
        e = 0
        params_value = {}
        params_value['w1'] = unwrapped_child[:32].reshape(8,4)
        params_value['b1'] = unwrapped_child[32:40].reshape(8,1)
        params_value['w2'] = unwrapped_child[40:56].reshape(2,8)
        params_value['b2'] = unwrapped_child[56:58].reshape(2,1)

        
        #    bias = params[s:e]
        #    unpacked_params.extend([weights,bias])              
        #return unpacked_params



# architecture = [
#         {"input_dim": 4, "output_dim": 32, "activation": "relu"},
#     # {"input_dim": 4, "output_dim": 6, "activation": "relu"},
#     # {"input_dim": 6, "output_dim": 6, "activation": "relu"},
#     # {"input_dim": 6, "output_dim": 4, "activation": "relu"},
#         {"input_dim": 32, "output_dim": 2, "activation": "sigmoid"},
#     ]





    def chooseAction(self, state, params_value):
        state = np.array([state[0], state[1], state[2], state[3]])
        probabilities, params_value = nn.forward_step(state,params_value,architecture)
        probabilities = np.transpose(probabilities)
        return np.argmax(softmax(probabilities[0]))

    def evaluate_fitness(self, agent):
        rewards = self.runSprint(1, agent)
        return mean(rewards)



    def episode(self,max_steps, agent):
        terminate = False
        step = 0
        episode_rewards = 0
        self.env.reset()
        observation, info = self.env.reset()
        state = observation
        while not terminate and step < max_steps:
            
            #Get a great action from policy
            action = self.chooseAction(state, agent)
            #Get next state and reward
            next_state, reward, terminate, _, _ = self.env.step(action)
            #Draw State
            self.env.render()
            if terminate:
                continue
            if reward==1:
                episode_rewards+=1
            else:
                episode_rewards-=0
        
            state = next_state
            
            #self.epsilon=self.epsilon*self.epsilon
            step += 1
        return episode_rewards


    def runSprint(self,episodes, agent):
        reward_over_time = []
        for _ in range(episodes):
            bonus = self.episode(100, agent)
            reward_over_time.append(bonus)
        self.env.close()
        return reward_over_time

x = EvolutionaryNN(5)
x.init_values()