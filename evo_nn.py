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
    def __init__(self, pop_size):
        
        self.population = [nn.init_layers(architecture) for i in range(pop_size)]
        self.fitness = [0.0]*pop_size
        self.unwrapped = []
        self.epsilon = 0.9

    def init_values(self):
        for i in range(len(self.population)):
            #Evaluate fitness of solution
            self.fitness[i] = self.evaluate_fitness(self.population[i])
            #Unwrap weights and biases for crossover
            self.unwrapped.append(self.unwrap(self.population[i]))


    def unwrap(self, params_values):
        wrapping = np.array([])
        for key in params_values.keys():
            if 'w' in key or 'b' in key:
                item = params_values[key]        
                unwrapped_value = np.ndarray.flatten(item)
                wrapping = np.concatenate((wrapping, unwrapped_value)) 
        return wrapping


    "Combine parents to create child"
    def crossover(self, mother_index, father_index):
        mother_gene = self.unwrapped[mother_index]
        father_gene = self.unwrapped[father_index]

        gene_length = len(mother_gene)

        cut_index = random.randint(0,gene_length-2)

        child = np.concatenate([mother_gene[:cut_index],father_gene[cut_index:]])
        assert len(child) == len(father_gene)
        return child


    def rewrap(self, unwrapped_child):
        params_value = {}
        params_value['w1'] = unwrapped_child[:32].reshape(8,4)
        params_value['b1'] = unwrapped_child[32:40].reshape(8,1)
        params_value['w2'] = unwrapped_child[40:56].reshape(2,8)
        params_value['b2'] = unwrapped_child[56:58].reshape(2,1)
        return params_value


    def mutate(self, child_gene, num_mutations):
        for i in range(num_mutations):
            random_pos = random.randint(0,len(child_gene)-1)
            add = random.randint(0,1)
            difference = random.uniform(0,0.5)
            if add:
                child_gene[random_pos]+=difference
            else:
                child_gene[random_pos]-=difference
        return child_gene
    

    def new_population(self):
        probability = random.random()
        best_index = np.argmax(self.fitness) if probability > self.epsilon else random.randint(0,len(self.fitness)-1)
        second_best = best_index = np.argmax(self.fitness) if probability > self.epsilon else random.randint(0,len(self.fitness)-1)
        while second_best == best_index:
            second_best = random.randint(0,len(self.fitness)-1)
        child_gene = self.crossover(best_index, second_best)
        child_gene = self.mutate(child_gene, 10)
        child_params = self.rewrap(child_gene)
        child_fitness = self.evaluate_fitness(child_params)
        if min(self.fitness) < child_fitness:
            i = np.argmin(self.fitness)
            self.fitness[i] = child_fitness
            self.population[i] = child_params
            self.unwrapped[i] = child_gene
        
        


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
        observation, info = self.env.reset()
        state = observation
        while not terminate and step < max_steps:
            #Get a great action from policy
            action = self.chooseAction(state, agent)
            #Get next state and reward
            next_state, reward, terminate, _, _ = self.env.step(action)
            #Draw State
            #self.env.render()
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


    def iterateGenerations(self, num):
        rewards_over_time = []
        self.env = gym.make('CartPole-v1')
        self.init_values()

        print("Average Fitness:", mean(self.fitness))
        for i in range(num):
            print(f"Generation {i}")
            self.new_population()
            print("Average Fitness:", mean(self.fitness))
            print("Best Fitness:", self.fitness[np.argmax(self.fitness)])
            rewards_over_time.append(mean(self.fitness))
            #rewards_over_time.append(self.fitness[np.argmax(self.fitness)])
            #if mean(self.fitness)==100:
            #    break
        return rewards_over_time

x = EvolutionaryNN(20)
results_Q = x.iterateGenerations(100)
t = list(range(0,100))
plt.plot(t, results_Q, color='b')

plt.title("Evolutionary Neural Network")
plt.xlabel("Generation #") 
plt.ylabel("Average Reward Per Generation") 
plt.legend()
plt.show()