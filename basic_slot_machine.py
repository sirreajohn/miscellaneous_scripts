# -*- coding: utf-8 -*-
"""
Spyder Editor

author : mahesh 
"""

import numpy as np
import matplotlib.pyplot as plt

class agent():
    def __init__(self):
        self.sum_rewards = 0
        self.sum_rewards_list = []
        self.iters = 0
        self.prob_list = []
    def call(self, threshold = 0.5):
        if self.iters == 0:
            self.prob_list.append(np.random.uniform(0,1))
            reward = slot_machine()
            self.sum_rewards += reward
            self.sum_rewards_list.append(reward)
            self.iters += 1
        else:
            prob = self.sum_rewards/self.iters
            self.prob_list.append(prob)
            if prob > threshold:
                reward = slot_machine()
                self.sum_rewards += reward
                self.sum_rewards_list.append(reward)
                self.iters += 1
            else:
                self.iters += 1
                self.sum_rewards_list.append(0)
    def print_vars(self):
        print('sum_of_rewards received',self.sum_rewards)
        print("in iterations ",self.iters)
        print("with probablity with each iter i : \n", self.prob_list)
    def graph(self,graph_no = 1):
        if graph_no == 1:
            iter_list = [*range(self.iters)]
            #plt.xticks(np.arange(min(iter_list), max(iter_list)+1,2)) use this for smallers episodes
            plt.plot(iter_list,self.sum_rewards_list)
            plt.xlabel("iterations")
            plt.ylabel("rewards")
            plt.title("classic slot machine problem")
            plt.show()
        elif graph_no == 2:
            iter_list = [*range(self.iters)]
            #plt.xticks(np.arange(min(iter_list), max(iter_list)+1,2)) # kinda messes up xlabel on bigger episodes
            plt.plot(iter_list,self.prob_list)
            plt.xlabel("iterations")
            plt.ylabel("probability of rewards")
            plt.title("classic slot machine problem")
            plt.show()
        else:
            raise ValueError("param range error, graph_no given :", graph_no)
            
def slot_machine():
    reward_range = [1,2,8,6,4,5,6,2,0,3,4,8,9,5,4,2]
    return np.random.choice(reward_range)



agent1 = agent()
episodes = 50
for i in range(episodes):
    agent1.call()
# agent1.print_vars()
agent1.graph(graph_no =2)
agent1.graph(graph_no =1)
    
