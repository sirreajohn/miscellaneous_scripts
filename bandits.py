 # -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:02:44 2020

@author: mahesh
"""
import math
import random
import matplotlib.pyplot as plt
import sqlite3 as sq


class bandits:
    def __init__(self, ads=12, epsilon=0.2):
        self.ads = ads
        self.ads_list = [*range(ads)]
        self.ads_selected = []
        self.number_of_selections = [0]*ads
        self.sum_of_rewards = [0]*ads
        self.bound_list = [1e400]*ads
        self.total_rewards = 0
        self.run = True
        self.iteration = 0
        self.selection_history = []
        self.epsilon = epsilon

    def display_ads(self, ads):
        print(ads)

    def update_bound(self, ad, iteration):
        average_reward = self.sum_of_rewards[ad]/self.number_of_selections[ad]
        delta_i = math.sqrt(3/2*math.log(iteration+1) /
                            self.number_of_selections[ad])
        self.bound_list[ad] = average_reward + delta_i

    def max_bound_index(self):
        return sorted(range(len(self.bound_list)), key=lambda sub: self.bound_list[sub])[-3:]

    def update_UCB(self, chosen_ad, iteration):
        self.number_of_selections[chosen_ad] = self.number_of_selections[chosen_ad] + 1
        self.selection_history.append(chosen_ad)
        self.sum_of_rewards[chosen_ad] = self.sum_of_rewards[chosen_ad] + 1
        self.total_rewards = self.total_rewards + 1
        self.update_bound(chosen_ad, iteration)

    def run_UCB(self, itera=10):
        """ do not use this function, it only works on script, use self.iter_ads() instead"""
        while (self.run and self.iteration <= itera):
            if self.iteration < 4:
                if self.iteration < 3:
                    if self.iteration < 2:
                        if self.iteration == 0:
                            self.display_ads([1, 2, 3])
                            ad = int(input("choose one from above phase 1: "))
                            self.update_UCB(ad, self.iteration)
                            self.iteration += 1
                        else:
                            self.display_ads([4, 5, 6])
                            ad = int(input("choose one from above phase 2: "))
                            self.update_UCB(ad, self.iteration)
                            self.iteration += 1
                    else:
                        self.display_ads([7, 8, 9])
                        ad = int(input("choose one from above phase 3: "))
                        self.update_UCB(ad, self.iteration)
                        self.iteration += 1
                else:
                    self.display_ads([10, 11, 12])
                    ad = int(input("choose one from above phase 4: "))
                    self.update_UCB(ad, self.iteration)
                    self.iteration += 1
                    print('random_phase complete, initiating UCB')
            else:
                if random.randint(0, 1) > self.epsilon:
                    self.display_ads(self.max_bound_index())
                    chosen_ad = int(input('choose one of the above: '))
                    self.update_UCB(chosen_ad, self.iteration)
                    self.iteration += 1
                else:
                    rand_ads = random.choices(self.ads_list, k=3)
                    self.display_ads(rand_ads)
                    chosen_ad = int(input('choose one of the above(random): '))
                    self.update_UCB(chosen_ad, self.iteration)
                    self.iteration += 1

    def plot_stats(self):
        plt.scatter([*range(self.iteration)], self.selection_history)
        plt.plot([*range(self.iteration)], self.selection_history)
        plt.title("selection history")
        plt.xlabel("iterations")
        plt.ylabel("selected ads")

    def print_vars(self):
        print(self.selection_history)
        print([*range(self.iteration)])
        return self.bound_list

    def iter_ads(self, iteration):
        """ since we give user choice we have to call self.update_bound() to update the bias for chosen ads"""
        if iteration == 0:
            return [1, 2, 3]
        elif iteration == 1:
            return [4, 5, 6]
        elif iteration == 2:
            return [7, 8, 9]
        elif iteration == 3:
            return [10, 11, 12]
        else:
            if random.randint(0, 1) > self.epsilon:
                return self.max_bound_index()
            else:
                return random.choices(self.ads_list, k=3)

    def write_iter(self):
        """disregard this for now"""
