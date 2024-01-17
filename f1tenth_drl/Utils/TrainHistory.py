import os, shutil
import csv
import numpy as np
from matplotlib import pyplot as plt
from f1tenth_drl.Utils.utils import *
from matplotlib.ticker import MultipleLocator


SIZE = 20000


def plot_data(values, moving_avg_period=10, title="Results", figure_n=2):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    if len(values) >= moving_avg_period:
        moving_avg = true_moving_average(values, moving_avg_period)
        plt.plot(moving_avg)    
    if len(values) >= moving_avg_period*5:
        moving_avg = true_moving_average(values, moving_avg_period * 5)
        plt.plot(moving_avg)    
    # plt.pause(0.001)


class TrainHistory():
    def __init__(self, path) -> None:
        self.path = path
        # self.path = conf.vehicle_path + run.path +  run.run_name 

        # training data
        self.ptr = 0
        self.lengths = np.zeros(SIZE)
        self.rewards = np.zeros(SIZE) 
        self.progresses = np.zeros(SIZE) 
        self.laptimes = np.zeros(SIZE) 
        self.t_counter = 0 # total steps
        
        # espisode data
        self.ep_counter = 0 # ep steps
        self.ep_reward = 0
        self.reward_list = []

    def add_step_data(self, new_r):
        self.ep_reward += new_r
        self.reward_list.append(new_r)
        self.ep_counter += 1
        self.t_counter += 1 

    def lap_done(self, reward, progress, show_reward=False):
        self.add_step_data(reward)
        self.lengths[self.ptr] = self.ep_counter
        self.rewards[self.ptr] = self.ep_reward
        self.progresses[self.ptr] = progress
        self.ptr += 1
        self.reward_list = []

        # if show_reward:
        #     plt.figure(8)
        #     plt.clf()
        #     plt.plot(self.ep_rewards)
        #     plt.plot(self.ep_rewards, 'x', markersize=10)
        #     plt.title(f"Ep rewards: total: {self.ep_reward:.4f}")
        #     plt.ylim([-1.1, 1.5])
        #     plt.pause(0.0001)

        self.ep_counter = 0
        self.ep_reward = 0
        self.ep_rewards = []

    def print_update(self, plot_reward=False):
        if self.ptr < 10:
            return
        
        mean10 = np.mean(self.rewards[self.ptr-10:self.ptr])
        mean100 = np.mean(self.rewards[max(0, self.ptr-100):self.ptr])
        # print(f"Run: {self.t_counter} --> Moving10: {mean10:.2f} --> Moving100: {mean100:.2f}  ")
        
        # if plot_reward:
        #     # raise NotImplementedError
        #     plot_data(self.rewards[0:self.ptr], figure_n=2)

    def save_csv_data(self):
        data = []
        ptr = self.ptr  #exclude the last entry
        for i in range(ptr): 
            data.append([i, self.rewards[i], self.lengths[i], self.progresses[i], self.laptimes[i]])
        save_csv_array(data, self.path + "/training_data_episodes.csv")

        t_steps = np.cumsum(self.lengths[0:ptr])/100
        plt.figure(3)
        
        plt.clf()
        plt.plot(t_steps, self.progresses[0:ptr], '.', color='darkblue', markersize=4)
        plt.plot(t_steps, true_moving_average(self.progresses[0:ptr], 20), linewidth='4', color='r')
        plt.ylabel("Average Progress")
        plt.xlabel("Training Steps (x100)")
        plt.ylim([0, 100])
        plt.grid(True)
        plt.savefig(self.path + "/training_progresses_steps.png")

        plt.clf()
        plt.plot(t_steps, self.rewards[0:ptr], '.', color='darkblue', markersize=4)
        plt.plot(t_steps, true_moving_average(self.rewards[0:ptr], 20), linewidth='4', color='r')
        plt.xlabel("Training Steps (x100)")
        plt.ylabel("Reward per Episode")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(self.path + "/training_rewards_steps.png")
        
        plt.close(3)

        # plt.figure(4)
        # plt.clf()
        # plt.plot(t_steps, self.progresses[0:self.ptr], '.', color='darkblue', markersize=4)
        # plt.plot(t_steps, true_moving_average(self.progresses[0:self.ptr], 20), linewidth='4', color='r')

        # plt.xlabel("Training Steps (x100)")
        # plt.ylabel("Progress")

        # plt.tight_layout()
        # plt.grid()
        # plt.savefig(self.path + "/training_progress_steps.png")

        # plt.close()





