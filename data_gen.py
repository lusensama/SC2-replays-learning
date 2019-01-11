import numpy as np
import keras
from pysc2.lib import features, point
from absl import app, flags
from pysc2.env.environment import TimeStep, StepType
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb
import importlib
import glob
from transform_replay import *
import os

class Mygenerator(keras.utils.Sequence):
    def __init__(self, batch_size, replay_path):
        self.batch_size = batch_size
        self.replays_path = replay_path
        self.all_replays = os.listdir(replay_path)

        
    def __len__(self):
        ls = os.listdir(self.replays_path)  # dir is your directory path
        number_files = len(ls)
        return int(np.floor(number_files / self.batch_size))

    def __getitem__(self, idx):
        # batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        # batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # read your data here using the batch lists, batch_x and batch_y
        # self.printlst()
        x, y = get64obs(self.all_replays[idx])

        return x, y

    # def printlst(self):
    #     for i in range(5):
    #         print(self.all_replays[i])
