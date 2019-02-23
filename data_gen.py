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

        
    # def __len__(self):
    #     ls = os.listdir(self.replays_path)  # dir is your directory path
    #     number_files = len(ls)
    #     return int(np.floor(number_files / self.batch_size))
    def __len__(self):
        ls = os.listdir(self.replays_path)  # dir is your directory path
        number_files = len(ls)
        return number_files


    def __getitem__(self, idx):
        # x, y = get64obs(self.all_replays[idx])
        # data = np.load("{0}/testmdata{1}.npz".format(self.replays_path, str(idx)))
        data = np.load('./replay_data/'+ self.all_replays[idx])

        x = [data['name1'], data['name2'], data['name3']]
        if data['name1'].shape[0] !=64 or data['name2'].shape[0] !=64 or data['name3'].shape[0] !=64:
            print(self.all_replays[idx])
            print("error")
        y = data['name4']
        # np.savez("testmdata{0}.npz".format(str(idx)), name1=x[0], name2=x[1], name3=[2], name4=y)
        return [x[0][32:], x[1][32:], x[2][32:]], y[32:]

    # def printlst(self):
    #     for i in range(5):
    #         print(self.all_replays[i])

