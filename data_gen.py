import numpy as np
import keras
from pysc2.lib import features, point
from absl import app, flags
from pysc2.env.environment import TimeStep, StepType
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb
import importlib
import glob
# from transform_replay import *
import math
import os
import random
import genc

class Mygenerator(keras.utils.Sequence):
    def __init__(self, batch_size, obs, replay_path):
        self.batch_size = batch_size
        self.replays_path = replay_path
        self.all_replays = os.listdir(replay_path)
        self.obs = obs

    # def __len__(self):
    #     ls = os.listdir(self.replays_path)  # dir is your directory path
    #     number_files = len(ls)
    #     return int(np.floor(number_files / self.batch_size))
    def __len__(self):
        ls = os.listdir(self.replays_path)  # dir is your directory path
        number_files = int(math.floor(len(ls) / self.batch_size))

        # number_files //= 2
        return number_files
    def __getitem__(self, idx):
        # print("Obtaining data...")
        # x, y = get64obs(self.all_replays[idx])
        # data = np.load("{0}/testmdata{1}.npz".format(self.replays_path, str(idx)))
        indexes = [self.replays_path+replay for replay in self.all_replays[idx * self.batch_size:(idx + 1) * self.batch_size]]
        return genc.gen(indexes, self.obs)
        # retX1 = []
        # retX2 = []
        # retX3 = []
        # retY = []
        # indexes = self.all_replays[idx * self.batch_size:(idx + 1) * self.batch_size]
        # for replay in indexes:
        #     data = np.load(self.replays_path + replay)
        #     # print(self.replays_path+ replay)
        #     x = [data['name1'], data['name2'], data['name3']]
        #     # if data['name1'].shape[0] !=64 or data['name2'].shape[0] !=64 or data['name3'].shape[0] !=64:
        #     #     print(self.all_replays[idx])
        #     #     print("error")
        #     y = data['name4']
        #     wanted_size = -self.obs
        #     retX1.append(x[0][wanted_size:])
        #     retX2.append(x[1][wanted_size:])
        #     X3 = [z.flatten() for z in x[2][wanted_size:]]
        #     # X3 = np.reshape(X3, (wanted_size, 522))
        #     retX3.append(X3)
        #     retY.append(y[wanted_size:][0])
        # return [np.asarray(retX1), np.asarray(retX2), np.asarray(retX3)], retY

    # def __getitem__(self, idx):
    #     # print("Obtaining data...")
    #     # x, y = get64obs(self.all_replays[idx])
    #     # data = np.load("{0}/testmdata{1}.npz".format(self.replays_path, str(idx)))
    #     indexes = self.all_replays[idx * self.batch_size:(idx + 1) * self.batch_size]
    #     retX1 = []
    #     retX2 = []
    #     retX3 = []
    #     retY = []
    #     for replay in indexes:
    #         data = np.load(self.replays_path + replay)
    #         # print(self.replays_path+ replay)
    #         x = [data['name1'], data['name2'], data['name3']]
    #         # if data['name1'].shape[0] !=64 or data['name2'].shape[0] !=64 or data['name3'].shape[0] !=64:
    #         #     print(self.all_replays[idx])
    #         #     print("error")
    #         y = data['name4']
    #         wanted_size = self.obs
    #         retX1.append(x[0][:wanted_size])
    #         retX2.append(x[1][:wanted_size])
    #         X3 = [z.flatten() for z in x[2][:wanted_size]]
    #         # X3 = np.reshape(X3, (wanted_size, 522))
    #         retX3.append(X3)
    #         retY.append(y[:wanted_size][0])
    #     # np.savez("testmdata{0}.npz".format(str(idx)), name1=x[0], name2=x[1], name3=[2], name4=y)
    #     # X, Y = np.array(retX), np.array(retY)
    #     # print(X.shape, Y.shape)
    #     # print("Data obtained.")
    #     return [np.asarray(retX1), np.asarray(retX2), np.asarray(retX3)], retY
    #     # return x[2][wanted_size:], y[wanted_size:]
    #     # X1 = np.ndarray(shape=(64,64,7), dtype=np.int32)
    #     # X1 = np.ndarray(shape=(64, 64, 17), dtype=np.int32)
    #     # X2 = np.ndarray(shape=(64, 64, 7), dtype=np.int32)
    #     # X3 = np.ndarray(shape=(64, 1, 522), dtype=np.int32)
    #
    #     # def printlst(self):
    #     #     for i in range(5):
    #     #         print(self.all_replays[i])

class My_cnn_generator(keras.utils.Sequence):
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
        number_files = int(math.floor(len(ls) / self.batch_size))
        # number_files //= 2
        return number_files

    def __getitem__(self, idx):
        # x, y = get64obs(self.all_replays[idx])
        # data = np.load("{0}/testmdata{1}.npz".format(self.replays_path, str(idx)))
        mini_indexes = random.randint(0, self.batch_size)
        data = np.load(self.replays_path + self.all_replays[idx])
        print(self.all_replays[idx])
        x = [data['name1'], data['name2'], data['name3']]
        y = data['name4']
        wanted_size = -self.batch_size

        # np.savez("testmdata{0}.npz".format(str(idx)), name1=x[0], name2=x[1], name3=[2], name4=y)
        # if y[wanted_size:][0] == 0:
        #     return [x[0][wanted_size:], x[1][wanted_size:], x[2][wanted_size:]], np.negative(np.ones(len(y[wanted_size:])))
        # return len(y[wanted_size:])
        return [x[0][wanted_size:], x[1][wanted_size:], x[2][wanted_size:]], y[wanted_size:]