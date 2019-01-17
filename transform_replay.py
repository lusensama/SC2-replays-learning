#!/usr/bin/env python

from pysc2.lib import features, point
from absl import app, flags
from pysc2.env.environment import TimeStep, StepType
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb
import importlib
import cv2
import random
import sc2reader
import glob
import time
import numpy as np
import os

FLAGS = flags.FLAGS
flags.DEFINE_string("replay_path", 'D:/University_Work/My_research/fixed_replays/Replays', "Path to a replay files.")
flags.DEFINE_string("agent", None, "Path to an agent.")
# flags.mark_flag_as_required("replay_path")
# flags.mark_flag_as_required("agent")
PATH_REPLAY = 'D:/University_Work/My_research/fixed_replays/Replays'


def get_random_steps(length, sample_size):
    rlist = random.sample(range(length), sample_size)
    rlist.sort()
    returnlist = []
    returnlist.append(rlist[0])
    for i in range(1, len(rlist)):
        returnlist.append(rlist[i] - rlist[i - 1])
    return returnlist

def extract_features(agent_obs):

    screen = agent_obs['feature_screen']
    re_screen = np.reshape(screen, (64, 64, 17))

    mini = agent_obs['feature_minimap']
    re_mini = np.reshape(mini, (64, 64, 7))

    temp = np.zeros(541, dtype='uint')
    for idx in agent_obs.available_actions:
        temp[idx] = 1
    info = np.concatenate((temp, agent_obs['player']), axis=None)

    return re_screen, re_mini, info
class ReplayEnv:
    def __init__(self,
                 replay_file_path,
                 agent,
                 player_id=2,
                 screen_size_px=(64, 64),
                 minimap_size_px=(64, 64),
                 discount=1.,
                 step_mul=100,
                 version = '3.16.1'):

        self.agent = agent
        self.discount = discount
        self.step_mul = step_mul

        self.run_config = run_configs.get()
        self.sc2_proc = self.run_config.start(version)
        self.controller = self.sc2_proc.controller
        self.current_screen = None
        self.current_minimap = None

        replay_data = self.run_config.replay_data(replay_file_path)
        ping = self.controller.ping()
        self.info = self.controller.replay_info(replay_data)
        if not self._valid_replay(self.info, ping):
            raise Exception("{} is not a valid replay file!".format(replay_file_path))

        screen_size_px = point.Point(*screen_size_px)
        minimap_size_px = point.Point(*minimap_size_px)
        interface = sc_pb.InterfaceOptions(
            raw=False, score=True,
            feature_layer=sc_pb.SpatialCameraSetup(width=24))
        screen_size_px.assign_to(interface.feature_layer.resolution)
        minimap_size_px.assign_to(interface.feature_layer.minimap_resolution)

        map_data = None
        if self.info.local_map_path:
            map_data = self.run_config.map_data(self.info.local_map_path)

        self._episode_length = self.info.game_duration_loops
        if self._episode_length < 2000:
            raise Exception("Game too short for analysis.")
        self._episode_steps = 0

        self.controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id))

        self._state = StepType.FIRST

    @staticmethod
    def _valid_replay(info, ping):
        """Make sure the replay isn't corrupt, and is worth looking at."""
        if (info.HasField("error") or
                    info.base_build != ping.base_build or  # different game version
                    info.game_duration_loops < 1000 or
                    len(info.player_info) != 2):
            # Probably corrupt, or just not interesting.
            return False
#   for p in info.player_info:
#       if p.player_apm < 10 or p.player_mmr < 1000:
#           # Low APM = player just standing around.
#           # Low MMR = corrupt replay or player who is weak.
#           return False
        return True


    # def orig(self, replay):
    #     _features = features.features_from_game_info(self.controller.game_info())
    #     label = 0
    #     minimap = []
    #     screen = []
    #     non_spatials = []
    #     counter = 0
    #     error = 0
    #     while True:
    #         self.step_mul = 100
    #         self.controller.step(self.step_mul)
    #         obs = self.controller.observe()
    #         try:
    #             agent_obs = _features.transform_obs(obs)
    #             counter +=1
    #         except:
    #             error += 1
    #             print("error ", self._episode_steps, " out of ", self._episode_length, " counter is ", counter, "/", error)
    #             pass
    #         img = agent_obs['feature_screen']
    #         mini = agent_obs['feature_minimap']
    #
    #         if obs.player_result:  # Episide over.
    #             self._state = StepType.LAST
    #             discount = 0
    #         else:
    #             discount = self.discount
    #
    #         self._episode_steps += self.step_mul
    #
    #         step = TimeStep(step_type=self._state, reward=0,
    #                         discount=discount, observation=agent_obs)
    #
    #         self.agent.step(step, obs.actions)
    #
    #         if obs.player_result:
    #             break
    #
    #         self._state = StepType.MID
    #     return [np.empty([])]

    def orig(self, replay):
        self.step_mul = 8
        _features = features.features_from_game_info(self.controller.game_info())
        label = 0
        minimaps = [np.empty((64,64,7), dtype=np.int32)]
        screens = [np.empty((64,64,17), dtype=np.int32)]
        non_spatials = np.empty((1, 541+11), dtype=np.int32)
        # times = get_random_steps(self._episode_length, 64)
        times = random.sample(range(self._episode_length//8), 100)
        times.sort()
        print(times)
        counter = 0
        error = 0
        next_time = 0
        while len(minimaps)<64:
            # self.step_mul = times[counter]
            self.controller.step(self.step_mul)
            obs = self.controller.observe()
            try:
                if times[counter]*8 == self._episode_steps:
                    counter += 1
                    agent_obs = _features.transform_obs(obs)
                    screen, minimap, info = extract_features(agent_obs)
                    screens = np.append(screens, [screen], axis=0)
                    minimaps = np.append(minimaps, [minimap], axis=0)
                    non_spatials = np.append(non_spatials, [info], axis=0)
                else:
                    agent_obs = _features.transform_obs(obs)

            except:
                error += 1
                print("error ", self._episode_steps, " out of ", self._episode_length, " counter is ", counter, "/", error)
                pass

            if obs.player_result:  # Episide over.
                self._state = StepType.LAST
                discount = 0
            else:
                discount = self.discount

            self._episode_steps += self.step_mul

            step = TimeStep(step_type=self._state, reward=0,
                            discount=discount, observation=agent_obs)

            self.agent.step(step, obs.actions)

            if obs.player_result:
                break

            self._state = StepType.MID

        self.sc2_proc.close()
        print(len(minimaps), ',',len(screens), ',',len(non_spatials))
        return minimaps, screens, non_spatials

    def get_smooth_observation(self, replay_path):
        _features = features.features_from_game_info(self.controller.game_info())
        label = 0
        minimaps = [np.empty((64,64,7), dtype=np.float32)]
        screens = [np.empty((64,64,17), dtype=np.float32)]
        non_spatials = np.empty((1, 541+11), dtype=np.float32)
        X = [np.empty([])]
        times = get_random_steps(self._episode_length, 64)
        print(times)
        for random_jump in times:
            while True and len(minimaps) < 64:
                self.step_mul = 20
                self.controller.step(self.step_mul)
                obs = self.controller.observe()
                agent_obs = _features.transform_obs(obs)
                try:
                    agent_obs = _features.transform_obs(obs)
                    # screen, minimap, info = extract_features(agent_obs)
                    # screens = np.append(screens, [screen], axis=0)
                    # minimaps = np.append(minimaps, [minimap], axis=0)
                    # non_spatials = np.append(non_spatials, info)
                except:
                    print("Error")
                    pass


                if obs.player_result: # Episide over.
                    self._state = StepType.LAST
                    discount = 0
                else:
                    discount = self.discount

                self._episode_steps += self.step_mul

                step = TimeStep(step_type=self._state, reward=0,
                                discount=discount, observation=agent_obs)

                self.agent.step(step, obs.actions)

                if obs.player_result:
                    break

                self._state = StepType.MID

        self._state = StepType.END
        X = np.array([minimaps, screens, non_spatials])
        return X




    def get_one_observation(self, replay_path):
        _features = features.features_from_game_info(self.controller.game_info())
        label = 0
        minimaps = [np.empty((64,64,7), dtype=np.float32)]
        screens = [np.empty((64,64,17), dtype=np.float32)]
        non_spatials = np.empty((1, 541+11), dtype=np.float32)
        X = [np.empty([])]
        times = get_random_steps(1000, 3)
        print(times)
        # replay_read = sc2reader.load_replay(replay_path, load_map=True)
        for random_jump in times:
            # self.controller.step(self.step_mul)
            while random_jump > 40:
                random_jump -= 40
                self.controller.step(40)
                self._episode_steps += 40
                print(self._episode_steps)
                obs = self.controller.observe()
                agent_obs = _features.transform_obs(obs)
                if obs.player_result:  # Episide over.
                    self._state = StepType.LAST
                    discount = 0
                else:
                    discount = self.discount
                step = TimeStep(step_type=self._state, reward=0,
                                discount=discount, observation=agent_obs)

                self.agent.step(step, obs.actions)
                self._state = StepType.MID

            self.controller.step(random_jump)

            obs = self.controller.observe()

            # try:
            agent_obs = _features.transform_obs(obs)
            screen, minimap, info = extract_features(agent_obs)
            screens = np.append(screens, [screen], axis=0)
            minimaps = np.append(minimaps, [minimap], axis=0)
            non_spatials = np.append(non_spatials, info)

            # X = np.append(X, [screen, mini, non_spatials], axis=0)
            # except:
            #     pass

            if obs.player_result: # Episide over.
                self._state = StepType.LAST
                discount = 0
            else:
                discount = self.discount

            self._episode_steps += random_jump
            # self._episode_steps += self.step_mul

            time.sleep(1)
            step = TimeStep(step_type=self._state, reward=0,
                            discount=discount, observation=agent_obs)

            self.agent.step(step, obs.actions)

            if obs.player_result:
                break

            self._state = StepType.MID
        X = np.array([minimaps, screens, non_spatials])
        return X

def get_label(replay_read):
    if replay_read.winner.number != 1:
        return 0
    return 1

def get64obs(replay_file):
    test_replay = os.path.join(PATH_REPLAY, replay_file)
    print(test_replay)
    agent_module, agent_name = 'ObserverAgent.ObserverAgent'.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)
    # replay_read = sc2reader.load_replay(test_replay)
    # label = get_label(replay_read)
    label = 1
    # G_O_O_D_B_O_Y_E = ReplayEnv(FLAGS.replay, agent_cls())
    G_O_O_D_B_O_Y_E = ReplayEnv(test_replay, agent_cls())
    Xs, Xm, Xsp = G_O_O_D_B_O_Y_E.orig(test_replay)
    # print(Xs)
    # X = G_O_O_D_B_O_Y_E.get_one_observation(test_replay)
    if label == 0:
        Y = np.zeros(3)
    elif label == 1:
        Y = np.ones(3)
    return [Xs,Xm,Xsp], Y


def clean_data(replay_file_path,
                 version = '3.16.1'):

    def _valid_replay(info, ping):
        """Make sure the replay isn't corrupt, and is worth looking at."""
        if (info.HasField("error") or
                    info.base_build != ping.base_build or  # different game version
                    info.game_duration_loops < 1000 or
                    len(info.player_info) != 2):
            return False
        return True

    new_path = "D:/University_Work/My_research/fixed_replays/bad_replays/"
    os.chdir(replay_file_path)
    all_replays = os.listdir(replay_file_path)
    run_config = run_configs.get()
    sc2_proc = run_config.start(version)
    controller = sc2_proc.controller
    counter = 0
    # for replay in reversed(list(glob.glob('*.SC2Replay'))):
    for i in range(len(all_replays)-26166, 1, -1):
        print(all_replays[i])
        # print(counter)
        old_replay = replay_file_path + all_replays[i]
        replay_data = run_config.replay_data(old_replay)
        ping = controller.ping()
        info = controller.replay_info(replay_data)
        if not _valid_replay(info, ping):
            print(counter)
            new_replay = new_path + all_replays[i]
            os.rename(old_replay, new_replay)
            sc2_proc.close()

        counter+=1

if __name__ == "__main__":
    # clean_data('D:/University_Work/My_research/fixed_replays/Replays/')
    app.run(get_smooth_observation)
