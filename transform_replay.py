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

FLAGS = flags.FLAGS
flags.DEFINE_string("replay_path", None, "Path to a replay files.")
flags.DEFINE_string("agent", None, "Path to an agent.")
flags.mark_flag_as_required("replay_path")
flags.mark_flag_as_required("agent")

class ReplayEnv:
    def __init__(self,
                 replay_file_path,
                 agent,
                 player_id=2,
                 screen_size_px=(64, 64),
                 minimap_size_px=(64, 64),
                 discount=1.,
                 step_mul=8,
                 version = '4.6.0'):

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
        info = self.controller.replay_info(replay_data)
        if not self._valid_replay(info, ping):
            raise Exception("{} is not a valid replay file!".format(replay_file_path))

        screen_size_px = point.Point(*screen_size_px)
        minimap_size_px = point.Point(*minimap_size_px)
        interface = sc_pb.InterfaceOptions(
            raw=False, score=True,
            feature_layer=sc_pb.SpatialCameraSetup(width=24))
        screen_size_px.assign_to(interface.feature_layer.resolution)
        minimap_size_px.assign_to(interface.feature_layer.minimap_resolution)

        map_data = None
        if info.local_map_path:
            map_data = self.run_config.map_data(info.local_map_path)

        self._episode_length = info.game_duration_loops
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

    def get_one_observation(self, replay_path):
        _features = features.features_from_game_info(self.controller.game_info())
        label = 0
        minimap = []
        screen = []
        non_spatials = []

        replay_read = sc2reader.load_replay(replay_path, load_map=True)
        max_time = replay_read.length.seconds
        while True:
            self.controller.step(self.step_mul)
            obs = self.controller.observe()
            try:
                agent_obs = _features.transform_obs(obs)
            except:
                pass
            img = agent_obs['feature_screen']
            mini = agent_obs['feature_minimap']

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

def get_label(replay_read):
    if replay_read.winner.number != 1:
        return 0
    return 1

def get64obs(noned):
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)
    replay_path = 'Blueshift LE (19).SC2Replay'
    replay_read = sc2reader.load_replay(replay_path, load_map=True)
    label = get_label(replay_read)
    # G_O_O_D_B_O_Y_E = ReplayEnv(FLAGS.replay, agent_cls())
    G_O_O_D_B_O_Y_E = ReplayEnv('Blueshift LE (19).SC2Replay', agent_cls())
    return G_O_O_D_B_O_Y_E.get_one_observation(replay_path)


if __name__ == "__main__":
    app.run(get64obs)
