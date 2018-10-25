#!/usr/bin/env python
# python transform_replay.py --replay '000a4ab29a10c7db1e2e7d0dcde9aad01fb297a703417c03e4a5137c0fb2af0d.SC2Replay' --agent ObserverAgent.ObserverAgent
class ObserverAgent():
    def step(self, time_step, actions):
        print("{}".format(time_step.observation["game_loop"]))
