import gym
from gym.spaces import Box, Dict
import numpy as np
from numpy import linalg as LA

from safe_explorer.core.config import Config

import numpy as np
import pybullet as p
import pybullet_data

class Ball2D_pybullet():
    def __init__(self):
        self._config = Config.get().env.ballnd

        self._physicsClient = p.connect(p.GUI)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._plane = p.loadURDF("plane.urdf")
        self._agent_1 = p.loadURDF("cube.urdf", [1, 0, 0], globalScaling=0.1)
        self._agent_2 = p.loadURDF("cube.urdf", [0, 1, 0], globalScaling=0.1)
        p.changeVisualShape(self._agent_1, -1, rgbaColor=[1, 1, 0, 0.7])
        p.changeVisualShape(self._agent_2, -1, rgbaColor=[0, 1, 1, 0.7])
        self._target = p.loadURDF("sphere_small.urdf", [0, 0, 0], globalScaling=1)
        
        self._cam_position, _ = p.getBasePositionAndOrientation(self._target)
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=135, cameraPitch=-45, cameraTargetPosition=self._cam_position)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self._config.frequency_ratio) 
        p.setRealTimeSimulation(0)

        # Set the properties for spaces
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = Dict({
            'agent_position': Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'rival_position': Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'target_position': Box(low=0, high=1, shape=(2,), dtype=np.float32)
        })

        # Sets all the episode specific variables
        self.reset()
        
    def reset(self):
        self._agent_1_position = \
            (1 - 2 * self._config.agent_slack) * np.random.random(2) + self._config.agent_slack
        p.resetBasePositionAndOrientation(self._agent_1, [self._agent_1_position[0], self._agent_1_position[1], 0.0] , [0.0, 0.0, 0.0, 1.0])

        self._agent_2_position = \
            (1 - 2 * self._config.agent_slack) * np.random.random(2) + self._config.agent_slack
        p.resetBasePositionAndOrientation(self._agent_1, [self._agent_2_position[0], self._agent_2_position[1], 0.0] , [0.0, 0.0, 0.0, 1.0])

        self._reset_target_location()
        self._current_time = 0.

        observation_1, step_reward_1, observation_2, step_reward_2, done, _ = self.step(np.zeros(2), np.zeros(2))
        
        return observation_1, observation_2
    
    def _get_reward_1(self):
        if self._config.enable_reward_shaping and self._is_agent_1_outside_shaping_boundary():
            return -1
        else:
            return np.clip(1 - 10 * LA.norm(self._agent_1_position - self._target_position) ** 2, 0, 1)
    
    def _get_reward_2(self):
        if self._config.enable_reward_shaping and self._is_agent_2_outside_shaping_boundary():
            return -1
        else:
            return np.clip(1 - 10 * LA.norm(self._agent_2_position - self._target_position) ** 2, 0, 1)
    
    def _reset_target_location(self):
        self._target_position = \
            (1 - 2 * self._config.target_margin) * np.random.random(2) + self._config.target_margin
        p.resetBasePositionAndOrientation(self._target, [self._target_position[0], self._target_position[1], 0.0], [0.0, 0.0, 0.0, 1.0])
    
    def _move_agent_1(self, velocity):
        # Assume that frequency of motor is 1 (one action per second)
        self._agent_1_position += self._config.frequency_ratio * velocity
        p.resetBasePositionAndOrientation(self._agent_1, [self._agent_1_position[0], self._agent_1_position[1], 0.0] , [0.0, 0.0, 0.0, 1.0])

    def _move_agent_2(self, velocity):
        # Assume that frequency of motor is 1 (one action per second)
        self._agent_2_position += self._config.frequency_ratio * velocity
        p.resetBasePositionAndOrientation(self._agent_2, [self._agent_2_position[0], self._agent_2_position[1], 0.0] , [0.0, 0.0, 0.0, 1.0])
    
    def _is_agent_1_outside_boundary(self):
        return np.any(self._agent_1_position < 0) or np.any(self._agent_1_position > 1)
    
    def _is_agent_2_outside_boundary(self):
        return np.any(self._agent_2_position < 0) or np.any(self._agent_2_position > 1)
    
    def _is_agent_1_outside_shaping_boundary(self):
        return np.any(self._agent_1_position < self._config.reward_shaping_slack) \
               or np.any(self._agent_1_position > 1 - self._config.reward_shaping_slack)
    
    def _is_agent_2_outside_shaping_boundary(self):
        return np.any(self._agent_2_position < self._config.reward_shaping_slack) \
               or np.any(self._agent_2_position > 1 - self._config.reward_shaping_slack)

    def _update_time(self):
        # Assume that frequency of motor is 1 (one action per second)
        self._current_time += self._config.frequency_ratio
    
    def _get_noisy_target_position(self):
        return self._target_position + \
               np.random.normal(0, self._config.target_noise_std, 2)
    
    def get_num_constraints(self):
        return 2 * 2 + 2 #2*dim + 2*(n_agents-1)

    def get_constraint_values(self):
        # For any given n, there will be 2 * n constraints
        # a lower and upper bound for each dim
        # We define all the constraints such that C_i = 0
        # _agent_1_position > 0 + _agent_slack => -_agent_1_position + _agent_slack < 0
        min_constraints_1 = self._config.agent_slack - self._agent_1_position
        # _agent_1_position < 1 - _agent_slack => _agent_1_position + agent_slack- 1 < 0
        max_constraint_1 = self._agent_1_position  + self._config.agent_slack - 1
        # |_agent_1_position-_agent_2_position| > _rival_slack => - |_agent_1_position-_agent_2_position| + _rival_slack < 0
        rival_constraint_1 = - np.abs(self._agent_1_position-self._agent_2_position)  + self._config.rival_slack

        min_constraints_2 = self._config.agent_slack - self._agent_2_position
        max_constraint_2 = self._agent_2_position  + self._config.agent_slack - 1
        rival_constraint_2 = - np.abs(self._agent_2_position-self._agent_1_position)  + self._config.rival_slack

        return np.concatenate([min_constraints_1, max_constraint_1, rival_constraint_1]), np.concatenate([min_constraints_2, max_constraint_2, rival_constraint_2])

    def step(self, action_1, action_2):

        # Check if the target needs to be relocated
        # Extract the first digit after decimal in current_time to add numerical stability
        if (int(100 * self._current_time) // 10) % (self._config.respawn_interval * 10) == 0:
            self._reset_target_location()

        # Increment time
        self._update_time()

        last_reward_1 = self._get_reward_1()
        last_reward_2 = self._get_reward_2()
        # Calculate new position of the agent
        self._move_agent_1(action_1)
        self._move_agent_2(action_2)

        # Find reward         
        reward_1 = self._get_reward_1()
        step_reward_1 = reward_1 - last_reward_1

        reward_2 = self._get_reward_2()
        step_reward_2 = reward_2 - last_reward_2

        # Prepare return payload
        observation_1 = {
            "agent_position": self._agent_1_position,
            "rival_position": self._agent_2_position,
            "target_postion": self._get_noisy_target_position()
        }

        observation_2 = {
            "agent_position": self._agent_2_position,
            "rival_position": self._agent_1_position,
            "target_postion": self._get_noisy_target_position()
        }

        done = self._is_agent_1_outside_boundary() \
               or self._is_agent_2_outside_boundary() \
               or int(self._current_time // 1) > self._config.episode_length

        return observation_1, step_reward_1, observation_2, step_reward_2, done, {}
