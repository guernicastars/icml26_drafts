import numpy as np
import gymnasium as gym

class TwoAgentHalfCheetah:
    """
    Two-Agent HalfCheetah Environment.
    Agent 0 controls front joints, Agent 1 controls back joints.
    Shared state (non-stationarity core).
    """
    def __init__(self, config=None):
        self.env = gym.make("HalfCheetah-v4")
        self.action_dim = self.env.action_space.shape[0] // 2
        self.obs_dim = self.env.observation_space.shape[0]
        self.agents = ["agent_0", "agent_1"]

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return {a: obs for a in self.agents}, info

    def step(self, action_dict):
        # Merge actions: [front_legs, back_legs]
        a0 = np.clip(action_dict["agent_0"], -1.0, 1.0)
        a1 = np.clip(action_dict["agent_1"], -1.0, 1.0)
        joint_action = np.concatenate([a0, a1])
        
        obs, reward, terminated, truncated, info = self.env.step(joint_action)
        
        # Fully cooperative: share reward
        rewards = {a: reward for a in self.agents}
        dones = {a: terminated or truncated for a in self.agents}
        dones["__all__"] = terminated or truncated
        obss = {a: obs for a in self.agents}
        
        return obss, rewards, dones, info
