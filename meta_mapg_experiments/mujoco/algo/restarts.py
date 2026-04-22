import torch
from collections import deque

class GlobalRestartManager:
    """
    Global Restart via Sliding-Window Reward Monitor.
    
    Implements the restart mechanism of Giannou et al. (NeurIPS 2022, §4):
    restart when the sliding-window average reward stays below a Nash-quality
    threshold for `patience` consecutive episodes, indicating the iterate
    is trapped in a sub-optimal basin.
    
    Unlike a high-water-mark approach, this correctly detects agents that
    temporarily reach a good reward but fall back into a bad gait.
    """
    def __init__(self, patience=100, threshold=-500):
        self.patience = patience
        self.threshold = threshold
        self.reward_window = deque(maxlen=patience)
        self.restart_count = 0

    def check_and_restart(self, current_reward, policy_net):
        """Returns True if restart triggered."""
        self.reward_window.append(current_reward)
        
        # Only evaluate once we have a full window
        if len(self.reward_window) < self.patience:
            return False
        
        window_avg = sum(self.reward_window) / len(self.reward_window)
        
        # Restart if the sliding-window average is below the Nash threshold
        if window_avg < self.threshold:
            self.restart_count += 1
            self.reward_window.clear()
            
            # Re-init weights (random mapping to different basin)
            for layer in policy_net.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            return True
        return False
