import torch

class GlobalRestartManager:
    """
    Global Restart via Randomization.
    Escapes poor local Nash when reaching reward plateau.
    Implements Giannou's local guarantees + Vlad's Global extension.
    """
    def __init__(self, patience=15, threshold=-500):
        self.patience = patience
        self.threshold = threshold
        self.best_reward = -float('inf')
        self.stuck_counter = 0
        self.restart_count = 0

    def check_and_restart(self, current_reward, policy_net):
        """Returns True if restart triggered."""
        if current_reward > self.best_reward + 10:
            self.best_reward = current_reward
            self.stuck_counter = 0
        else:
            self.stuck_counter += 1

        # Trigger if stuck AND sub-optimal (below threshold)
        if self.stuck_counter >= self.patience and self.best_reward < self.threshold:
            self.stuck_counter = 0
            self.best_reward = -float('inf')
            self.restart_count += 1
            
            # Re-init weights (random mapping to different basin)
            for layer in policy_net.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            return True
        return False
