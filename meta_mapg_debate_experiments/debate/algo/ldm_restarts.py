class LDM_Restarter:
    """
    LLM Restarts. Triggers when language model falls into Deceptive/Collusive Nash.
    Detects plateaus in Truthfulness score using Pluralistic judges.
    """
    def __init__(self, patience=3, threshold=0):
        self.patience = patience
        self.threshold = threshold
        self.stuck_counter = 0
        self.restart_count = 0
        
    def check_and_restart(self, recent_rewards, peft_model):
        """
        Reinitialize LoRA adapter weights if stuck in poor embedding space.
        """
        avg_reward = sum(recent_rewards) / (len(recent_rewards) + 1e-8)
        if avg_reward < self.threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            
        if self.stuck_counter >= self.patience:
            self.stuck_counter = 0
            self.restart_count += 1
            
            # Reset LoRA A/B matrices
            for name, module in peft_model.named_modules():
                if 'lora_A' in name or 'lora_B' in name:
                    torch.nn.init.normal_(module.weight, std=0.02)
            return True
        return False
