import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.episode_rewards = []
        self.start_time = time.time()
        
    def log_metrics(self, episode, epsilon, reward, loss=None):
        """记录训练指标"""
        self.episode_rewards.append(reward)
        
        # 记录到TensorBoard
        self.writer.add_scalar('Episode Reward', reward, episode)
        self.writer.add_scalar('Epsilon', epsilon, episode)
        if loss is not None:
            self.writer.add_scalar('Training Loss', loss, episode)
            
        # 控制台输出
        if episode % 20 == 0:
            avg_reward = np.mean(self.episode_rewards[-20:])
            time_elapsed = time.time() - self.start_time
            print(f"Episode: {episode:4d} | "
                  f"Avg Reward: {avg_reward:6.1f} | "
                  f"Epsilon: {epsilon:4.2f} | "
                  f"Time: {time_elapsed:4.0f}s")
            
    def close(self):
        self.writer.close()