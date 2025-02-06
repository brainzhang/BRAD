
# Environment Wrapper for CartPole-v1
import gym

class CartPoleEnvWrapper:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

    def reset(self):
        state, _ = self.env.reset()  # 兼容 Gym 0.26+
        return state

    def step(self, action):
        results = self.env.step(action)
        if len(results) == 5:
            state, reward, terminated, truncated, info = results
            done = terminated or truncated
        else:
            state, reward, done, info = results
        return state, reward, done, info
