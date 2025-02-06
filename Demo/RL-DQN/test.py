import sys
import os
sys.path.append(os.path.abspath("/home/brad/developments/Demo/RL-DQN/"))
import torch
from Enviroments import CartPoleEnvWrapper

def test(model_path):
    env = CartPoleEnvWrapper()
    model = torch.load(model_path)
    model.eval()
    
    state = env.reset()
    total_reward = 0
    
    while True:
        action = model(torch.FloatTensor(state)).argmax().item()
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        
        if done:
            print(f"Total reward: {total_reward}")
            break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/cartpole_model.pth", help="Path to the trained model")
    args = parser.parse_args()
    test(args.model_path)