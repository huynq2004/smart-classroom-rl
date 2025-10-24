import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import SmartRoomEnv
from algorithms.bdq_lite import BDQLiteAgent

env = SmartRoomEnv(dt_minutes=15, scenario_csv=None)
state_dim = env.observation_space.shape[0]
action_dims = env.action_space.nvec.tolist()

agent = BDQLiteAgent(state_dim, action_dims, config={
    "gamma": 0.99, "lr": 0.001, "buffer_size": 1, "batch_size": 1
})
agent.q_net.load_state_dict(torch.load("results/bdq/models/bdq_best.pt", map_location="cpu", weights_only=True))
agent.q_net.eval()
agent.eps = 0.0

s, _ = env.reset()
done = False
total_reward = 0

while not done:
    a = agent.select_action(s, greedy=True)
    s2, r, term, trunc, info = env.step(a)
    done = term or trunc
    total_reward += float(r)
    s = s2

print(f"Reward 1 episode test: {total_reward:.2f}")
