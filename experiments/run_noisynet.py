# experiments/run_noisynet.py
import os, csv, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import SmartRoomEnv
from algorithms.noisynet_dqn import NoisyDQNAgent
from config import load_config, set_seed, ensure_dirs

def evaluate(agent, env, episodes=5):
    scores = []
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        tot = 0.0
        while not done:
            a = agent.select_action(s, greedy=True)
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc
            tot += float(r)
        scores.append(tot)
    return float(np.mean(scores)), float(np.std(scores))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/noisynet.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    out_dir = cfg.get("out_dir", "results")
    ensure_dirs(out_dir)

    env = SmartRoomEnv(
        dt_minutes=cfg.get("env", {}).get("dt_minutes", 15),
        scenario_csv=cfg.get("env", {}).get("scenario_csv", None),
        noise_std=cfg.get("env", {}).get("noise_std", 0.1),
    )
    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec.tolist()

    agent = NoisyDQNAgent(state_dim, action_dims, config={
        "buffer_size": cfg["algo"]["buffer_size"],
        "batch_size": cfg["algo"]["batch_size"],
        "gamma": cfg["algo"]["gamma"],
        "lr": cfg["algo"]["lr"],
        "tau": cfg["algo"]["tau"]
    })

    log_csv = os.path.join(out_dir, "logs", "noisynet_training_log.csv")
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "reward", "eval_mean", "eval_std", "loss"])

    episodes = cfg["train"]["episodes"]
    save_every = cfg["train"].get("save_every", 20)
    eval_every = cfg["train"].get("eval_every", 50)
    eval_eps = cfg["train"].get("eval_episodes", 5)

    rewards = []
    best = float("-inf")
    for ep in range(episodes):
        ep_reward = agent.train_episode(env)
        rewards.append(ep_reward)

        eval_mean, eval_std = (float("nan"), float("nan"))
        if eval_every and (ep % eval_every == 0 or ep == episodes-1):
            eval_mean, eval_std = evaluate(agent, env, episodes=eval_eps)

        print(f"[NoisyNet] Ep {ep:04d} | R={ep_reward:.2f} | Eval={eval_mean:.2f}±{eval_std:.2f}")
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, ep_reward, eval_mean, eval_std, ""])

        if save_every and (ep % save_every == 0 or ep == episodes-1):
            torch.save(agent.q_net.state_dict(), os.path.join(out_dir, "models", f"noisynet_ckpt_{ep}.pt"))

        if ep_reward > best:
            best = ep_reward
            torch.save(agent.q_net.state_dict(), os.path.join(out_dir, "models", "noisynet_best.pt"))

    torch.save(agent.q_net.state_dict(), os.path.join(out_dir, "models", "noisynet_latest.pt"))

    fig_path = os.path.join(out_dir, "figures", "noisynet_reward_curve.png")
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("NoisyNet Training Curve")
    plt.savefig(fig_path, dpi=160)
    plt.close()

if __name__ == "__main__":
    main()
