import os, csv, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import load_config, set_seed, ensure_dirs
from envs import SmartRoomEnv
from algorithms.multi_head_dqn import MultiHeadDQNAgent

def evaluate(agent, env, episodes=5):
    scores = []
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            a = agent.select_action(s, greedy=True)
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc
            total += float(r)
        scores.append(total)
    return float(np.mean(scores)), float(np.std(scores))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/configs/multihead.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    out_dir = cfg.get("out_dir", "results")
    ensure_dirs(out_dir)

    env = SmartRoomEnv(
        dt_minutes=cfg.get("env", {}).get("dt_minutes", 15),
        scenario_csv=cfg.get("env", {}).get("scenario_csv", None)
    )
    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec.tolist()

    agent = MultiHeadDQNAgent(state_dim, action_dims, config={
        "buffer_size": cfg["algo"]["buffer_size"],
        "batch_size": cfg["algo"]["batch_size"],
        "gamma": cfg["algo"]["gamma"],
        "lr": cfg["algo"]["lr"],
        "epsilon_start": cfg["algo"]["epsilon_start"],
        "epsilon_end": cfg["algo"]["epsilon_end"],
        "epsilon_decay": cfg["algo"]["epsilon_decay"],
        "tau": cfg["algo"].get("tau", 0.01)
    })

    log_csv = os.path.join(out_dir, "logs", "multihead_training_log.csv")
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "reward", "epsilon", "eval_mean", "eval_std"])

    episodes = cfg["train"]["episodes"]
    save_every = cfg["train"].get("save_every", 20)
    eval_every = cfg["train"].get("eval_every", 50)
    eval_eps = cfg["train"].get("eval_episodes", 5)

    rewards = []
    best_reward = float("-inf")

    for ep in range(episodes):
        ep_reward = agent.train_episode(env)
        rewards.append(ep_reward)

        eval_mean, eval_std = (np.nan, np.nan)
        if eval_every and (ep % eval_every == 0 or ep == episodes - 1):
            eval_mean, eval_std = evaluate(agent, env, episodes=eval_eps)

        print(f"[MultiHead] Ep {ep:04d} | R={ep_reward:.2f} | eps={agent.eps:.3f} | Eval={eval_mean:.2f}±{eval_std:.2f}")
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, ep_reward, agent.eps, eval_mean, eval_std])

        if save_every and (ep % save_every == 0 or ep == episodes - 1):
            torch.save(agent.q_net.state_dict(), os.path.join(out_dir, "models", f"multihead_checkpoint_{ep}.pt"))

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(agent.q_net.state_dict(), os.path.join(out_dir, "models", "multihead_best.pt"))

    torch.save(agent.q_net.state_dict(), os.path.join(out_dir, "models", "multihead_latest.pt"))

    fig_path = os.path.join(out_dir, "figures", "multihead_reward_curve.png")
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Multi-Head DQN Training Curve")
    plt.savefig(fig_path, dpi=160)
    plt.close()

if __name__ == "__main__":
    main()

