import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os, csv, argparse, time
import numpy as np
import torch
import matplotlib.pyplot as plt

from envs import SmartRoomEnv
from algorithms.ppo_agent import PPOAgent
from config import load_config, set_seed, ensure_dirs
from algorithms.common.utils import make_result_dirs   


def make_envs(n, env_kwargs):
    return [SmartRoomEnv(**env_kwargs) for _ in range(n)]


def collect_one_episode(agent, env):
    """Chạy 1 episode, trả (buffer, episode_return)."""
    s, _ = env.reset()
    S, A, LP, R, V, D = [], [], [], [], [], []

    done = False
    while not done:
        a, logp, v = agent.act(s)
        s_next, r, term, trunc, _ = env.step(a)
        done = bool(term or trunc)

        S.append(s)
        A.append(a)
        LP.append(logp)
        R.append(r)
        V.append(v)
        D.append(done)

        s = s_next

    # V_last để tính GAE
    _, _, v_last = agent.act(s)
    V.append(v_last)  # len = T+1
    return {
        "states": S, "actions": A, "logps": LP,
        "rewards": R, "values": V, "dones": D
    }, float(sum(R))


def evaluate(agent, env, episodes=5):
    scores = []
    for _ in range(episodes):
        s, _ = env.reset()
        done, tot = False, 0.0
        while not done:
            a, _, _ = agent.act(s)
            s, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            tot += r
        scores.append(tot)
    return float(np.mean(scores)), float(np.std(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/ppo.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    out_dir = cfg.get("out_dir", "results")
    ensure_dirs(out_dir)

    # Tạo thư mục riêng cho PPO
    paths = make_result_dirs("ppo")

    # Env config
    env_kwargs = {
        "dt_minutes":  cfg["env"].get("dt_minutes", 15),
        "scenario_csv": cfg["env"].get("scenario_csv", None),
    }

    EPISODES   = cfg["train"]["episodes"]
    NUM_ACTORS = cfg["train"].get("num_actors", 1)
    EVAL_EVERY = cfg["train"].get("eval_every", 20)
    EVAL_EPS   = cfg["train"].get("eval_episodes", 5)
    SAVE_EVERY = cfg["train"].get("save_every", 50)

    envs = make_envs(NUM_ACTORS, env_kwargs)
    # Suy ra kích thước không gian
    state_dim   = envs[0].observation_space.shape[0]
    action_dims = envs[0].action_space.nvec.tolist()

    agent = PPOAgent(state_dim, action_dims, cfg["algo"])

    # Đường dẫn riêng cho PPO
    log_path = os.path.join(paths["logs"], "ppo_training_log.csv")
    fig_path = os.path.join(paths["figures"], "ppo_reward_curve.png")

    # Khởi tạo file log
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["episode","reward","eval_mean","eval_std",
                                "policy_loss","value_loss","entropy"])

    reward_curve = []
    best_eval = -np.inf

    # Train theo EPISODE: mỗi vòng gom N episode → update
    for ep in range(EPISODES):
        t0 = time.time()

        # Thu thập N episode (có thể xem như "mini-batch" theo episode)
        batch_states, batch_actions, batch_logps, batch_advs, batch_returns = [], [], [], [], []
        ep_rewards = []

        for i in range(NUM_ACTORS):
            buf, ep_ret = collect_one_episode(agent, envs[i])
            advs, rets = agent.compute_gae(buf["rewards"], buf["values"], buf["dones"])

            batch_states.extend(buf["states"])
            batch_actions.extend(buf["actions"])
            batch_logps.extend(buf["logps"])
            batch_advs.extend(advs.tolist())
            batch_returns.extend(rets.tolist())
            ep_rewards.append(ep_ret)

        # Chuẩn hoá mảng states → [M, state_dim]
        states_np = np.vstack([np.asarray(s, dtype=np.float32) for s in batch_states])

        rollout = {
            "states":  states_np,
            "actions": np.asarray(batch_actions, dtype=np.int64),
            "logps":   np.asarray(batch_logps, dtype=np.float32),
            "advs":    np.asarray(batch_advs, dtype=np.float32),
            "returns": np.asarray(batch_returns, dtype=np.float32)
        }

        stats = agent.update(rollout)
        mean_ep_reward = float(np.mean(ep_rewards))
        reward_curve.append(mean_ep_reward)

        eval_mean, eval_std = (np.nan, np.nan)
        if (ep % EVAL_EVERY == 0) or (ep == EPISODES - 1):
            eval_mean, eval_std = evaluate(agent, envs[0], episodes=EVAL_EPS)
            if eval_mean > best_eval:
                best_eval = eval_mean
                torch.save(agent.net.state_dict(), os.path.join(paths["models"], "ppo_best.pt"))

        if ep % SAVE_EVERY == 0:
            torch.save(agent.net.state_dict(), os.path.join(paths["models"], f"ppo_ckpt_{ep}.pt"))
        torch.save(agent.net.state_dict(), os.path.join(paths["models"], "ppo_latest.pt"))

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                ep, mean_ep_reward, eval_mean, eval_std,
                stats.get("policy_loss", np.nan),
                stats.get("value_loss",  np.nan),
                stats.get("entropy",     np.nan)
            ])

        print(f"[PPO] Ep {ep:04d} | Reward={mean_ep_reward:.3f} | Eval={eval_mean:.3f}±{eval_std:.3f} | Time={time.time()-t0:.1f}s")

    #  Lưu và vẽ biểu đồ vào thư mục riêng
    plt.plot(reward_curve)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (mean over N actors)")
    plt.title(f"PPO-Clip (Episode-based, N={NUM_ACTORS})")
    plt.grid(True)
    plt.savefig(fig_path, dpi=160)
    plt.close()

    print(f" Saved log -> {log_path}")
    print(f" Saved figure -> {fig_path}")


if __name__ == "__main__":
    main()
