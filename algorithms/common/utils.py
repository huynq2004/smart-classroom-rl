import os
import csv
import torch
import matplotlib.pyplot as plt

import torch
import numpy as np


def gather_per_branch(Q_heads, actions):
    """
    Q_heads: list length N, mỗi phần tử Tensor [B, n_a]
    actions: LongTensor [B, N] (chỉ số hành động theo từng nhánh)
    return : list length N, mỗi phần tử Tensor [B] = Q_d(s, a_d)
    """
    B = actions.size(0)
    out = []
    for d, Qd in enumerate(Q_heads):
        a_idx = actions[:, d].long().view(B, 1)  # [B,1]
        out.append(Qd.gather(1, a_idx).squeeze(1))
    return out


def mean_over_branches(vals_list):
    """
    vals_list: list length N, phần tử shape [B] hoặc [B,1]
    return   : Tensor [B], trung bình theo nhánh
    """
    stack = torch.stack([v.view(-1) for v in vals_list], dim=1)  # [B, N]
    return stack.mean(dim=1)


def hard_update(target_net, online_net):
    target_net.load_state_dict(online_net.state_dict())


def epsilon_linear(eps_start, eps_end, eps_decay, step):
    """
    Epsilon giảm tuyến tính: sau 'eps_decay' bước sẽ đạt eps_end.
    """
    if eps_decay <= 0:
        return eps_end
    progress = min(1.0, step / float(eps_decay))
    return float(eps_start + (eps_end - eps_start) * progress)


def set_seed_everywhere(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def soft_update(target_net, online_net, tau=0.01):
    for tp, op in zip(target_net.parameters(), online_net.parameters()):
        tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)

def epsilon_decay(eps, eps_end, decay_rate):
    return max(eps_end, eps - decay_rate)

def compute_q_sum(Q_heads, actions):
    """
    Q(s,a) = ∑_i Q_i(s, a_i)
    - Q_heads: list [B, n_i]
    - actions: LongTensor [B, n_heads]
    return: FloatTensor [B]
    """
    q_sum = 0.0
    for i, Q_i in enumerate(Q_heads):
        a_i = actions[:, i].long().unsqueeze(1)   # [B,1]
        q_i = Q_i.gather(1, a_i)                  # [B,1]
        q_sum = q_sum + q_i
    return q_sum.squeeze(1)    


def save_training_log(log_file, iteration, reward):
    """
    Ghi 1 dòng dữ liệu (iteration, reward) vào file CSV log.
    Tự động tạo folder nếu chưa có.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([iteration, reward])

def plot_reward_curve(rewards, fig_file, title="Training Reward Curve"):
    """
    Vẽ và lưu biểu đồ reward theo iteration.
    Dùng chung cho PPO, BDQ, ARQ.
    """
    os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Reward")
    plt.title(title)
    plt.grid(True)
    plt.savefig(fig_file, dpi=160)
    plt.close()

def ensure_dir(path):
    """
    Đảm bảo thư mục tồn tại, nếu không thì tạo mới.
    """
    os.makedirs(path, exist_ok=True)

def count_parameters(model):
    """
    Đếm tổng số tham số trainable trong mạng.
    Hữu ích để theo dõi độ phức tạp mô hình.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 🆕 Hàm bổ sung để quản lý thư mục kết quả
def make_result_dirs(algorithm_name):
    """
    Tạo cấu trúc thư mục lưu kết quả cho từng thuật toán.
    results/
        └── <algorithm_name>/
            ├── figures/
            ├── logs/
            └── models/
    Trả về dict chứa đường dẫn 3 thư mục con.
    """
    base_dir = os.path.join("results", algorithm_name)
    sub_dirs = ["figures", "logs", "models"]
    for sub in sub_dirs:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)
    return {
        "figures": os.path.join(base_dir, "figures"),
        "logs": os.path.join(base_dir, "logs"),
        "models": os.path.join(base_dir, "models")
    }
