# algorithms/ar_q.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.common.networks import BDQNet
from algorithms.common.replay_buffer import ReplayBuffer
from algorithms.common.utils import soft_update, epsilon_decay

class ARQAgent:
    """
    Auto-Regressive Q agent.
    - Uses BDQNet architecture but with augmented state: [state | prefix_onehot]
    - Selection: selects actions sequentially for each branch, conditioning on prefix.
    """
    def __init__(self, state_dim, action_dims, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_dims = list(action_dims)
        self.n_branches = len(self.action_dims)
        # prefix dimension = sum of choices lengths (one-hot per branch)
        self.prefix_dim = sum(self.action_dims)
        self.state_dim = state_dim
        self.state_dim_aug = state_dim + self.prefix_dim

        # network: trunk expects state_dim_aug
        self.q_net = BDQNet(self.state_dim_aug, self.action_dims).to(self.device)
        self.target_net = BDQNet(self.state_dim_aug, self.action_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config["lr"])
        self.buffer = ReplayBuffer(config["buffer_size"])

        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        self.tau = config.get("tau", 0.01)

        self.eps = config["epsilon_start"]
        self.eps_end = config["epsilon_end"]
        self.eps_decay = config["epsilon_decay"]

        # offsets for placing one-hot blocks in prefix vector
        self.offsets = np.cumsum([0] + self.action_dims[:-1])

    # ---------- prefix helpers ----------
    def _encode_prefix(self, prefix_actions):
        """
        prefix_actions: list or array of chosen ints for first k branches (k <= n_branches)
        returns: 1D numpy array length prefix_dim with ones at chosen positions
        """
        arr = np.zeros(self.prefix_dim, dtype=np.float32)
        for i, a in enumerate(prefix_actions):
            if a is None:
                continue
            off = self.offsets[i]
            arr[off + int(a)] = 1.0
        return arr

    def _state_with_prefix(self, state, prefix_actions):
        # state: 1D array; prefix_actions: list of chosen ints for first k branches
        p = self._encode_prefix(prefix_actions)
        return np.concatenate([np.array(state, dtype=np.float32), p], axis=0)

    # ---------- action selection (sequential) ----------
    def select_action(self, state, greedy=False):
        """
        Build full action tuple sequentially.
        greedy=True => always argmax (for eval)
        """
        prefix = []
        for i in range(self.n_branches):
            state_aug = self._state_with_prefix(state, prefix)
            state_t = torch.as_tensor(state_aug, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                Q_heads = self.q_net(state_t)
            Q_i = Q_heads[i]  # [1, n_i]
            if (not greedy) and (np.random.rand() < self.eps):
                choice = int(np.random.randint(0, self.action_dims[i]))
            else:
                choice = int(Q_i.argmax(dim=1).item())
            prefix.append(choice)
        return prefix

    # ---------- compute Q-sum for a given tuple (sequentially) ----------
    def _q_sum_for_tuple(self, network, state, action_tuple):
        """Compute Q(s,a) = sum_i Q_i(s, prefix_{i}, a_i) using given network (online or target)."""
        total = 0.0
        for i in range(self.n_branches):
            prefix = action_tuple[:i]  # actions chosen before branch i
            state_aug = self._state_with_prefix(state, prefix)
            state_t = torch.as_tensor(state_aug, dtype=torch.float32, device=self.device).unsqueeze(0)
            Q_heads = network(state_t)
            q_i = Q_heads[i][0, int(action_tuple[i])]   # scalar tensor
            total = total + q_i
        return total.squeeze()  # tensor scalar

    # ---------- update ----------
    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}

        s_batch, a_batch, r_batch, s_next_batch, done_batch = self.buffer.sample(self.batch_size)
        B = len(s_batch)

        # convert to types
        r = torch.as_tensor(r_batch, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done_batch, dtype=torch.float32, device=self.device)

        # compute Q_pred (sum) for each sample by evaluating sequentially with online network
        Q_preds = []
        for j in range(B):
            q_pred = self._q_sum_for_tuple(self.q_net, s_batch[j], a_batch[j])  # tensor
            Q_preds.append(q_pred)
        Q_pred = torch.stack(Q_preds).to(self.device).float().squeeze()  # [B]

        # compute a_star (greedy sequence) on s_next using online net
        a_star_batch = []
        for j in range(B):
            s_next = s_next_batch[j]
            prefix = []
            for i in range(self.n_branches):
                s_aug = self._state_with_prefix(s_next, prefix)
                s_t = torch.as_tensor(s_aug, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    Q_heads_online = self.q_net(s_t)
                pick = int(Q_heads_online[i].argmax(dim=1).item())
                prefix.append(pick)
            a_star_batch.append(prefix)

        # compute Q_tgt_sum for s_next and a_star using target_net
        Q_tgts = []
        for j in range(B):
            q_tgt = self._q_sum_for_tuple(self.target_net, s_next_batch[j], a_star_batch[j])
            Q_tgts.append(q_tgt)
        Q_tgt = torch.stack(Q_tgts).to(self.device).float().squeeze()  # [B]

        y = r + self.gamma * (1.0 - done) * Q_tgt

        loss = nn.MSELoss()(Q_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.target_net, self.q_net, self.tau)
        self.eps = epsilon_decay(self.eps, self.eps_end, self.eps_decay)

        return {"loss": float(loss.item()), "eps": float(self.eps)}

    # ---------- train episode ----------
    def train_episode(self, env):
        s, _ = env.reset()
        done = False
        total_r = 0.0
        while not done:
            a = self.select_action(s)
            s_next, r, term, trunc, info = env.step(a)
            done = bool(term or trunc)
            self.buffer.push(s, a, r, s_next, done)
            stats = self.update()
            s = s_next
            total_r += float(r)
        return total_r
