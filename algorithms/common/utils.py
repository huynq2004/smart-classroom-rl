import torch

def hard_update(target_net, online_net):
    target_net.load_state_dict(online_net.state_dict())

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
    return q_sum.squeeze(1)                        # [B]
