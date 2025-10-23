# Multi-Head DQN Algorithm Flow - Smart Classroom Project

## Tổng quan dự án

Dự án này triển khai thuật toán **Multi-Head DQN** để điều khiển phòng học thông minh, tối ưu hóa năng lượng và tiện nghi người dùng.

## Luồng thực thi chính

### 1. Entry Point - `main.py`

```bash
python main.py --algo multihead --config experiments/configs/multihead.json
```

**Mục đích**: Điểm khởi đầu của toàn bộ hệ thống
- Nhận tham số thuật toán (`--algo`) và cấu hình (`--config`)
- Gọi script tương ứng dựa trên thuật toán được chọn
- Cho Multi-Head DQN: chuyển điều khiển đến `experiments/run_multi_head.py`

### 2. Training Script - `experiments/run_multi_head.py`

**Mục đích**: Quản lý quá trình huấn luyện Multi-Head DQN

**Các bước chính**:
1. **Load cấu hình** từ `experiments/configs/multihead.json`
2. **Khởi tạo môi trường** `SmartRoomEnv`
3. **Tạo agent** `MultiHeadDQNAgent`
4. **Vòng lặp huấn luyện** cho N episodes
5. **Đánh giá và lưu model** định kỳ

**Files được import**:
- `config.py`: Quản lý cấu hình và seed
- `envs.SmartRoomEnv`: Môi trường phòng học thông minh
- `algorithms.multi_head_dqn.MultiHeadDQNAgent`: Agent Multi-Head DQN

### 3. Agent Implementation - `algorithms/multi_head_dqn.py`

**Mục đích**: Triển khai thuật toán Multi-Head DQN

**Cấu trúc chính**:
- `MultiHeadNet`: Neural network với trunk chung + multiple heads
- `MultiHeadDQNAgent`: Agent chính thực hiện thuật toán

## Chi tiết thuật toán Multi-Head DQN

### Kiến trúc Neural Network

```python
class MultiHeadNet(nn.Module):
    def __init__(self, state_dim, action_dims):
        # Trunk chung: state -> shared features
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        # Multiple heads: shared features -> Q-values cho từng action dimension
        self.heads = nn.ModuleList([nn.Linear(256, n) for n in action_dims])
```

**Mục đích**:
- **Trunk**: Chia sẻ feature extraction cho tất cả heads
- **Heads**: Mỗi head dự đoán Q-values cho một dimension của action space

### Action Space trong Smart Classroom

```python
# Action space: MultiDiscrete [3,3,3, 3,3, 2,2]
# Fan1, Fan2, Fan3, AC1, AC2, Light1, Light2
action_dims = [3, 3, 3, 3, 3, 2, 2]
```

**Ý nghĩa**:
- **Fan1-3**: 3 mức độ (Tắt, Chậm, Nhanh)
- **AC1-2**: 3 mức độ (Tắt, Nhẹ, Mạnh)  
- **Light1-2**: 2 mức độ (Tắt, Bật)

### State Space

```python
# State: [T, L, N, T_out] + [D1, D2, D3, D4, D5, D6, D7]
# T: Nhiệt độ trong phòng
# L: Độ sáng trong phòng
# N: Số người
# T_out: Nhiệt độ ngoài trời
# D1-D7: Trạng thái thiết bị
```

## Luồng thuật toán chi tiết

### Bước 1: Khởi tạo (Initialization)

```python
def __init__(self, state_dim, action_dims, config):
    # Khởi tạo mạng Q_online và Q_target
    self.q_net = MultiHeadNet(state_dim, action_dims)
    self.target_net = MultiHeadNet(state_dim, action_dims)
    
    # Bộ nhớ lặp lại D
    self.buffer = ReplayBuffer(config["buffer_size"])
    
    # Tham số thuật toán
    self.gamma = config["gamma"]  # Discount factor
    self.eps = config["epsilon_start"]  # Exploration rate
```

### Bước 2: Chọn hành động (Action Selection)

```python
def select_action(self, state, greedy=False):
    if (not greedy) and (np.random.rand() < self.eps):
        # Khám phá: chọn ngẫu nhiên cho từng head
        return [int(np.random.randint(0, n)) for n in self.action_dims]
    
    # Khai thác: chọn tối ưu cho từng head
    state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
    with torch.no_grad():
        Q_heads = self.q_net(state_t)
    actions = [int(Q.argmax(dim=1).item()) for Q in Q_heads]
    return actions
```

**Tương ứng với mã giả**:
- Với xác suất ε → chọn hành động ngẫu nhiên cho từng head
- Ngược lại → tính feature chung = trunk(s), sau đó chọn argmax cho từng head

### Bước 3: Huấn luyện một Episode

```python
def train_episode(self, env):
    s, _ = env.reset()  # Khởi tạo trạng thái ban đầu
    done = False
    total_r = 0.0
    
    while not done:
        # 1. Chọn hành động
        a = self.select_action(s)
        
        # 2. Thực hiện hành động trong môi trường
        s_next, r, terminated, truncated, info = env.step(a)
        done = bool(terminated or truncated)
        
        # 3. Lưu vào bộ nhớ D
        self.buffer.push(s, a, r, s_next, done)
        
        # 4. Huấn luyện mạng Q
        stats = self.update()
        
        s = s_next
        total_r += float(r)
    return total_r
```

### Bước 4: Cập nhật mạng Q (Q-network Update)

```python
def update(self):
    if len(self.buffer) < self.batch_size:
        return {}
    
    # Lấy minibatch từ D
    s_batch, a_batch, r_batch, s_next_batch, done_batch = self.buffer.sample(self.batch_size)
    
    # Tính Q_pred(s,a) = Σ_i Q_i(s, a_i)
    Q_heads = self.q_net(s)
    Q_pred = compute_q_sum(Q_heads, a)
    
    # Double DQN target
    with torch.no_grad():
        # Chọn action tối ưu từ online net
        Q_heads_next_online = self.q_net(s_next)
        a_star = torch.stack([Q.argmax(dim=1) for Q in Q_heads_next_online], dim=1)
        
        # Tính Q-target từ target net
        Q_heads_next_target = self.target_net(s_next)
        Q_target_sum = compute_q_sum(Q_heads_next_target, a_star)
        y = r + self.gamma * (1.0 - done) * Q_target_sum
    
    # Tính loss và cập nhật
    loss = nn.MSELoss()(Q_pred, y)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    # Soft update target network
    soft_update(self.target_net, self.q_net, self.tau)
    
    # Decay epsilon
    self.eps = epsilon_decay(self.eps, self.eps_end, self.eps_decay)
```

**Tương ứng với mã giả**:
- Lấy minibatch từ D: (sj, aj, rj, s′j)
- Với mỗi head i: tính Q_target dựa trên Double DQN
- Tính loss = MSE(Q_pred, Q_target)
- Cập nhật θ bằng gradient descent
- Soft update θ⁻ ← θ

### Bước 5: Tính toán Reward trong Smart Classroom

```python
def compute_reward(T_next, L_next, N, D_curr, D_next, energy_kwh, ...):
    # Tiết kiệm năng lượng (âm)
    reward_energy = -c_energy * energy_kwh
    
    # Tiện nghi nhiệt độ
    if N > 0:  # Có người trong phòng
        discomfort_temp = max(0, abs(T_next - T_target) - delta_T)
        reward_temp = -c_temp * discomfort_temp
    else:
        reward_temp = 0
    
    # Tiện nghi ánh sáng  
    if N > 0:
        discomfort_light = max(0, abs(L_next - L_target) / L_target)
        reward_light = -c_light * discomfort_light
    else:
        reward_light = 0
    
    # Chi phí bật/tắt thiết bị
    switch_count = np.sum(D_curr != D_next)
    reward_switch = -c_switch * switch_count
    
    total_reward = reward_energy + reward_temp + reward_light + reward_switch
    return total_reward, discomfort_temp, discomfort_light, switch_count
```

## Files và mục đích của chúng

### Core Algorithm Files
- **`algorithms/multi_head_dqn.py`**: Triển khai thuật toán Multi-Head DQN
- **`algorithms/common/networks.py`**: Neural network architectures chung
- **`algorithms/common/replay_buffer.py`**: Bộ nhớ lặp lại cho DQN
- **`algorithms/common/utils.py`**: Utility functions (soft update, epsilon decay, Q-sum)

### Environment Files  
- **`envs/smart_room_env.py`**: Môi trường phòng học thông minh
- **`envs/utils.py`**: Utility functions cho môi trường (tính nhiệt độ, ánh sáng, reward)

### Configuration Files
- **`config.py`**: Load config, set seed, tạo directories
- **`experiments/configs/multihead.json`**: Cấu hình tham số cho Multi-Head DQN

### Training Scripts
- **`experiments/run_multi_head.py`**: Script huấn luyện Multi-Head DQN
- **`main.py`**: Entry point chọn thuật toán

## Tham số quan trọng

### Từ config file (`multihead.json`):
```json
{
  "algo": {
    "buffer_size": 50000,      // Kích thước bộ nhớ lặp lại N
    "batch_size": 64,          // Kích thước minibatch
    "gamma": 0.99,             // Discount factor γ
    "lr": 0.001,               // Learning rate
    "epsilon_start": 0.20,     // Tỷ lệ khám phá ban đầu ε
    "epsilon_end": 0.05,       // Tỷ lệ khám phá cuối
    "epsilon_decay": 0.00005,  // Tốc độ giảm epsilon
    "tau": 0.01                // Soft update rate cho target network
  }
}
```

### Môi trường:
```json
{
  "env": {
    "dt_minutes": 15,          // Khoảng thời gian mỗi step (phút)
    "scenario_csv": null       // File scenario (nếu có)
  }
}
```

## Kết quả đầu ra

### Files được tạo:
- **`results/models/`**: Các checkpoint của model
  - `multihead_checkpoint_X.pt`: Checkpoint mỗi 20 episodes
  - `multihead_best.pt`: Model tốt nhất
  - `multihead_latest.pt`: Model cuối cùng

- **`results/logs/`**: Logs huấn luyện
  - `multihead_training_log.csv`: Reward, epsilon, evaluation metrics

- **`results/figures/`**: Biểu đồ
  - `multihead_reward_curve.png`: Đường cong reward theo episodes

## Ưu điểm của Multi-Head DQN trong Smart Classroom

1. **Hiệu quả**: Chia sẻ feature extraction giữa các heads
2. **Linh hoạt**: Mỗi thiết bị có policy riêng nhưng học từ shared features  
3. **Ổn định**: Double DQN + Target network giảm overestimation
4. **Thực tế**: Phù hợp với action space phức tạp của phòng học thông minh

Thuật toán này tối ưu hóa việc điều khiển 7 thiết bị (3 fan, 2 AC, 2 đèn) để cân bằng giữa tiết kiệm năng lượng và tiện nghi người dùng trong phòng học thông minh.

