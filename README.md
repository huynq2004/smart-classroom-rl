# 📘 Smart Room RL – Reinforcement Learning cho Phòng Học Thông Minh

## 1. Giới thiệu
Dự án này mô phỏng và huấn luyện các thuật toán **học tăng cường (RL)** để điều khiển **nhiều thiết bị trong phòng học thông minh** (đèn, quạt, điều hòa, rèm cửa, …).  
Mục tiêu: cân bằng **thoải mái (comfort)** và **tiết kiệm năng lượng (energy saving)**, đồng thời giảm **switching cost** (số lần bật/tắt thiết bị).

- **Môi trường (Environment):** trạng thái liên tục (nhiệt độ, độ sáng, số người, nhiệt độ ngoài trời, trạng thái thiết bị).  
- **Hành động (Action):** rời rạc đa chiều (multi-discrete), ví dụ: quạt {0,1,2}, điều hòa {0,1,2}, đèn {0,1}.  
- **Reward:** kết hợp ba yếu tố (năng lượng, comfort, switching cost).  

## 2. Thuật toán triển khai
Trong dự án, nhóm thử nghiệm **3 thuật toán**:
1. **BDQ-lite (Factored Double Dueling DQN)** – tách hành động thành nhiều nhánh, phù hợp với action space lớn.  
2. **Multi-head DQN** – baseline, mỗi head dự đoán một nhánh hành động.  
3. **Auto-Regressive Q** – chọn hành động tuần tự theo từng thiết bị.  

## 3. Cấu trúc dự án

```plaintext
smart-room-rl/
│── README.md               # Tài liệu mô tả (file này)
│── requirements.txt        # Thư viện cần cài (torch, gymnasium, numpy,…)
│── main.py                 # Điểm khởi chạy, gọi thử nghiệm
│── config.py               # Tham số chung (gamma, epsilon, batch_size, …)

├── envs/                   # Môi trường phòng học
│   ├── smart_room_env.py   # Định nghĩa MDP: state, action, reward, step
│   └── utils.py            # Hàm phụ (tính comfort, energy, lux,...)

├── algorithms/             # Cài đặt các thuật toán
│   ├── bdq_lite.py         # BDQ-lite agent
│   ├── multi_head_dqn.py   # Multi-head DQN agent
│   ├── ar_q.py             # Auto-Regressive Q agent
│   └── common/
│       ├── networks.py     # Kiến trúc mạng (trunk, dueling head, multi-head)
│       ├── replay_buffer.py# Bộ nhớ lưu kinh nghiệm
│       └── utils.py        # Soft update target net, epsilon decay

├── experiments/            # Thí nghiệm
│   ├── run_bdq.py          # Script huấn luyện BDQ-lite
│   ├── run_multi_head.py   # Script huấn luyện Multi-head DQN
│   ├── run_arq.py          # Script huấn luyện Auto-Regressive Q
│   └── configs/            # Hyperparams cho từng thuật toán (.json/.yaml)

├── results/                # Kết quả huấn luyện
│   ├── logs/               # Reward theo episode, loss
│   ├── models/             # Trọng số mạng (θ, θ_target) đã train
│   └── figures/            # Biểu đồ reward, comfort, energy

└── report/                 # Báo cáo + slide nhóm
    ├── report.docx
    ├── presentation.pptx
    └── images/             # Sơ đồ, hình minh họa
```

## 4. Luồng hoạt động
1. **Chạy thí nghiệm:**  
   ```bash
   python experiments/run_bdq.py --config experiments/configs/bdq.json
   ```
   → Khởi tạo môi trường + agent BDQ → train qua nhiều episodes.

2. **Mỗi episode:**  
   - `env.reset()` → trạng thái ban đầu.  
   - Agent chọn action (online Q hoặc random).  
   - `env.step(action)` → trạng thái mới + reward.  
   - Transition lưu vào ReplayBuffer.  
   - Agent update mạng Q (tính Q_pred, Q_target).  

3. **Sau huấn luyện:**  
   - Kết quả (reward curve, energy/comfort metrics) được lưu trong `results/figures/`.  
   - Model đã train lưu trong `results/models/`.  
   - Biểu đồ đưa vào `report/presentation.pptx`.  
 
