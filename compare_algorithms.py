import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_training_logs():
    """Load training logs từ 3 thuật toán: PPO, BDQ, NoisyNet"""
    logs = {}
    
    # BDQ
    if os.path.exists("results/bdq/logs/bdq_training_log.csv"):
        logs['BDQ'] = pd.read_csv("results/bdq/logs/bdq_training_log.csv")
    
    # PPO
    if os.path.exists("results/ppo/logs/ppo_training_log.csv"):
        logs['PPO'] = pd.read_csv("results/ppo/logs/ppo_training_log.csv")
    
    # NoisyNet
    if os.path.exists("results/noisynet/logs/noisynet_training_log.csv"):
        logs['NoisyNet'] = pd.read_csv("results/noisynet/logs/noisynet_training_log.csv")
    
    return logs

def analyze_performance(logs):
    """Phân tích performance của các thuật toán"""
    print("=== SO SÁNH PERFORMANCE CÁC THUẬT TOÁN ===\n")
    
    for algo_name, df in logs.items():
        print(f"📊 {algo_name}:")
        print(f"   - Số episodes: {len(df)}")
        print(f"   - Reward trung bình: {df['reward'].mean():.2f}")
        print(f"   - Reward tốt nhất: {df['reward'].max():.2f}")
        print(f"   - Reward cuối cùng: {df['reward'].iloc[-1]:.2f}")
        
        # Epsilon cuối cùng (chỉ có cho BDQ)
        if 'epsilon' in df.columns:
            final_eps = df['epsilon'].iloc[-1]
            print(f"   - Epsilon cuối: {final_eps:.3f}")
        
        # Evaluation scores (nếu có)
        if 'eval_mean' in df.columns:
            eval_data = df.dropna(subset=['eval_mean'])
            if len(eval_data) > 0:
                print(f"   - Eval score tốt nhất: {eval_data['eval_mean'].max():.2f}")
                print(f"   - Eval score cuối: {eval_data['eval_mean'].iloc[-1]:.2f}")
        
        # PPO specific metrics
        if algo_name == 'PPO' and 'policy_loss' in df.columns:
            policy_loss_data = df.dropna(subset=['policy_loss'])
            if len(policy_loss_data) > 0:
                print(f"   - Policy Loss cuối: {policy_loss_data['policy_loss'].iloc[-1]:.4f}")
                print(f"   - Value Loss cuối: {policy_loss_data['value_loss'].iloc[-1]:.4f}")
        
        print()

def plot_comparison(logs):
    """Vẽ đồ thị so sánh"""
    plt.figure(figsize=(20, 12))
    
    # 1. Training Rewards
    plt.subplot(2, 3, 1)
    for algo_name, df in logs.items():
        plt.plot(df['episode'], df['reward'], label=f'{algo_name}', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Training Reward')
    plt.title('Training Rewards Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Epsilon Decay (chỉ cho BDQ)
    plt.subplot(2, 3, 2)
    for algo_name, df in logs.items():
        if 'epsilon' in df.columns:
            plt.plot(df['episode'], df['epsilon'], label=f'{algo_name}', alpha=0.7)
        else:
            # Cho PPO và NoisyNet, vẽ một đường ngang để so sánh
            plt.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label=f'{algo_name} (no epsilon)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Evaluation Scores (nếu có)
    plt.subplot(2, 3, 3)
    for algo_name, df in logs.items():
        eval_data = df.dropna(subset=['eval_mean'])
        if len(eval_data) > 0:
            plt.plot(eval_data['episode'], eval_data['eval_mean'], 
                    label=f'{algo_name}', marker='o', markersize=3)
    plt.xlabel('Episode')
    plt.ylabel('Evaluation Score')
    plt.title('Evaluation Scores Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Moving Average Rewards
    plt.subplot(2, 3, 4)
    window = 20
    for algo_name, df in logs.items():
        moving_avg = df['reward'].rolling(window=window).mean()
        plt.plot(df['episode'], moving_avg, label=f'{algo_name} (MA-{window})', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    plt.title(f'Moving Average Rewards (Window={window})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. PPO Losses (nếu có)
    plt.subplot(2, 3, 5)
    if 'PPO' in logs and 'policy_loss' in logs['PPO'].columns:
        ppo_df = logs['PPO']
        policy_loss_data = ppo_df.dropna(subset=['policy_loss'])
        if len(policy_loss_data) > 0:
            plt.plot(policy_loss_data['episode'], policy_loss_data['policy_loss'], 
                    label='Policy Loss', alpha=0.7)
            plt.plot(policy_loss_data['episode'], policy_loss_data['value_loss'], 
                    label='Value Loss', alpha=0.7)
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.title('PPO Losses')
            plt.legend()
            plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No PPO Loss Data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('PPO Losses (No Data)')
    
    # 6. Algorithm Comparison Summary
    plt.subplot(2, 3, 6)
    algorithms = list(logs.keys())
    final_rewards = [logs[algo]['reward'].iloc[-1] for algo in algorithms]
    colors = ['blue', 'red', 'green', 'orange', 'purple'][:len(algorithms)]
    
    bars = plt.bar(algorithms, final_rewards, color=colors, alpha=0.7)
    plt.ylabel('Final Reward')
    plt.title('Final Rewards Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, final_rewards):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def detailed_analysis(logs):
    """Phân tích chi tiết"""
    print("\n=== PHÂN TÍCH CHI TIẾT ===\n")
    
    # Tạo DataFrame tổng hợp
    summary_data = []
    for algo_name, df in logs.items():
        data = {
            'Algorithm': algo_name,
            'Episodes': len(df),
            'Final_Reward': df['reward'].iloc[-1],
            'Best_Reward': df['reward'].max(),
            'Avg_Reward': df['reward'].mean(),
            'Convergence_Episode': find_convergence_episode(df)
        }
        
        # Thêm epsilon nếu có
        if 'epsilon' in df.columns:
            data['Final_Epsilon'] = df['epsilon'].iloc[-1]
        else:
            data['Final_Epsilon'] = 'N/A'
            
        summary_data.append(data)
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Ranking
    print(f"\n🏆 RANKING:")
    print(f"1. Best Final Reward: {summary_df.loc[summary_df['Final_Reward'].idxmax(), 'Algorithm']}")
    print(f"2. Best Average Reward: {summary_df.loc[summary_df['Avg_Reward'].idxmax(), 'Algorithm']}")
    print(f"3. Most Stable: {summary_df.loc[summary_df['Convergence_Episode'].idxmin(), 'Algorithm']}")
    
    # PPO specific analysis
    if 'PPO' in logs and 'policy_loss' in logs['PPO'].columns:
        print(f"\n📊 PPO SPECIFIC METRICS:")
        ppo_df = logs['PPO']
        policy_loss_data = ppo_df.dropna(subset=['policy_loss'])
        if len(policy_loss_data) > 0:
            print(f"   - Final Policy Loss: {policy_loss_data['policy_loss'].iloc[-1]:.4f}")
            print(f"   - Final Value Loss: {policy_loss_data['value_loss'].iloc[-1]:.4f}")
            print(f"   - Final Entropy: {policy_loss_data['entropy'].iloc[-1]:.4f}")
            print(f"   - Avg Policy Loss: {policy_loss_data['policy_loss'].mean():.4f}")
            print(f"   - Avg Value Loss: {policy_loss_data['value_loss'].mean():.4f}")

def find_convergence_episode(df, window=50, threshold=0.1):
    """Tìm episode mà reward đã hội tụ"""
    if len(df) < window:
        return len(df)
    
    moving_avg = df['reward'].rolling(window=window).mean()
    for i in range(window, len(df)):
        recent_avg = moving_avg.iloc[i-window:i].mean()
        current_avg = moving_avg.iloc[i]
        if abs(current_avg - recent_avg) < threshold:
            return i
    return len(df)

def main():
    """Main function"""
    print("🔍 Đang load training logs...")
    logs = load_training_logs()
    
    if not logs:
        print("❌ Không tìm thấy log files!")
        return
    
    print(f"✅ Đã load {len(logs)} thuật toán: {list(logs.keys())}")
    
    # Phân tích performance
    analyze_performance(logs)
    
    # Phân tích chi tiết
    detailed_analysis(logs)
    
    # Vẽ đồ thị
    print("\n📈 Đang vẽ đồ thị so sánh...")
    plot_comparison(logs)
    print("✅ Đã lưu đồ thị: algorithm_comparison.png")

if __name__ == "__main__":
    main()

