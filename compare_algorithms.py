import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_training_logs():
    """Load training logs từ 3 thuật toán"""
    logs = {}
    
    # BDQ
    if os.path.exists("results/bdq/logs/bdq_training_log.csv"):
        logs['BDQ'] = pd.read_csv("results/bdq/logs/bdq_training_log.csv")
    
    # MultiHead
    if os.path.exists("results/multihead/logs/multihead_training_log.csv"):
        logs['MultiHead'] = pd.read_csv("results/multihead/logs/multihead_training_log.csv")
    
    # AR-Q
    if os.path.exists("results/arq/logs/arq_training_log.csv"):
        logs['ARQ'] = pd.read_csv("results/arq/logs/arq_training_log.csv")
    
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
        
        # Epsilon cuối cùng
        final_eps = df['epsilon'].iloc[-1]
        print(f"   - Epsilon cuối: {final_eps:.3f}")
        
        # Evaluation scores (nếu có)
        eval_data = df.dropna(subset=['eval_mean'])
        if len(eval_data) > 0:
            print(f"   - Eval score tốt nhất: {eval_data['eval_mean'].max():.2f}")
            print(f"   - Eval score cuối: {eval_data['eval_mean'].iloc[-1]:.2f}")
        
        print()

def plot_comparison(logs):
    """Vẽ đồ thị so sánh"""
    plt.figure(figsize=(15, 10))
    
    # 1. Training Rewards
    plt.subplot(2, 2, 1)
    for algo_name, df in logs.items():
        plt.plot(df['episode'], df['reward'], label=f'{algo_name}', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Training Reward')
    plt.title('Training Rewards Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Epsilon Decay
    plt.subplot(2, 2, 2)
    for algo_name, df in logs.items():
        plt.plot(df['episode'], df['epsilon'], label=f'{algo_name}', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Evaluation Scores (nếu có)
    plt.subplot(2, 2, 3)
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
    plt.subplot(2, 2, 4)
    window = 20
    for algo_name, df in logs.items():
        moving_avg = df['reward'].rolling(window=window).mean()
        plt.plot(df['episode'], moving_avg, label=f'{algo_name} (MA-{window})', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    plt.title(f'Moving Average Rewards (Window={window})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def detailed_analysis(logs):
    """Phân tích chi tiết"""
    print("\n=== PHÂN TÍCH CHI TIẾT ===\n")
    
    # Tạo DataFrame tổng hợp
    summary_data = []
    for algo_name, df in logs.items():
        summary_data.append({
            'Algorithm': algo_name,
            'Episodes': len(df),
            'Final_Reward': df['reward'].iloc[-1],
            'Best_Reward': df['reward'].max(),
            'Avg_Reward': df['reward'].mean(),
            'Final_Epsilon': df['epsilon'].iloc[-1],
            'Convergence_Episode': find_convergence_episode(df)
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Ranking
    print(f"\n🏆 RANKING:")
    print(f"1. Best Final Reward: {summary_df.loc[summary_df['Final_Reward'].idxmax(), 'Algorithm']}")
    print(f"2. Best Average Reward: {summary_df.loc[summary_df['Avg_Reward'].idxmax(), 'Algorithm']}")
    print(f"3. Most Stable: {summary_df.loc[summary_df['Convergence_Episode'].idxmin(), 'Algorithm']}")

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

