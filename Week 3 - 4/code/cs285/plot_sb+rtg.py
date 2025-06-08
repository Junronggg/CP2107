import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalar(log_dir, tag='Eval_AverageReturn'):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    if tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        return steps, values
    return None, None

def aggregate_runs(log_dirs, tag='Eval_AverageReturn'):
    all_returns = []
    all_steps = []
    for d in log_dirs:
        steps, values = extract_scalar(d, tag)
        if steps is not None:
            all_returns.append(values)
            all_steps.append(steps)
    min_len = min(len(r) for r in all_returns)
    all_returns = np.array([r[:min_len] for r in all_returns])
    all_steps = all_steps[0][:min_len]  # Assume same steps
    mean = np.mean(all_returns, axis=0)
    std = np.std(all_returns, axis=0)
    return all_steps, mean, std

# Example: Customize these with your actual folder names
experiment_groups = {
    'Small batch without reward-to-go': [
        'D:\work\CP2107\homework_fall2023\hw2\data\q2_pg_cartpole_seed1_CartPole-v0_08-06-2025_17-27-17',
        'D:\work\CP2107\homework_fall2023\hw2\data\q2_pg_cartpole_seed5_CartPole-v0_08-06-2025_17-46-39',
        'D:\work\CP2107\homework_fall2023\hw2\data\q2_pg_cartpole_seed10_CartPole-v0_08-06-2025_18-28-05',
        'D:\work\CP2107\homework_fall2023\hw2\data\q2_pg_cartpole_seed30_CartPole-v0_08-06-2025_22-25-01'
    ],
    'Small batch with reward-to-go': [
        'D:\work\CP2107\homework_fall2023\hw2\data\q2_pg_cartpole_rtg_seed1_CartPole-v0_08-06-2025_17-29-24',
        'D:\work\CP2107\homework_fall2023\hw2\data\q2_pg_cartpole_rtg_seed5_CartPole-v0_08-06-2025_17-53-35',
        'D:\work\CP2107\homework_fall2023\hw2\data\q2_pg_cartpole_rtg_seed10_CartPole-v0_08-06-2025_18-29-52',
        'D:\work\CP2107\homework_fall2023\hw2\data\q2_pg_cartpole_rtg_seed30_CartPole-v0_08-06-2025_22-26-24'
    ]    
}

colors = {
    'Small batch without reward-to-go': "#8EEFE0",  # light green-blue
    'Small batch with reward-to-go': "#61BF58",     # light green
}

# make surve smoother
def smooth_curve(y, smooth_width=4):
    """Apply simple moving average to smooth the curve."""
    box = np.ones(smooth_width) / smooth_width
    return np.convolve(y, box, mode='same')

# Plot
plt.figure(figsize=(10, 6))
for label, dirs in experiment_groups.items():
    steps, mean, std = aggregate_runs(dirs)
    mean_smooth = smooth_curve(mean, smooth_width=5)
    std_smooth = smooth_curve(std, smooth_width=5)
    c = colors[label]
    plt.plot(steps, mean_smooth, label=label, color=c, linewidth=4.0)
    plt.fill_between(steps, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3, color=c)

plt.title('Effect of Reward-to-go (rtg) on Small Batch Data')
plt.xlabel('Training Iteration')
plt.ylabel('Average Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cartpole_comparison_sb_rtg.png")
plt.show()
