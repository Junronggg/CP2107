import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

def load_tb_data(path):
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    
    # Get wall clock, number of steps, and value for scalar 'Train_AverageReturn'
    try:
        events = event_acc.Scalars('Train_AverageReturn')
        steps = list(map(lambda x: x.step, events))
        values = list(map(lambda x: x.value, events))
        return steps, values
    except:
        return None, None

def plot_learning_curves(data_dir, exp_names, title, output_file):
    plt.figure(figsize=(10, 6))
    
    for exp_name in exp_names:
        # Find the most recent matching experiment directory
        exp_dirs = glob.glob(os.path.join(data_dir, f"q2_pg_{exp_name}_CartPole-v0*"))
        if not exp_dirs:
            print(f"No data found for experiment {exp_name}")
            continue
            
        latest_dir = max(exp_dirs, key=os.path.getctime)
        event_files = glob.glob(os.path.join(latest_dir, "events.out.tfevents.*"))
        
        if not event_files:
            print(f"No event files found in {latest_dir}")
            continue
            
        steps, values = load_tb_data(latest_dir)
        if steps is None:
            continue
            
        plt.plot(steps, values, label=exp_name)
    
    plt.xlabel('Number of Environment Steps')
    plt.ylabel('Average Return')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

# Directory containing the experiment data
data_dir = "homework_fall2023/hw2/data"

# Small batch experiments
small_batch_exps = ['sb_no_rtg_nb', 'sb_rtg_nb']
plot_learning_curves(
    data_dir,
    small_batch_exps,
    'Learning Curves (Small Batch, b=1000)',
    'small_batch_comparison.png'
)

print("Plots have been saved as small_batch_comparison.png") 