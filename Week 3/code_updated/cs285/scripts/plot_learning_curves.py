import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import json

def load_data(log_dir):
    """Load data from experiment logs."""
    data = {}
    print(f"Searching for data in: {log_dir}")
    for exp_dir in glob.glob(os.path.join(log_dir, "q2_pg_*CartPole-v1*")):
        print(f"Found experiment directory: {exp_dir}")
        exp_name = os.path.basename(exp_dir)
        
        # Load metrics.json
        metrics_file = os.path.join(exp_dir, "metrics.json")
        if not os.path.exists(metrics_file):
            print(f"No metrics.json found in {exp_dir}")
            continue
            
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Extract learning curve data
        steps = []
        returns = []
        for metric in metrics:
            if "Train_EnvstepsSoFar" in metric:
                steps.append(metric["Train_EnvstepsSoFar"])
            if "Eval_AverageReturn" in metric:
                returns.append(metric["Eval_AverageReturn"])
                
        if steps and returns:
            # Clean up the experiment name for the legend
            clean_name = exp_name.replace('q2_pg_', '').replace('_CartPole-v1', '').split('_')[0]
            data[clean_name] = {
                "steps": np.array(steps),
                "returns": np.array(returns)
            }
            print(f"Loaded data for experiment: {clean_name}")
    
    return data

def plot_learning_curves(data, save_path=None):
    """Plot learning curves for all experiments."""
    plt.figure(figsize=(10, 6))
    
    for exp_name, exp_data in data.items():
        plt.plot(exp_data["steps"], exp_data["returns"], label=exp_name, linewidth=2)
    
    plt.xlabel("Number of Environment Steps")
    plt.ylabel("Average Return")
    plt.title("CartPole-v1 Learning Curves")
    plt.legend()
    plt.grid(True)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {save_path}")
    plt.show()

def main():
    # Path to the data directory
    data_path = os.path.join(os.getcwd(), "homework_fall2023", "hw2", "data")
    
    if not os.path.exists(data_path):
        print(f"Data directory not found at: {data_path}")
        return
        
    data = load_data(data_path)
    
    if not data:
        print("No CartPole-v1 experiment data found")
        return
        
    # Create the plots directory if it doesn't exist
    plots_dir = os.path.join(data_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot and save the learning curves
    save_path = os.path.join(plots_dir, "cartpole_learning_curves.png")
    plot_learning_curves(data, save_path)

if __name__ == "__main__":
    main() 