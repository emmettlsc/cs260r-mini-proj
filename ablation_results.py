# analyze_ablation.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def analyze_ablation_results(grid_id):
    # Load results
    results_file = Path(f"ablation_study_grid_{grid_id}/ablation_results.npz")
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
        
    # Load and convert to DataFrame for easier analysis
    data = np.load(results_file, allow_pickle=True)['results'].item()
    
    # Prepare data for DataFrame
    rows = []
    for param_str, result in data.items():
        if "error" in result:
            # Skip runs with errors
            continue
            
        row = {}
        # Extract parameters
        for param_name, param_value in result["params"].items():
            row[param_name] = param_value
            
        # Extract metrics
        if "validation" in result:
            for metric_name, metric_value in result["validation"].items():
                row[f"val_{metric_name}"] = metric_value
                
        rows.append(row)
        
    if not rows:
        print("No valid results found!")
        return
        
    # Create DataFrame
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} results")
    print(f"Columns: {df.columns.tolist()}")
    
    # Identify parameter columns vs metric columns
    param_cols = [col for col in df.columns if not col.startswith('val_')]
    metric_cols = [col for col in df.columns if col.startswith('val_')]
    
    # Show summary statistics
    print("\nSummary Statistics:")
    print(df[metric_cols].describe())
    
    # Create visualizations based on which grid was used
    plt.figure(figsize=(12, 8))
    
    if grid_id == 1:  # Learning rate and batch size
        # Plot impact of learning rate on route completion for different batch sizes
        if 'learning_rate' in df.columns and 'batch_size' in df.columns:
            sns.lineplot(data=df, x='learning_rate', y='val_route_completion', 
                        hue='batch_size', marker='o', palette='viridis')
            plt.title('Impact of Learning Rate and Batch Size on Route Completion')
            plt.xscale('log')  # Log scale for learning rate
            plt.tight_layout()
            plt.savefig(f"ablation_study_grid_{grid_id}/lr_batch_route_completion.png")
            
            # Plot impact on reward
            plt.figure(figsize=(12, 8))
            sns.lineplot(data=df, x='learning_rate', y='val_mean_reward', 
                        hue='batch_size', marker='o', palette='viridis')
            plt.title('Impact of Learning Rate and Batch Size on Mean Reward')
            plt.xscale('log')
            plt.tight_layout()
            plt.savefig(f"ablation_study_grid_{grid_id}/lr_batch_reward.png")
            
    elif grid_id == 2:  # n_steps and entropy coefficient
        # Plot impact of n_steps on route completion for different entropy coefficients
        if 'n_steps' in df.columns and 'ent_coef' in df.columns:
            sns.lineplot(data=df, x='n_steps', y='val_route_completion', 
                        hue='ent_coef', marker='o', palette='viridis')
            plt.title('Impact of Step Size and Entropy Coefficient on Route Completion')
            plt.tight_layout()
            plt.savefig(f"ablation_study_grid_{grid_id}/nsteps_ent_route_completion.png")
            
            # Plot impact on reward
            plt.figure(figsize=(12, 8))
            sns.lineplot(data=df, x='n_steps', y='val_mean_reward', 
                        hue='ent_coef', marker='o', palette='viridis')
            plt.title('Impact of Step Size and Entropy Coefficient on Mean Reward')
            plt.tight_layout()
            plt.savefig(f"ablation_study_grid_{grid_id}/nsteps_ent_reward.png")
            
    elif grid_id == 3:  # n_epochs and clip_range
        # Plot impact of n_epochs on route completion for different clip ranges
        if 'n_epochs' in df.columns and 'clip_range' in df.columns:
            sns.lineplot(data=df, x='n_epochs', y='val_route_completion', 
                        hue='clip_range', marker='o', palette='viridis')
            plt.title('Impact of Epochs and Clip Range on Route Completion')
            plt.tight_layout()
            plt.savefig(f"ablation_study_grid_{grid_id}/epochs_clip_route_completion.png")
            
            # Plot impact on reward
            plt.figure(figsize=(12, 8))
            sns.lineplot(data=df, x='n_epochs', y='val_mean_reward', 
                        hue='clip_range', marker='o', palette='viridis')
            plt.title('Impact of Epochs and Clip Range on Mean Reward')
            plt.tight_layout()
            plt.savefig(f"ablation_study_grid_{grid_id}/epochs_clip_reward.png")
    
    # Create a heat map if there are enough data points
    if len(df) >= 4 and len(param_cols) >= 2:
        plt.figure(figsize=(10, 8))
        pivot_table = df.pivot_table(
            index=param_cols[0], 
            columns=param_cols[1], 
            values='val_route_completion'
        )
        sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.3f')
        plt.title(f'Heatmap of Route Completion by {param_cols[0]} and {param_cols[1]}')
        plt.tight_layout()
        plt.savefig(f"ablation_study_grid_{grid_id}/heatmap_route_completion.png")
    
    # Find best configuration
    best_idx = df['val_route_completion'].idxmax()
    best_config = df.iloc[best_idx]
    
    print("\nBest Configuration:")
    for param in param_cols:
        print(f"- {param}: {best_config[param]}")
    print("\nMetrics:")
    for metric in metric_cols:
        print(f"- {metric}: {best_config[metric]:.4f}")
    
    return df

if __name__ == "__main__":
    # Analyze each grid separately
    grid_ids = [1, 2, 3]
    
    for grid_id in grid_ids:
        try:
            print(f"\n=== Analyzing Grid {grid_id} ===")
            df = analyze_ablation_results(grid_id)
            if df is not None:
                print(f"Analysis for Grid {grid_id} complete.")
        except Exception as e:
            print(f"Error analyzing Grid {grid_id}: {e}")