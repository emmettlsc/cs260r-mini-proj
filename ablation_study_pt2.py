import argparse
import datetime
import logging
import os
import uuid
import torch
from collections import defaultdict
from pathlib import Path
import itertools

import numpy as np
from metadrive.engine.logger import set_log_level
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import ActorCriticPolicy

from env import get_training_env, get_validation_env

set_log_level(logging.ERROR)

torch.set_num_threads(64)

def get_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

from generalization_experiment import remove_reset_seed_and_add_monitor, CustomizedEvalCallback

def run_ablation_experiment(grid_id):
    """
    Run ablation study with a grid of hyperparameters.
    
    Args:
        grid_id (int): ID of the hyperparameter grid to use (1, 2, or 3)
    """
    # Define enhanced grids to explore more parameters
    
    # Grid 1: Network Architecture Exploration
    if grid_id == 1:
        param_grid = {
            'policy_arch': [
                # Default architecture
                [dict(pi=[64, 64], vf=[64, 64])],
                # Deeper architecture
                [dict(pi=[128, 128], vf=[128, 128])],
                # Asymmetric architecture (deeper policy)
                [dict(pi=[128, 128, 64], vf=[64, 64])]
            ],
            'activation_fn': [
                torch.nn.ReLU,
                torch.nn.Tanh
            ]
        }
    
    # Grid 2: Value Function Parameters
    elif grid_id == 2:
        param_grid = {
            'clip_range_vf': [None, 0.1, 0.2, 0.4],
            'vf_coef': [0.1, 0.5, 1.0]
        }
    
    # Grid 3: Advanced Stability Parameters
    elif grid_id == 3:
        param_grid = {
            'target_kl': [None, 0.01, 0.05],
            'normalize_advantage': [True, False],
            'max_grad_norm': [0.3, 0.5, 0.7]
        }
    else:
        raise ValueError(f"Invalid grid_id: {grid_id}. Must be 1, 2, or 3.")
    
    # Generate all combinations
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    print(f"Running ablation study with Grid {grid_id}")
    print(f"Parameter grid: {param_grid}")
    print(f"Total combinations: {len(param_combinations)}")
    
    # Base directory for results
    base_dir = Path(f"ablation_study_advanced_grid_{grid_id}")
    os.makedirs(base_dir, exist_ok=True)
    
    # Store all results
    all_results = {}
    results_file = base_dir / "ablation_results.npz"
    
    # Load existing results if available
    if os.path.exists(results_file):
        try:
            loaded_data = np.load(results_file, allow_pickle=True)
            all_results = dict(loaded_data['results'].item())
            print(f"Loaded existing results from {results_file}")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            all_results = {}
    
    # Create environment config - using 3 maps for a good balance of training speed and diversity
    env_config = {
        "num_scenarios": 3,
    }
    
    # Go through each parameter combination
    for i, param_values in enumerate(param_combinations):
        # Create parameter dictionary for this run
        params = dict(zip(param_names, param_values))
        
        # Generate a unique string for this parameter combination
        if grid_id == 1:
            # Handle the special case of network architecture
            arch_str = str(params['policy_arch']).replace(' ', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace(':', '_').replace(',', '_').replace('\'', '')
            act_str = params['activation_fn'].__name__
            param_str = f"arch_{arch_str}_act_{act_str}"
        else:
            param_str = "_".join([f"{name}_{value}" for name, value in params.items()])
        
        # Check if this combination was already run
        if param_str in all_results:
            print(f"Skipping parameter combination {i+1}/{len(param_combinations)}: {param_str} (already done)")
            continue
        
        print(f"\n=== Running parameter combination {i+1}/{len(param_combinations)}: {param_str} ===")
        
        # Set up experiment directory
        exp_name = f"ablation_grid{grid_id}_{param_str[:50]}"  # Limit length for filesystem
        trial_name = f"{exp_name}_{get_time_str()}_{uuid.uuid4().hex[:6]}"
        trial_dir = base_dir / trial_name
        os.makedirs(trial_dir, exist_ok=True)
        os.makedirs(trial_dir / "models", exist_ok=True)
        os.makedirs(trial_dir / "eval", exist_ok=True)
        
        # Training environment setup
        def get_training_env_for_ablation():
            return get_training_env(env_config)
        
        # Environment setup
        num_train_envs = 32
        num_eval_envs = 5
        
        train_env = make_vec_env(
            remove_reset_seed_and_add_monitor(get_training_env_for_ablation, trial_dir), 
            n_envs=num_train_envs,
            vec_env_cls=SubprocVecEnv
        )
        
        eval_env = make_vec_env(
            remove_reset_seed_and_add_monitor(get_validation_env, trial_dir), 
            n_envs=num_eval_envs,
            vec_env_cls=SubprocVecEnv
        )
        
        # Callback setup
        save_freq = 50000
        eval_freq = 100000
        
        checkpoint_callback = CheckpointCallback(
            name_prefix="ablation_model",
            verbose=1,
            save_freq=save_freq,
            save_path=str(trial_dir / "models")
        )
        
        eval_callback = CustomizedEvalCallback(
            eval_env,
            best_model_save_path=str(trial_dir / "eval"),
            log_path=str(trial_dir / "eval"),
            eval_freq=eval_freq // num_train_envs,
            n_eval_episodes=50,
            verbose=1
        )
        
        callbacks = CallbackList([checkpoint_callback, eval_callback])
        
        # Default parameters - using optimal values from previous ablation
        default_params = {
            'policy': ActorCriticPolicy,
            'env': train_env,
            'n_steps': 128,
            'batch_size': 64,
            'n_epochs': 20,  # From previous ablation
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'learning_rate': 0.0001,  # From previous ablation
            'clip_range': 0.2,
            'clip_range_vf': None,
            'normalize_advantage': True,
            'ent_coef': 0.01,  # From previous ablation
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'target_kl': None,
            'tensorboard_log': str(trial_dir),
            'verbose': 1,
            'policy_kwargs': None
        }
        
        # Handle special grid 1 case (network architecture)
        if grid_id == 1:
            policy_arch = params['policy_arch']
            activation_fn = params['activation_fn']
            
            default_params['policy_kwargs'] = dict(
                net_arch=policy_arch,
                activation_fn=activation_fn
            )
        else:
            # Update with the specific parameters for this run
            for name, value in params.items():
                default_params[name] = value
        
        # Create model
        model = PPO(**default_params)
        
        try:
            # Train model - increase training time for better evaluation
            total_timesteps = 1000000  # 1M steps
            
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
            
            # Save final model
            final_model_path = trial_dir / "models" / "final_model.zip"
            model.save(final_model_path)
            
            # Final evaluation
            print("=== Final Evaluation ===")
            
            # Evaluate on validation environment
            print("Evaluating on validation environment...")
            mean_reward, std_reward = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=100,
                return_episode_rewards=False
            )
            
            # Get metrics from evaluation files
            metrics = {}
            try:
                eval_npz = np.load(str(trial_dir / "eval" / "evaluations.npz"), allow_pickle=True)
                if "route_completion" in eval_npz:
                    last_idx = -1  # Get the last evaluation
                    if isinstance(eval_npz["route_completion"], list) and eval_npz["route_completion"]:
                        metrics["route_completion"] = float(np.mean(eval_npz["route_completion"][last_idx]))
                    else:
                        metrics["route_completion"] = float(np.mean(eval_npz["route_completion"]))
                
                if "total_cost" in eval_npz:
                    last_idx = -1
                    if isinstance(eval_npz["total_cost"], list) and eval_npz["total_cost"]:
                        metrics["total_cost"] = float(np.mean(eval_npz["total_cost"][last_idx]))
                    else:
                        metrics["total_cost"] = float(np.mean(eval_npz["total_cost"]))
                
                if "arrive_dest" in eval_npz:
                    last_idx = -1
                    if isinstance(eval_npz["arrive_dest"], list) and eval_npz["arrive_dest"]:
                        metrics["success_rate"] = float(np.mean(eval_npz["arrive_dest"][last_idx]))
                    else:
                        metrics["success_rate"] = float(np.mean(eval_npz["arrive_dest"]))
            except Exception as e:
                print(f"Error loading evaluation metrics: {e}")
            
            # Save results
            # Store parameter info in a more readable format
            param_info = {}
            if grid_id == 1:
                param_info['net_arch'] = str(params['policy_arch'])
                param_info['activation_fn'] = params['activation_fn'].__name__
            else:
                param_info = params.copy()
                
            results = {
                "params": param_info,
                "validation": {
                    "mean_reward": float(mean_reward),
                    "std_reward": float(std_reward),
                    **metrics
                }
            }
            
            # Add to overall results
            all_results[param_str] = results
            
            # Save all results
            np.savez(results_file, results=all_results)
            
        except Exception as e:
            print(f"Error during training/evaluation: {e}")
            # Save error in results
            if grid_id == 1:
                param_info = {
                    'net_arch': str(params['policy_arch']),
                    'activation_fn': params['activation_fn'].__name__
                }
            else:
                param_info = params.copy()
                
            all_results[param_str] = {"params": param_info, "error": str(e)}
            np.savez(results_file, results=all_results)
        finally:
            # Clean up
            train_env.close()
            eval_env.close()
    
    print(f"Ablation study complete! Results saved to {results_file}")
    
    # Print summary of results
    print("\n=== Results Summary ===")
    for param_str, result in all_results.items():
        if "error" in result:
            print(f"{param_str}: Error - {result['error']}")
        elif "validation" in result:
            metrics_str = ", ".join([
                f"{k}: {v:.4f}" for k, v in result["validation"].items() 
                if k not in ["std_reward"]
            ])
            print(f"{param_str}: {metrics_str}")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, required=True, choices=[1, 2, 3],
                       help="Grid ID to run (1, 2, or 3)")
    args = parser.parse_args()
    
    run_ablation_experiment(args.grid)