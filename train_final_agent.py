# train_final_agent.py
import torch
import os
import datetime
import uuid
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import get_training_env, get_validation_env
from generalization_experiment import remove_reset_seed_and_add_monitor, CustomizedEvalCallback

def get_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def train_final_agent():
    # Set up directories
    exp_name = "final_agent"
    trial_name = f"{exp_name}_{get_time_str()}_{uuid.uuid4().hex[:8]}"
    trial_dir = Path("final_agent") / trial_name
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(trial_dir / "models", exist_ok=True)
    os.makedirs(trial_dir / "eval", exist_ok=True)
    
    print(f"Training final agent. Results will be saved to {trial_dir}")
    
    # Set up environment
    # Using 10 maps with 1000 scenes as determined from generalization experiment
    env_config = { "num_scenarios": 3 }
    
    # Environment setup
    num_train_envs = 64  # Use more parallel environments for faster training
    num_eval_envs = 5
    
    # Create training environment maker function
    def get_final_training_env():
        return get_training_env(env_config)
    
    # Create environments
    train_env = make_vec_env(
        remove_reset_seed_and_add_monitor(get_final_training_env, trial_dir), 
        n_envs=num_train_envs,
        vec_env_cls=SubprocVecEnv
    )
    
    eval_env = make_vec_env(
        remove_reset_seed_and_add_monitor(get_validation_env, trial_dir), 
        n_envs=num_eval_envs,
        vec_env_cls=SubprocVecEnv
    )
    
    # Callbacks
    save_freq = 100000  # Save less frequently for longer training
    eval_freq = 200000  # Evaluate less frequently
    
    checkpoint_callback = CheckpointCallback(
        name_prefix="final_model",
        verbose=1,
        save_freq=save_freq,
        save_path=str(trial_dir / "models")
    )
    
    eval_callback = CustomizedEvalCallback(
        eval_env,
        best_model_save_path=str(trial_dir / "eval"),
        log_path=str(trial_dir / "eval"),
        eval_freq=eval_freq // num_train_envs,
        n_eval_episodes=100,  # More evaluation episodes for better estimates
        verbose=1
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # Use the optimized parameters from ablation study
    model = PPO(
        policy=ActorCriticPolicy,
        env=train_env,
        n_steps=128,
        batch_size=64,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=0.0001,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(trial_dir),
        verbose=1,
    )
    
    # Train for much longer
    total_timesteps = 5_000_000  # 5M steps for final agent
    
    torch.set_num_threads(64)
    
    # Train the model
    print(f"Starting training for {total_timesteps} steps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = trial_dir / "models" / "final_model.zip"
    model.save(final_model_path)
    print(f"Training complete! Final model saved to {final_model_path}")
    
    # Clean up
    train_env.close()
    eval_env.close()
    
    return final_model_path

if __name__ == "__main__":
    final_model_path = train_final_agent()
    print(f"Final model path: {final_model_path}")
    print("Now you can convert this model for submission using the conversion script.")