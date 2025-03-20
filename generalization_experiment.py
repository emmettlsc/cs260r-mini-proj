import argparse
import datetime
import logging
import os
import uuid
import torch
from collections import defaultdict
from pathlib import Path

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

# Remove MetaDrive's logging information when episode ends.
set_log_level(logging.ERROR)

# c6i.32xlarge 128 vCPUs, 256 GiB mem
torch.set_num_threads(64)

def get_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def remove_reset_seed_and_add_monitor(make_env, trial_dir):
    """
    MetaDrive env's reset function takes a seed argument and use it to determine the map to load.
    However, in stable-baselines3, it calls reset function with a seed argument serving as the random seed,
    which is not what we want. We do a trick here to remap the random seed to map index.

    Stable-baselines3 recommends using Monitor wrapper to log training data. We add a Monitor wrapper here.
    """
    from gymnasium import Wrapper
    from stable_baselines3.common.monitor import Monitor
    class NewClass(Wrapper):
        def reset(self, seed=None, **kwargs):
            # PZH: We do a trick here to remap the seed to the map index. This can help randomize the maps.
            if seed is not None:
                new_seed = self.env.start_index + (seed % self.env.num_scenarios)
            else:
                new_seed = None
            return self.env.reset(seed=new_seed, **kwargs)

    def new_make_env():
        env = make_env()
        NewClass.__name__ = env.__class__.__name__ + "WithoutResetSeed"
        wrapped_env = NewClass(env)
        wrapped_env = Monitor(env=wrapped_env, filename=str(trial_dir))
        return wrapped_env

    return new_make_env

class CustomizedEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluations_info_buffer = defaultdict(list)

    def _log_success_callback(self, locals_, globals_):
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

            maybe_is_success2 = info.get("arrive_dest", None)
            if maybe_is_success2 is not None:
                self._is_success_buffer.append(maybe_is_success2)

            assert (maybe_is_success is None) or (maybe_is_success2 is None), "We cannot have two success flags!"

            for k in ["route_completion", "total_cost", "arrive_dest", "max_step", "out_of_road", "crash"]:
                if k in info:
                    self.evaluations_info_buffer[k].append(info[k])

        if "raw_action" in info:
            self.evaluations_info_buffer["raw_action"].append(info["raw_action"])

    def _on_step(self) -> bool:
        """
        PZH Note: Overall this function is copied from original EvalCallback._on_step.
        We additionally record evaluations_info_buffer to the logger.
        """

        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.vec_env import sync_envs_normalization

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                # PZH: Save evaluations_info_buffer to the log file
                for k, v in self.evaluations_info_buffer.items():
                    kwargs[k] = v

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,  # type: ignore[arg-type]
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            # PZH: Add this metric.
            self.logger.record("eval/num_episodes", len(episode_rewards))

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # PZH: We record evaluations_info_buffer to the logger
            for k, v in self.evaluations_info_buffer.items():
                self.logger.record("eval/{}".format(k), np.mean(np.asarray(v)))

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

def run_generalization_experiment(map_count, scene_counts, seed=42):
    """
    generalization results gen here 
    """
    valid_scene_counts = [count for count in scene_counts if count >= map_count]
    
    base_dir = Path(f"generalization_experiment_maps_{map_count}")
    os.makedirs(base_dir, exist_ok=True)
    
    all_results = {}
    results_file = base_dir / "all_results.npz"
    
    # existing results if available
    if os.path.exists(results_file):
        try:
            loaded_data = np.load(results_file, allow_pickle=True)
            all_results = dict(loaded_data['results'].item())
            print(f"Loaded existing results from {results_file}")
        except Exception as e:
            print(f"Error loading existing results: {e}")
    
    for scene_count in valid_scene_counts:
        # Skip if  done
        if str(scene_count) in all_results:
            print(f"Skipping scene count {scene_count} as it's already in results")
            continue
            
        print(f"\n=== Running experiment with {map_count} maps, {scene_count} scenes ===")
        
        # experiment config
        exp_name = f"gen_ppo_maps_{map_count}_scenes_{scene_count}"
        trial_name = f"{exp_name}_{get_time_str()}_{uuid.uuid4().hex[:8]}"
        trial_dir = base_dir / trial_name
        os.makedirs(trial_dir, exist_ok=True)
        os.makedirs(trial_dir / "models", exist_ok=True)
        os.makedirs(trial_dir / "eval_train", exist_ok=True)
        os.makedirs(trial_dir / "eval_val", exist_ok=True)
        
        print(f"Results will be saved to {trial_dir}")
        
        # ===== Setup environment =====
        num_train_envs = 64
        num_eval_envs = 5
        
        def get_training_env_with_scenes():
            return get_training_env({"num_scenarios": scene_count, "start_seed": 100})
        
        # Create environments
        train_env = make_vec_env(
            remove_reset_seed_and_add_monitor(get_training_env_with_scenes, trial_dir), 
            n_envs=num_train_envs,
            vec_env_cls=SubprocVecEnv
        )
        
        eval_train_env = make_vec_env(
            remove_reset_seed_and_add_monitor(get_training_env_with_scenes, trial_dir), 
            n_envs=num_eval_envs,
            vec_env_cls=SubprocVecEnv
        )
        
        eval_val_env = make_vec_env(
            remove_reset_seed_and_add_monitor(get_validation_env, trial_dir), 
            n_envs=num_eval_envs,
            vec_env_cls=SubprocVecEnv
        )
        
        # ===== Setup callbacks =====
        save_freq = 5000
        eval_freq = 25000
        
        checkpoint_callback = CheckpointCallback(
            name_prefix="ppo_model",
            verbose=1,
            save_freq=save_freq,
            save_path=str(trial_dir / "models")
        )
        
        eval_train_callback = CustomizedEvalCallback(
            eval_train_env,
            best_model_save_path=str(trial_dir / "eval_train"),
            log_path=str(trial_dir / "eval_train"),
            eval_freq=max(eval_freq // num_train_envs, 1),
            n_eval_episodes=50,
            verbose=1
        )
        
        eval_val_callback = CustomizedEvalCallback(
            eval_val_env,
            best_model_save_path=str(trial_dir / "eval_val"),
            log_path=str(trial_dir / "eval_val"),
            eval_freq=max(eval_freq // num_train_envs, 1),
            n_eval_episodes=50,
            verbose=1
        )
        
        callbacks = CallbackList([checkpoint_callback, eval_train_callback, eval_val_callback])
        
        # ===== Setup training algorithm =====
        model = PPO(
            policy=ActorCriticPolicy,
            env=train_env,
            n_steps=128,
            batch_size=64,
            n_epochs=10,
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
        
        # ===== Train model =====
        # total_timesteps = 250000 this was prty good, gonna try a mil
        total_timesteps = 1000000

        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
            
            # save final model so I can recover it if needed
            final_model_path = trial_dir / "models" / "final_model.zip"
            model.save(final_model_path)
            
            # ===== Final evaluation =====
            print("=== Final Evaluation ===")

            # training env eval
            print(f"Evaluating on training environment (maps: {map_count}, scenes: {scene_count})...")
            train_mean_reward, train_std_reward = evaluate_policy(
                model,
                eval_train_env,
                n_eval_episodes=100,
                return_episode_rewards=False
            )
            
            train_metrics = {}
            try:
                train_npz = np.load(str(trial_dir / "eval_train" / "evaluations.npz"), allow_pickle=True)
                if "route_completion" in train_npz:
                    train_metrics["route_completion"] = float(np.mean(train_npz["route_completion"]))
                if "arrive_dest" in train_npz:
                    train_metrics["success_rate"] = float(np.mean(train_npz["arrive_dest"]))
            except Exception as e:
                print(f"Error loading training metrics: {e}")
            
            # validation env eval
            print("Evaluating on validation environment...")
            val_mean_reward, val_std_reward = evaluate_policy(
                model,
                eval_val_env,
                n_eval_episodes=100,
                return_episode_rewards=False  # Set to False when unpacking mean and std
            )
            
            # route completion and other metrics from callback
            val_metrics = {}
            try:
                val_npz = np.load(str(trial_dir / "eval_val" / "evaluations.npz"), allow_pickle=True)
                if "route_completion" in val_npz:
                    val_metrics["route_completion"] = float(np.mean(val_npz["route_completion"]))
                if "arrive_dest" in val_npz:
                    val_metrics["success_rate"] = float(np.mean(val_npz["arrive_dest"]))
            except Exception as e:
                print(f"Error loading validation metrics: {e}")
            
            # combine and sace results
            scene_results = {
                "train": {
                    "mean_reward": train_mean_reward,
                    "std_reward": train_std_reward,
                    **train_metrics
                },
                "validation": {
                    "mean_reward": val_mean_reward,
                    "std_reward": val_std_reward,
                    **val_metrics
                }
            }
            np.save(trial_dir / "scene_results.npy", scene_results)
            all_results[str(scene_count)] = scene_results
            np.savez(results_file, results=all_results)
            
        except Exception as e:
            print(f"Error during training/evaluation: {e}")
            all_results[str(scene_count)] = {"error": str(e)}
            np.savez(results_file, results=all_results)
        finally:
            train_env.close()
            eval_train_env.close()
            eval_val_env.close()
    
    print(f"All experiments complete! Final results saved to {results_file}")
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", type=int, choices=[1, 3, 10], required=True, 
                        help="Number of maps to train on (1, 3, or 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--scene-counts", type=str, default="1,5,20,50,100,300,1000",
                        help="Comma-separated list of scene counts to evaluate")
    args = parser.parse_args()
    
    scene_counts = [int(x) for x in args.scene_counts.split(",")]
    
    run_generalization_experiment(args.maps, scene_counts, args.seed)