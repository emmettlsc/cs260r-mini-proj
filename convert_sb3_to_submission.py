"""
This script reads the saved sb3 agent and save useful data into a new file that will be used
later in `agents/agent_sb3/agent.py` without stable-baselines3 dependency.
"""

import torch
from stable_baselines3.common.save_util import load_from_zip_file


if __name__ == '__main__':
    # TODO: Change the path to whatever ablation says is the best
    #ckpt = "runs/ppo_metadrive/ppo_metadrive_2025-02-20_20-34-09_ec54bb59/models/rl_model_320000_steps.zip"
    ckpt = "final_agent/final_agent_2025-03-20_17-16-12_cae90892/models/final_model.zip"
    
    data, params, pytorch_variables = load_from_zip_file(ckpt)
    policy_kwargs = data.get("policy_kwargs", {})

    # Just make a customized dict to store data we want to use later.
    new_data = dict(
        action_space=data["action_space"],
        observation_space=data["observation_space"],
        state_dict=params['policy'],
        policy_kwargs=policy_kwargs
    )
    torch.save(new_data, "my_agent.pt")
