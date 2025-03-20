import pathlib

import torch

# Use dot here to denote importing the file in the folder hosting this file.
from .common.policies import ActorCriticPolicy

FOLDER_ROOT = pathlib.Path(__file__).parent  # The path to the folder hosting this file.


class Policy:
    """
    This class is the interface where the evaluation scripts communicate with your trained agent.

    You can initialize your model and load weights in the __init__ function. At each environment interactions,
    the batched observation `obs`, a numpy array with shape (Batch Size, Obs Dim), will be passed into the __call__
    function. You need to generate the action, a numpy array with shape (Batch Size, Act Dim=2), and return it.

    Do not change the name of this class.

    Please do not import any external package.
    """
    CREATOR_NAME = "Emmett Cocke"
    CREATOR_UID = "605615161"

    def __init__(self):
        # data = torch.load(FOLDER_ROOT / "example_sb3_ppo_agent.pt", weights_only=False)
        data = torch.load(FOLDER_ROOT / "my_agent.pt", weights_only=False, map_location=torch.device('cpu'))
        torch.set_num_threads(64)
        policy_kwargs = data.get("policy_kwargs", {})

        policy = ActorCriticPolicy(
            action_space=data["action_space"],
            observation_space=data["observation_space"],
            lr_schedule=lambda x: 0.0,
            **policy_kwargs
        )
        missing_keys, unexpected_keys = policy.load_state_dict(data["state_dict"])
        assert not missing_keys, f"Missing keys: {missing_keys}"
        assert not unexpected_keys, f"Unexpected keys: {unexpected_keys}"
        self.policy = policy

    def __call__(self, obs):
        with torch.no_grad():
            action, value, log_probability = self.policy(torch.from_numpy(obs))
        return action.cpu().numpy()
