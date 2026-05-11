"""Load the trained rl-games policy in standalone mode.

The policy is the EXACT same one used by:
    python scripts/rl_games/play.py --task=Isaac-Velocity-CaT-Flat-AlienGo-Play-v0 ...

That is, `<run_dir>/nn/solo_cat.pth` together with the running-mean/std
observation normalizer stored inside the same checkpoint. We deliberately
bypass the Isaac Lab env and Isaac Sim launcher: only rl-games is needed.

Notes for sim-to-sim correctness:
- normalize_input=True in agent.yaml means observations are normalized by a
  RunningMeanStd that lives inside the model. `PpoPlayerContinuous.restore()`
  loads that normalizer's state alongside the actor weights.
- Player must run in deterministic mode (use the mu head of the Gaussian),
  matching `player.deterministic=True` in agent.yaml.
- The observation space we declare here is just a placeholder Box shape for
  rl-games to build the right MLP input dimension; the actual observation
  contents are produced by `obs_builder.build_obs`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import gym  # rl-games uses classic gym, not gymnasium
import gym.spaces
import numpy as np
import torch
import yaml


OBS_DIM = 57
ACT_DIM = 12


@dataclass
class PolicyHandle:
    player: object
    device: torch.device
    obs_dim: int = OBS_DIM
    act_dim: int = ACT_DIM
    run_dir: str = ""
    checkpoint_path: str = ""

    @torch.inference_mode()
    def act(self, obs_np: np.ndarray) -> np.ndarray:
        """Run one deterministic forward pass.

        Args:
            obs_np: shape (OBS_DIM,), float32. Already in the SAME order /
                scale as Isaac Lab observation manager (see obs_builder).

        Returns:
            action_np: shape (ACT_DIM,), float32. RAW policy output, identical
                to `agent.get_action(..., is_deterministic=True)` in play.py.
                Caller is responsible for applying the JointPositionAction
                pipeline (scale 0.3 + default joint offset).
        """
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        # PpoPlayerContinuous.get_action auto-unsqueezes if has_batch_dimension is False.
        # It also runs the model's internal RunningMeanStd normalizer automatically.
        action = self.player.get_action(obs_t, is_deterministic=True)
        return action.detach().cpu().numpy().astype(np.float32).reshape(-1)


def load_policy(
    run_dir: str,
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
) -> PolicyHandle:
    """Build an rl-games PpoPlayerContinuous from a saved Isaac Lab run.

    Args:
        run_dir: e.g. "logs/rl_games/solo_cat/2026-05-07_21-28-14".
        checkpoint: explicit .pth path. If None, picks "<run_dir>/nn/solo_cat.pth"
            (the best checkpoint, same as play.py without --use_last_checkpoint).
        device: "cuda" / "cuda:0" / "cpu". Defaults to cuda if available.

    Returns:
        PolicyHandle wrapping the rl-games player.
    """
    # Defer rl-games imports so optional sim setup steps don't trigger TF/CUDA init twice.
    from rl_games.algos_torch import players
    from rl_games.common.algo_observer import DefaultAlgoObserver
    from rl_games.torch_runner import Runner

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    run_dir = os.path.abspath(run_dir)
    agent_yaml = os.path.join(run_dir, "params", "agent.yaml")
    if not os.path.isfile(agent_yaml):
        raise FileNotFoundError(f"agent.yaml not found: {agent_yaml}")

    if checkpoint is None:
        checkpoint = os.path.join(run_dir, "nn", "solo_cat.pth")
    checkpoint = os.path.abspath(checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    with open(agent_yaml, "r") as f:
        cfg = yaml.unsafe_load(f)  # the yaml contains python tuple/slice tags

    # The yaml uses cuda:0 hard-coded for training; let the user override.
    cfg["params"]["config"]["device"] = device
    cfg["params"]["config"]["device_name"] = device
    cfg["params"]["config"]["multi_gpu"] = False

    # Bypass env creation: provide env_info directly so rl-games builds the network
    # with the right input/output dims without needing IsaacSim/IsaacLab loaded.
    cfg["params"]["config"]["env_info"] = {
        "observation_space": gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        ),
        "action_space": gym.spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        ),
        "agents": 1,
        "value_size": 1,
    }
    cfg["params"]["config"]["num_actors"] = 1
    # Force deterministic at the player level (matches Isaac Lab play.py).
    cfg["params"]["config"].setdefault("player", {})
    cfg["params"]["config"]["player"]["deterministic"] = True
    cfg["params"]["config"]["player"]["use_vecenv"] = False

    cfg["params"]["load_checkpoint"] = True
    cfg["params"]["load_path"] = checkpoint

    # Replicate the algo registration that scripts/rl_games/play.py uses via
    # cat_envs.tasks.utils.rl_games.build_alg_runner.build_alg_runner. We avoid
    # importing cat_envs (which transitively imports omni.kit) by inlining the
    # one line we need: register the player builder for "cat_a2c_continuous".
    runner = Runner(DefaultAlgoObserver())
    runner.player_factory.register_builder(
        "cat_a2c_continuous", lambda **kwargs: players.PpoPlayerContinuous(**kwargs)
    )
    runner.load(cfg)
    player = runner.create_player()
    # restore loads BOTH the actor MLP weights AND the running_mean_std state dict
    player.restore(checkpoint)
    player.reset()

    return PolicyHandle(
        player=player,
        device=torch.device(device),
        run_dir=run_dir,
        checkpoint_path=checkpoint,
    )
