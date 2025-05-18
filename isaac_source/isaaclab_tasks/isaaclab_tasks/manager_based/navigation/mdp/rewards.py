# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    return (1 - torch.tanh(distance / std)) * (env.episode_length_buf < 150).float()

def arrived_reward(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    
    return (distance < std).float() * (env.episode_length_buf < 300).float()

def ref_reward(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    if not hasattr(env, "ref_path"):
        return torch.zeros([env.num_envs], device=env.device).reshape(-1,)
    else:
        asset = env.scene['robot']
        pos_of_robot = asset.data.root_state_w[:, :2] - env.scene.env_origins.reshape(-1, 3)[..., :2]
        ref_path = env.ref_path
        min_dists = []
        for i in range(len(ref_path)):
            pos = pos_of_robot[i]  
            traj = ref_path[i]     
            
            dists = torch.norm(traj - pos, dim=1)
            min_dist = torch.min(dists)  
            min_dists.append(min_dist)

        min_dists = torch.stack(min_dists)  
        reward = -torch.tanh(min_dists / 5.0)
        return reward

def velocity_reward(env: ManagerBasedRLEnv, std: float=1.0) -> torch.Tensor:
    asset = env.scene['robot']
    vel = asset.data.root_state_w[:, 7:9]
    vel = torch.linalg.norm(vel, dim=1)
    return torch.clamp(vel - std, min=-1., max=0.2).float()

# def velocity_reward(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
#     """Reward position tracking with tanh kernel."""
#     command = env.command_manager.get_command(command_name)
#     des_pos_b = command[:, :2]
#     distance = torch.norm(des_pos_b, dim=1)
#     return (distance < std).float() * (env.episode_length_buf < 300).float()

def arrived_term(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    return (distance < std).to(torch.bool)

def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()
