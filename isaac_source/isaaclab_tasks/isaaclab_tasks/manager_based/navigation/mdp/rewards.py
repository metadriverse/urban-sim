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
    return (1 - torch.tanh(distance / std)).float()

def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()

def moving_towards_goal_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    movement_xy = command[:, -1:]
    reward = movement_xy[:, 0]
    return reward * (env.episode_length_buf >= 10).float() 

def target_vel_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    target_pos = command[:, :2]
    distance_to_target_pos = torch.linalg.norm(target_pos, dim=1, keepdim=True)
    
    asset = env.scene['robot']
    vel = asset.data.root_lin_vel_b[:, 0:2]
    
    vel_direction = target_pos / distance_to_target_pos.clamp_min(1e-6)
    reward_vel = (vel * vel_direction).sum(-1)
    return reward_vel

