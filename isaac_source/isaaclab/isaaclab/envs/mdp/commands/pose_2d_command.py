# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import TerrainBasedPose2dCommandCfg, UniformPose2dCommandCfg


import omni
# from isaacsim.asset.gen.omap import _omap
import cv2
import pytest
import numpy as np

class UniformPose2dCommand(CommandTerm):
    """Command generator that generates pose commands containing a 3-D position and heading.

    The command generator samples uniform 2D positions around the environment origin. It sets
    the height of the position command to the default root height of the robot. The heading
    command is either set to point towards the target or is sampled uniformly.
    This can be configured through the :attr:`Pose2dCommandCfg.simple_heading` parameter in
    the configuration.
    """

    cfg: UniformPose2dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPose2dCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        self.temp_target_vec = torch.zeros_like(self.pos_command_w)
        self.relative_movement = torch.zeros_like(self.pos_command_w)
        self.distance_between_frame = torch.zeros_like(self.pos_command_w)
        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
            

    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
        if hasattr(self, 'relative_movement'):
            return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1), self.relative_movement, self.distance_between_frame[:, 0:1]], dim=1)
        else:
            return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1), torch.zeros_like(self.pos_command_b), self.distance_between_frame[:, 0:1]], dim=1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        self.metrics["error_pos_2d"] = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))

    def _resample_command(self, env_ids: Sequence[int]):
        self.use_ref_path = False
        self.generated = False
        self.generating_demo_data = True
        if False:#self.use_ref_path and not self.generated:
            print('[INFO] Using ref path to provide dense reward')
            
            physx = omni.physx.acquire_physx_interface()
            stage_id = omni.usd.get_context().get_stage_id()
            generator = _omap.Generator(physx, stage_id)
            generator.update_settings(.05, 4, 5, 6)
            
            self.output_list = []
            env = self._env
            for idx in range(env.num_envs):
                origin = env.scene.env_origins[idx].cpu().numpy()[..., :2].reshape(2,)
                generator.set_transform((15 + origin[0], 15 + origin[1], 0), (-15, -15, 0.1), (15, 15, 1.5))
                generator.generate2d()
                points = generator.get_occupied_positions()
                buffer = np.array(generator.get_buffer(), dtype=np.uint8)
                h, w = int(30 // 0.05), int(30 // 0.05)  # 地图尺寸
                buffer_2d = buffer.reshape((h, w))

                occupied_mask = (buffer_2d == 4).astype(np.uint8) * 255  # 4 → 255，其它 → 0

                kernel = np.ones((25, 25), dtype=np.uint8)
                dilated = cv2.dilate(occupied_mask, kernel, iterations=1)

                updated = (buffer_2d != 4) & (dilated > 0)
                buffer_2d[updated] = 4
                w = int(30 // 0.05)
                h = int(30 // 0.05)
                mask = np.array(buffer_2d, dtype=np.uint8).reshape((h, w))
                img = np.zeros((h, w, 3), dtype=np.uint8)

                img[mask == 4] = [0, 0, 0]       # occupied - black
                img[mask == 5] = [255, 255, 255] # free - white
                img[mask == 6] = [128, 128, 128] # unknown - gray
                img = cv2.flip(img, 0)  # flip the image vertically
                img = cv2.flip(img, 1) 
                # img = img.transpose(1, 0, 2)
                
                world_start = np.array([2, 15]).reshape(1, 2)
                u = (world_start[:, 0] - 0) / 30
                v = 1.0 - (world_start[:, 1] - 0) / 30
                x_px = u * 30 // 0.05
                y_px = v * 30 // 0.05
                start = np.array([y_px, x_px]).astype(int).reshape(2,)  # [x, y]
                # start = np.array([x_px, y_px]).astype(int).reshape(2,)                
                output = generate_paths(start,(img[..., 0] == 255).astype(np.uint8))
                self.output_list.append(output)
                
            self.generated = True
        if self.generating_demo_data and hasattr(self._env, 'ref_path'):
            return   
        elif self.use_ref_path:
            if not hasattr(self._env, 'ref_path'):
                self._env.ref_path = [[] for _ in range(self._env.scene.num_envs)]
            env_ids_for_sampling = env_ids.cpu().numpy().reshape(len(env_ids), ).tolist()
            for env_id in env_ids_for_sampling:
                stop_sampling = False
                while not stop_sampling:
                    end = output.sample_random_end_point()
                    path = output.unroll_path(end)
                    path = path[:, ::-1]
                    u = path[:, 0] / (30 // 0.05)
                    v = 1.0 - path[:, 1] / (30 // 0.05)
                    x_world = u * 30.
                    y_world = v * 30.
                
                    #valid = (mask[path[:, 0], path[:, 1]] > 0).all()
                    try:
                        if True:# abs(y_world[1] - y_world[0]) < 0.01 and x_world[1] - x_world[0] > 0.01:
                            stop_sampling = True
                            x_world = [1.0, 2.0, 3.0] + x_world.tolist()
                            y_world = [15., 15., 15.] + y_world.tolist()
                    except:
                        continue
                self.pos_command_w[env_id, 0] = self._env.scene.env_origins[env_id, 0] + x_world[-1]
                self.pos_command_w[env_id, 1] = self._env.scene.env_origins[env_id, 1] + y_world[-1]
                self.pos_command_w[env_id, 2] = self._env.scene.env_origins[env_id, 2] + self.robot.data.default_root_state[env_id, 2]
                self._env.ref_path[env_id] = torch.cat([torch.tensor([x_world], device=self.device).reshape(-1, 1), torch.tensor([y_world], device=self.device).reshape(-1, 1)], dim=1)
          
        if True:
            self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]
            # offset the position command by the current root position
            r = torch.empty(len(env_ids), device=self.device)
            self.pos_command_w[env_ids, 0] += r.uniform_(*self.cfg.ranges.pos_x)
            self.pos_command_w[env_ids, 1] += r.uniform_(*self.cfg.ranges.pos_y)
            self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]  
        elif not self.use_ref_path:      
            sampling_times = 0
            suceess_env_sub_ids = []
            # positions_of_assets = self._env.scene.asset_position[env_ids].reshape(len(env_ids), -1, 2)
            while sampling_times <= 200:
                # obtain env origins for the environments
                # self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]
                # # offset the position command by the current root position
                # r = torch.empty(len(env_ids), device=self.device)
                # self.pos_command_w[env_ids, 0] += r.uniform_(*self.cfg.ranges.pos_x)
                # self.pos_command_w[env_ids, 1] += r.uniform_(*self.cfg.ranges.pos_y)
                # self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]
                
                env_ids_to_be_checked = [id_ for id_ in env_ids if id_ not in suceess_env_sub_ids]
                if len(env_ids_to_be_checked) == 0:
                    break
                env_ids_to_be_checked = torch.tensor(env_ids_to_be_checked).reshape(len(env_ids_to_be_checked), )
                positions_of_assets = self._env.scene.asset_position[env_ids_to_be_checked].reshape(len(env_ids_to_be_checked), -1, 2)
                self.pos_command_w[env_ids_to_be_checked] = self._env.scene.env_origins[env_ids_to_be_checked]
                # offset the position command by the current root position
                r = torch.empty(len(env_ids_to_be_checked), device=self.device)
                self.pos_command_w[env_ids_to_be_checked, 0] += r.uniform_(*self.cfg.ranges.pos_x)
                self.pos_command_w[env_ids_to_be_checked, 1] += r.uniform_(*self.cfg.ranges.pos_y)
                self.pos_command_w[env_ids_to_be_checked, 2] += self.robot.data.default_root_state[env_ids_to_be_checked, 2]
                
                distance_to_asset = self.pos_command_w[env_ids_to_be_checked, 0:2].reshape(-1, 1, 2) - positions_of_assets - self._env.scene.env_origins[env_ids_to_be_checked][..., :2].reshape(-1, 1, 2)
                distance_to_asset = torch.min(torch.linalg.norm(distance_to_asset, dim=-1), dim=1).values
                valid_pos = distance_to_asset > 1.5
                valid_pos = valid_pos.reshape(len(env_ids_to_be_checked), ).cpu().numpy().tolist()
                for idx, p in enumerate(valid_pos):
                    if bool(p):
                        suceess_env_sub_ids.append(env_ids_to_be_checked[idx])
                
                sampling_times += 1
                if sampling_times == 50:
                    print('[INFO] Sampling steps -> max step')
                
                # if self.cfg.simple_heading:
                #     # set heading command to point towards target
                #     target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
                #     target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
                #     flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

                #     # compute errors to find the closest direction to the current heading
                #     # this is done to avoid the discontinuity at the -pi/pi boundary
                #     curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
                #     curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()

                #     # set the heading command to the closest direction
                #     self.heading_command_w[env_ids] = torch.where(
                #         curr_to_target < curr_to_flipped_target,
                #         target_direction,
                #         flipped_target_direction,
                #     )
                # else:
                #     # random heading command
                #     self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)
                
                if self.cfg.simple_heading:
                    # set heading command to point towards target
                    target_vec = self.pos_command_w[env_ids_to_be_checked] - self.robot.data.root_pos_w[env_ids_to_be_checked]
                    target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
                    flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

                    # compute errors to find the closest direction to the current heading
                    # this is done to avoid the discontinuity at the -pi/pi boundary
                    curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids_to_be_checked]).abs()
                    curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids_to_be_checked]).abs()

                    # set the heading command to the closest direction
                    self.heading_command_w[env_ids_to_be_checked] = torch.where(
                        curr_to_target < curr_to_flipped_target,
                        target_direction,
                        flipped_target_direction,
                    )
                else:
                    # random heading command
                    self.heading_command_w[env_ids_to_be_checked] = r.uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        """Re-target the position command to the current root state."""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)
        if not hasattr(self, 'temp_target_vec'):
            self.temp_target_vec = target_vec.clone()
            self.relative_movement = torch.zeros_like(self.temp_target_vec)
        else:
            self.distance_between_frame[:, 0] = torch.norm(self.temp_target_vec[:, :2], dim=1) - torch.norm(target_vec, dim=1) # distance in prev - distance in current > 0 is better
            self.relative_movement = -(target_vec - self.temp_target_vec)
            self.temp_target_vec = target_vec.clone()

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the box marker
        self.goal_pose_visualizer.visualize(
            translations=self.pos_command_w,
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w),
                torch.zeros_like(self.heading_command_w),
                self.heading_command_w,
            ),
        )


class TerrainBasedPose2dCommand(UniformPose2dCommand):
    """Command generator that generates pose commands based on the terrain.

    This command generator samples the position commands from the valid patches of the terrain.
    The heading commands are either set to point towards the target or are sampled uniformly.

    It expects the terrain to have a valid flat patches under the key 'target'.
    """

    cfg: TerrainBasedPose2dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TerrainBasedPose2dCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the terrain asset
        self.terrain: TerrainImporter = env.scene["terrain"]

        # obtain the valid targets from the terrain
        if "target" not in self.terrain.flat_patches:
            raise RuntimeError(
                "The terrain-based command generator requires a valid flat patch under 'target' in the terrain."
                f" Found: {list(self.terrain.flat_patches.keys())}"
            )
        # valid targets: (terrain_level, terrain_type, num_patches, 3)
        self.valid_targets: torch.Tensor = self.terrain.flat_patches["target"]

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new position targets from the terrain
        ids = torch.randint(0, self.valid_targets.shape[2], size=(len(env_ids),), device=self.device)
        self.pos_command_w[env_ids] = self.valid_targets[
            self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids], ids
        ]
        # offset the position command by the current root height
        self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            # compute errors to find the closest direction to the current heading
            # this is done to avoid the discontinuity at the -pi/pi boundary
            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()

            # set the heading command to the closest direction
            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            # random heading command
            r = torch.empty(len(env_ids), device=self.device)
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)
