# Copyright (c) 2022-2025, The UrbanSim Project Developers.
# Author: Honglin He
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
import random
from collections.abc import Sequence
import trimesh
from shapely.geometry import Polygon, MultiPolygon, Point
from typing import Any
import os
import copy
import json
from dataclasses import MISSING

import carb
import omni.usd
from pxr import Gf, Sdf, Semantics, Usd, UsdGeom, Vt
from pxr import UsdPhysics, PhysxSchema, UsdShade
from pxr import UsdSkel
from isaacsim.core.cloner import GridCloner
from isaacsim.core.prims import XFormPrim
import isaacsim.core.utils.stage as stage_utils
import isaacsim.core.utils.prims as prims_utils
from pxr import PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.utils import Timer, configclass
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg, TerrainGeneratorCfg
from isaacsim.core.utils.stage import add_reference_to_stage
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObject,
    DeformableObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.scene.interactive_scene import InteractiveScene
from isaaclab.scene.interactive_scene_cfg import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, SensorBase, SensorBaseCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

from urbansim.utils.geometry import (construct_lane, 
                                    construct_continuous_line_polygon, 
                                    construct_continuous_polygon, 
                                    construct_broken_line_polygon,
                                    generate_random_road,
                                    get_road_trimesh)

from .urban_scene_cfg import UrbanSceneCfg
from .utils import *

from urbansim.assets.rigid_object import RigidObject
from urbansim.assets.rigid_object_cfg import RigidObjectCfg

URBANSIM_PATH = os.environ.get('URBANSIM_PATH', './')

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR, check_file_path, read_file

LANE_HEIGHT = 0.2

def spawn_pg_humans(
    prim_path: str,
    cfg: DiversHumanCfg,
) -> Usd.Prim:
    # stage = omni.usd.get_context().get_stage()
    stage = stage_utils.get_current_stage()
    # convert orientation ordering (wxyz to xyzw)
    # manually clone prims if the source prim path is a regex expression
    with Sdf.ChangeBlock():
        for i, prim_path_group in enumerate(cfg.assets_cfg):
            # spawn single instance
            prim_path = prim_path_group[1]
            env_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
            proto_path = prim_path_group[0]
            env_spec.inheritPathList.Prepend(Sdf.Path(proto_path))
            Sdf.CopySpec(env_spec.layer, Sdf.Path(proto_path), env_spec.layer, Sdf.Path(prim_path))
            _ = UsdGeom.Xform(stage.GetPrimAtPath(proto_path)).GetPrim().GetPrimStack()
            translate_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:translate")
            if translate_spec is None:
                translate_spec = Sdf.AttributeSpec(env_spec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
            translate_spec.default = Gf.Vec3d(*prim_path_group[2])
            orient_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:orient")
            if orient_spec is None:
                orient_spec = Sdf.AttributeSpec(env_spec, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
            orient_spec.default = Gf.Quatf(prim_path_group[3][0], Gf.Vec3f(prim_path_group[3][1], prim_path_group[3][2], prim_path_group[3][3]))
            op_order_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
            if op_order_spec is None:
                op_order_spec = Sdf.AttributeSpec(env_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray)
            op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

    # delete the dataset prim after spawning
    prims_utils.delete_prim("/World/DatasetDynamic")

    # return the prim
    return cfg.assets_cfg

def spawn_pg_objects(
    prim_path: str,
    cfg: DiverseAssetCfg,
) -> Usd.Prim:
    # stage = omni.usd.get_context().get_stage()
    stage = stage_utils.get_current_stage()
    # convert orientation ordering (wxyz to xyzw)
    # manually clone prims if the source prim path is a regex expression
    with Sdf.ChangeBlock():
        for i, prim_path_group in enumerate(cfg.assets_cfg):
            # spawn single instance
            prim_path = prim_path_group[0]
            env_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
            proto_path = '/World/Dataset/' + prim_path.split('/')[-1][:-4]
            env_spec.inheritPathList.Prepend(Sdf.Path(proto_path))
            Sdf.CopySpec(env_spec.layer, Sdf.Path(proto_path), env_spec.layer, Sdf.Path(prim_path))
            _ = UsdGeom.Xform(stage.GetPrimAtPath(proto_path)).GetPrim().GetPrimStack()
            translate_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:translate")
            if translate_spec is None:
                translate_spec = Sdf.AttributeSpec(env_spec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
            translate_spec.default = Gf.Vec3d(*prim_path_group[1])
            orient_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:orient")
            if orient_spec is None:
                orient_spec = Sdf.AttributeSpec(env_spec, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
            orient_spec.default = Gf.Quatd(*prim_path_group[2])
            op_order_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
            if op_order_spec is None:
                op_order_spec = Sdf.AttributeSpec(env_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray)
            op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

    # delete the dataset prim after spawning
    prims_utils.delete_prim("/World/Dataset")

    # return the prim
    return cfg.assets_cfg

# generation of random objects
@configclass
class DiverseAssetCfg(sim_utils.SpawnerCfg):
    func: sim_utils.SpawnerCfg.func = spawn_pg_objects
    assets_cfg: list = MISSING

# generation of random humans
@configclass
class DiversHumanCfg(sim_utils.SpawnerCfg):
    func: sim_utils.SpawnerCfg.func = spawn_pg_humans
    assets_cfg: list = MISSING

class UrbanScene(InteractiveScene):
    """
    A scene that contains entities added to the simulation.
    Note: some functions are dependent on metadrive/metaurban.
    """
    
    def __init__(self, cfg: UrbanSceneCfg):
        """Initializes the scene.

        Args:
            cfg: The configuration class for the scene.
        """
        # store inputs
        self.cfg = cfg
        # HACK: disable physics replication to support procedural generation for infinite scenes
        if self.cfg.scenario_generation_method is None:
            self.cfg.scenario_generation_method = "predefined"
            print(f'[INFO] scenario_generation_method is set to: {self.cfg.scenario_generation_method}')
        assert self.cfg.scenario_generation_method in ['async procedural generation', 
                                                       'sync procedural generation', 
                                                       'limited async procedural generation', 
                                                       'limited sync procedural generation', 
                                                       'predefined'], "Invalid scenario type not in ['async procedural generation', 'sync procedural generation', 'limited async procedural generation', 'limited sync procedural generation', 'predefined']"
        if "async" in self.cfg.scenario_generation_method:
            self.cfg.replicate_physics = False
            print(f'[NOTE] Make sure that the physics replication is disabled for async procedural generation.')
            print(f'[INFO] scenario_generation_method: {self.cfg.scenario_generation_method}')
            print(f'[INFO] replicate_physics: {self.cfg.replicate_physics}')
        
        # add deafult values
        self.procedural_generated_terrains = {}
        self.polygons_of_lane = [[] for _ in range(self.cfg.num_envs)]
        self.white_line_polygons = [[] for _ in range(self.cfg.num_envs)]
        self.yellow_line_polygons = [[] for _ in range(self.cfg.num_envs)]
        self.sidewalk_polygons = [[] for _ in range(self.cfg.num_envs)]
        self.near_road_sidewalk_polygons = [[] for _ in range(self.cfg.num_envs)]
        self.near_road_buffer_polygons = [[] for _ in range(self.cfg.num_envs)]
        self.far_road_sidewalk_polygons = [[] for _ in range(self.cfg.num_envs)]
        self.far_road_buffer_polygons = [[] for _ in range(self.cfg.num_envs)]
        self.house_region_polygons = [[] for _ in range(self.cfg.num_envs)]
        self.all_region_list = []
        self.walkable_terrain_list = []
        
        # unique static assets
        self.unique_static_asset_path = []
        if self.cfg.static_assets_path is None:
            self.cfg.static_assets_path = f"{URBANSIM_PATH}/assets/usds/"
            print(f'[INFO] static_assets_path: {self.cfg.static_assets_path}')
        all_assets = os.listdir(self.cfg.static_assets_path)
        all_assets = [asset for asset in all_assets if asset.endswith('.usd') and 'non_metric' not in asset]
        self.unique_static_asset_path = [os.path.join(self.cfg.static_assets_path, asset) for asset in all_assets]
        self.unique_static_asset_path = self.unique_static_asset_path[:min(len(self.unique_static_asset_path), self.cfg.maximum_static_object_number)]
        print(f'[INFO] number of unique static assets: {len(self.unique_static_asset_path)}')
        print('[INFO] example of unique static asset path:', self.unique_static_asset_path[0])
        
        # unique dynamic assets
        self.unique_dynamic_asset_path = []
        if self.cfg.dynamic_assets_path is None:
            self.cfg.dynamic_assets_path = f"{URBANSIM_PATH}/assets/peds/"
            print(f'[INFO] dynamic_assets_path: {self.cfg.dynamic_assets_path}')
        all_assets = os.listdir(self.cfg.dynamic_assets_path)
        self.unique_dynamic_asset_path = [os.path.join(self.cfg.dynamic_assets_path, asset, asset.split('_')[-1] + '.usd') for asset in all_assets]
        self.unique_dynamic_asset_path = self.unique_dynamic_asset_path[:min(len(self.unique_dynamic_asset_path), self.cfg.maximum_dynamic_object_number)]
        print(f'[INFO] number of unique dynamic assets: {len(self.unique_dynamic_asset_path)}')
        print('[INFO] example of unique dynamic asset path:', self.unique_dynamic_asset_path[0])
        
        super(UrbanScene, self).__init__(self.cfg)
        self.terrain_offsets = self._default_env_origins.cpu().numpy()
        self.use_dynamic_pedestrians = False
        print(f'[INFO] use_dynamic_pedestrians: {self.use_dynamic_pedestrians}')
        self.count = 0
    
    def _add_entities_from_cfg(self, procedural_generation=False):
        """Add scene entities from the config."""
        # store paths that are in global collision filter
        self._global_prim_paths = list()
        # parse the entire scene config and resolve regex
        for asset_name, asset_cfg in self.cfg.__dict__.items():
            # skip keywords
            # note: easier than writing a list of keywords: [num_envs, env_spacing, lazy_sensor_update]
            if asset_name in InteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None:
                continue
            
            if isinstance(asset_cfg, TerrainGeneratorCfg):
                continue
            if isinstance(asset_cfg, sim_utils.PreviewSurfaceCfg):
                continue
            if isinstance(asset_cfg, sim_utils.MdlFileCfg):
                continue

            if not hasattr(self, 'only_create_assets'):
                self.only_create_assets = False
            # for procedural generation
            if isinstance(asset_cfg, list):
                iter_n = 0
                for sub_cfg in asset_cfg:
                    if isinstance(sub_cfg, dict):
                        for sub_cfg_key, sub_cfg_value in sub_cfg.items():
                            if isinstance(sub_cfg_value, RigidObjectCfg):
                                sub_cfg_value.prim_path = sub_cfg_value.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                                self._rigid_objects[sub_cfg_key] = sub_cfg_value.class_type(sub_cfg_value)
                            if hasattr(sub_cfg_value, "collision_group") and sub_cfg_value.collision_group == -1:
                                asset_paths = sim_utils.find_matching_prim_paths(sub_cfg_value.prim_path)
                                self._global_prim_paths += asset_paths
                    elif isinstance(sub_cfg, TerrainImporterCfg):
                        sub_cfg.prim_path = sub_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                        if hasattr(sub_cfg, "collision_group") and sub_cfg.collision_group == -1:
                            asset_paths = sim_utils.find_matching_prim_paths(sub_cfg.prim_path)
                            self._global_prim_paths += asset_paths
                        if asset_name == 'terrain_non_walkable_list':
                            sub_cfg.num_envs = self.cfg.num_envs
                            sub_cfg.env_spacing = self.cfg.env_spacing
                            self.all_region_list.append(sub_cfg.class_type(sub_cfg))
                        else:
                            sub_cfg.num_envs = self.cfg.num_envs
                            sub_cfg.env_spacing = self.cfg.env_spacing
                            self.walkable_terrain_list.append(sub_cfg.class_type(sub_cfg))
                    elif isinstance(sub_cfg, sim_utils.MdlFileCfg):
                        sub_cfg.func(f'/World/Looks/{asset_name}_{iter_n:03d}', sub_cfg)
                    elif isinstance(sub_cfg, RigidObjectCfg):
                        # sub_cfg.init_state.pos = list(sub_cfg.init_state.pos)
                        # sub_cfg.init_state.pos[2] = 0.0
                        # sub_cfg.init_state.pos = tuple(sub_cfg.init_state.pos)
                        sub_cfg.prim_path = sub_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                        self._rigid_objects[f'{asset_name}_{iter_n}'] = sub_cfg.class_type(sub_cfg)
                        if hasattr(sub_cfg, "collision_group") and sub_cfg.collision_group == -1:
                            asset_paths = sim_utils.find_matching_prim_paths(sub_cfg.prim_path)
                            self._global_prim_paths += asset_paths
                    else:
                        pass
                    iter_n += 1
                continue
            try:
                asset_cfg.prim_path = asset_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
            except:
                setattr(self, asset_name, asset_cfg)
                continue
            
            if procedural_generation:
                if asset_name in InteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None:
                    continue
                asset_cfg.prim_path = asset_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                if hasattr(asset_cfg, "collision_group") and asset_cfg.collision_group == -1:
                    asset_paths = sim_utils.find_matching_prim_paths(asset_cfg.prim_path)
                    self._global_prim_paths += asset_paths
                if isinstance(asset_cfg, TerrainImporterCfg):
                    # terrains are special entities since they define environment origins
                    asset_cfg.num_envs = self.cfg.num_envs
                    asset_cfg.env_spacing = self.cfg.env_spacing
                    # asset_cfg.prim_path = asset_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                    if asset_name.startswith('procedural_generation_'):
                        self.procedural_generated_terrains[asset_name] = asset_cfg.class_type(asset_cfg)
                        if 'lane' in asset_name:
                            for idx, center in enumerate(self.terrain_offsets):
                                mesh_list = []
                                for po_idx, polygon in enumerate(self.polygons_of_lane[idx]):
                                    polygon = np.array(polygon).reshape(len(polygon), 2)
                                    lane_mesh = trimesh.creation.extrude_polygon(polygon=Polygon(np.stack([polygon[..., 0] + center[0], polygon[..., 1] + center[1]], axis=1)), height=LANE_HEIGHT, engine="triangle")
                                    mesh_list.append(lane_mesh)
                                combined_mesh = trimesh.util.concatenate(mesh_list)
                                uvs = []
                                for vertex in combined_mesh.vertices:
                                    uv = [vertex[0], vertex[1]]
                                    uvs.append(uv)
                                uvs = np.array(uvs)
                                uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
                                uvs *= 20
                                combined_mesh.visual.uvs = uvs
                                self.procedural_generated_terrains[asset_name].import_mesh(f'lane_{idx:04d}', combined_mesh)
                        if 'whiteline' in asset_name:
                            for idx, center in enumerate(self.terrain_offsets):
                                mesh_list = []
                                for po_idx, polygon in enumerate(self.white_line_polygons[idx]):
                                    polygon = np.array(polygon).reshape(len(polygon), 2)
                                    lane_mesh = trimesh.creation.extrude_polygon(polygon=Polygon(np.stack([polygon[..., 0] + center[0], polygon[..., 1] + center[1]], axis=1)), height=LANE_HEIGHT + 0.01, engine="triangle")
                                    mesh_list.append(lane_mesh)
                                combined_mesh = trimesh.util.concatenate(mesh_list)
                                self.procedural_generated_terrains[asset_name].import_mesh(f'whiteline_{idx:04d}', combined_mesh)
                        if 'yellowline' in asset_name:
                            for idx, center in enumerate(self.terrain_offsets):
                                mesh_list = []
                                for po_idx, polygon in enumerate(self.yellow_line_polygons[idx]):
                                    polygon = np.array(polygon).reshape(len(polygon), 2)
                                    lane_mesh = trimesh.creation.extrude_polygon(polygon=Polygon(np.stack([polygon[..., 0] + center[0], polygon[..., 1] + center[1]], axis=1)), height=LANE_HEIGHT + 0.01, engine="triangle")
                                    mesh_list.append(lane_mesh)
                                combined_mesh = trimesh.util.concatenate(mesh_list)
                                self.procedural_generated_terrains[asset_name].import_mesh(f'yellowline_{idx:04d}', combined_mesh)
                                    
                        if 'sidewalk' in asset_name:
                            for idx, center in enumerate(self.terrain_offsets):
                                mesh_list = []
                                for po_idx, polygon in enumerate(self.s_polygons[idx]):
                                    if len(polygon) < 3:
                                        continue
                                    polygon = np.array(polygon).reshape(len(polygon), 2)
                                    polygon = Polygon(np.stack([polygon[..., 0] + center[0], polygon[..., 1] + center[1]], axis=1))
                                    lane_mesh = trimesh.creation.extrude_polygon(polygon=polygon, height=LANE_HEIGHT + 0.02)
                                    uv = []
                                    mesh = lane_mesh
                                    for vertex in lane_mesh.vertices:
                                        u = (vertex[0] - np.min(mesh.vertices[:, 0])) / (np.max(mesh.vertices[:, 0]) - np.min(mesh.vertices[:, 0]))
                                        v = (vertex[1] - np.min(mesh.vertices[:, 1])) / (np.max(mesh.vertices[:, 1]) - np.min(mesh.vertices[:, 1]))
                                        uv.append([u, v])
                                    uv = np.array(uv)

                                    lane_mesh.visual.uv = uv
                                    mesh_list.append(lane_mesh)
                                if len(mesh_list) == 0:
                                    continue
                                combined_mesh = trimesh.util.concatenate(mesh_list)
                                uvs = []
                                for vertex in combined_mesh.vertices:
                                    uv = [vertex[0], vertex[1]]
                                    uvs.append(uv)
                                uvs = np.array(uvs)
                                uvs[:, 0] = (uvs[:, 0] - np.min(uvs[:, 0])) / (np.max(uvs[:, 0]) - np.min(uvs[:, 0]))
                                uvs[:, 1] = (uvs[:, 1] - np.min(uvs[:, 1])) / (np.max(uvs[:, 1]) - np.min(uvs[:, 1]))
                                uvs *= 40
                                combined_mesh.visual.uvs = uvs
                                self.procedural_generated_terrains[asset_name].import_mesh(f'sidewalk_{idx:04d}', combined_mesh)
                        if 'near_road' in asset_name:
                            for idx, center in enumerate(self.terrain_offsets):
                                mesh_list = []
                                for po_idx, polygon in enumerate(self.n_polygons[idx]):
                                    if len(polygon) < 3:
                                        continue
                                    polygon = np.array(polygon).reshape(len(polygon), 2)
                                    lane_mesh = trimesh.creation.extrude_polygon(polygon=Polygon(np.stack([polygon[..., 0] + center[0], polygon[..., 1] + center[1]], axis=1)), height=LANE_HEIGHT + 0.02, engine="triangle")
                                    mesh_list.append(lane_mesh)
                                if len(mesh_list) == 0:
                                    continue
                                combined_mesh = trimesh.util.concatenate(mesh_list)
                                uvs = []
                                for vertex in combined_mesh.vertices:
                                    uv = [vertex[0], vertex[1]]
                                    uvs.append(uv)
                                if len(uvs) > 0:
                                    uvs = np.array(uvs)
                                    uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
                                    uvs *= 80
                                    combined_mesh.visual.uvs = uvs
                                self.procedural_generated_terrains[asset_name].import_mesh(f'nearroad_{idx:04d}', combined_mesh)
                        if 'near_buffer' in asset_name:
                            for idx, center in enumerate(self.terrain_offsets):
                                mesh_list = []
                                for po_idx, polygon in enumerate(self.nb_polygons[idx]):
                                    if len(polygon) < 3:
                                        continue
                                    polygon = np.array(polygon).reshape(len(polygon), 2)
                                    lane_mesh = trimesh.creation.extrude_polygon(polygon=Polygon(np.stack([polygon[..., 0] + center[0], polygon[..., 1] + center[1]], axis=1)), height=LANE_HEIGHT + 0.02, engine="triangle")
                                    mesh_list.append(lane_mesh)
                                if len(mesh_list) == 0:
                                    continue
                                combined_mesh = trimesh.util.concatenate(mesh_list)
                                uvs = []
                                for vertex in combined_mesh.vertices:
                                    uv = [vertex[0], vertex[1]]
                                    uvs.append(uv)
                                if len(uvs) > 0:
                                    uvs = np.array(uvs)
                                    uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
                                    uvs *= 80
                                    combined_mesh.visual.uvs = uvs
                                self.procedural_generated_terrains[asset_name].import_mesh(f'nearbuffer_{idx:04d}', combined_mesh)
                        if 'farfrom_road' in asset_name:
                            for idx, center in enumerate(self.terrain_offsets):
                                mesh_list = []
                                for po_idx, polygon in enumerate(self.f_polygons[idx]):
                                    if len(polygon) < 3:
                                        continue
                                    polygon = np.array(polygon).reshape(len(polygon), 2)
                                    lane_mesh = trimesh.creation.extrude_polygon(polygon=Polygon(np.stack([polygon[..., 0] + center[0], polygon[..., 1] + center[1]], axis=1)), height=LANE_HEIGHT + 0.02, engine="triangle")
                                    mesh_list.append(lane_mesh)
                                if len(mesh_list) == 0:
                                    continue
                                combined_mesh = trimesh.util.concatenate(mesh_list)
                                uvs = []
                                for vertex in combined_mesh.vertices:
                                    uv = [vertex[0], vertex[1]]
                                    uvs.append(uv)
                                if len(uvs) > 0:
                                    uvs = np.array(uvs)
                                    uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
                                    uvs *= 80
                                    combined_mesh.visual.uvs = uvs
                                self.procedural_generated_terrains[asset_name].import_mesh(f'farfromroad_{idx:04d}', combined_mesh)
                        if 'farfrom_buffer' in asset_name:
                            for idx, center in enumerate(self.terrain_offsets):
                                mesh_list = []
                                for po_idx, polygon in enumerate(self.fb_polygons[idx]):
                                    if len(polygon) < 3:
                                        continue
                                    polygon = np.array(polygon).reshape(len(polygon), 2)
                                    lane_mesh = trimesh.creation.extrude_polygon(polygon=Polygon(np.stack([polygon[..., 0] + center[0], polygon[..., 1] + center[1]], axis=1)), height=LANE_HEIGHT + 0.02, engine="triangle")
                                    mesh_list.append(lane_mesh)
                                if len(mesh_list) == 0:
                                    continue
                                combined_mesh = trimesh.util.concatenate(mesh_list)
                                uvs = []
                                for vertex in combined_mesh.vertices:
                                    uv = [vertex[0], vertex[1]]
                                    uvs.append(uv)
                                if len(uvs) > 0:
                                    uvs = np.array(uvs)
                                    uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
                                    uvs *= 80
                                    combined_mesh.visual.uvs = uvs
                                self.procedural_generated_terrains[asset_name].import_mesh(f'farfrombuffer_{idx:04d}', combined_mesh)
                        if 'house' in asset_name:
                            for idx, center in enumerate(self.terrain_offsets):
                                mesh_list = []
                                for po_idx, polygon in enumerate(self.h_polygons[idx]):
                                    if len(polygon) < 3:
                                        continue
                                    polygon = np.array(polygon).reshape(len(polygon), 2)
                                    lane_mesh = trimesh.creation.extrude_polygon(polygon=Polygon(np.stack([polygon[..., 0] + center[0], polygon[..., 1] + center[1]], axis=1)), height=LANE_HEIGHT + 0.02 if po_idx < len(self.h_polygons[idx]) -1 else LANE_HEIGHT + 0.02)
                                    mesh_list.append(lane_mesh)
                                if len(mesh_list) == 0:
                                    continue
                                combined_mesh = trimesh.util.concatenate(mesh_list)
                                uvs = []
                                for vertex in combined_mesh.vertices:
                                    uv = [vertex[0], vertex[1]]
                                    uvs.append(uv)
                                uvs = np.array(uvs)
                                uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
                                uvs *= 40
                                combined_mesh.visual.uvs = uvs
                                self.procedural_generated_terrains[asset_name].import_mesh(f'house_region_{idx:04d}_{po_idx:04d}', combined_mesh)
                    else:
                        self._terrain = asset_cfg.class_type(asset_cfg)
                elif isinstance(asset_cfg, RigidObjectCfg):
                    if asset_name not in list(self._articulations.keys()):
                        self._rigid_objects[asset_name] = asset_cfg.class_type(asset_cfg)
            else:
                # resolve regex
                if hasattr(asset_cfg, "prim_path"):
                    asset_cfg.prim_path = asset_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                # create asset
                if isinstance(asset_cfg, TerrainImporterCfg):
                    # terrains are special entities since they define environment origins
                    asset_cfg.num_envs = self.cfg.num_envs
                    asset_cfg.env_spacing = self.cfg.env_spacing
                    self._terrain = asset_cfg.class_type(asset_cfg)
                elif isinstance(asset_cfg, ArticulationCfg):
                    self._articulations[asset_name] = asset_cfg.class_type(asset_cfg)
                elif isinstance(asset_cfg, DeformableObjectCfg):
                    self._deformable_objects[asset_name] = asset_cfg.class_type(asset_cfg)
                elif isinstance(asset_cfg, RigidObjectCfg):
                    self._rigid_objects[asset_name] = asset_cfg.class_type(asset_cfg)
                elif isinstance(asset_cfg, RigidObjectCollectionCfg):
                    for rigid_object_cfg in asset_cfg.rigid_objects.values():
                        rigid_object_cfg.prim_path = rigid_object_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                    self._rigid_object_collections[asset_name] = asset_cfg.class_type(asset_cfg)
                    for rigid_object_cfg in asset_cfg.rigid_objects.values():
                        if hasattr(rigid_object_cfg, "collision_group") and rigid_object_cfg.collision_group == -1:
                            asset_paths = sim_utils.find_matching_prim_paths(rigid_object_cfg.prim_path)
                            self._global_prim_paths += asset_paths
                elif isinstance(asset_cfg, SensorBaseCfg):
                    # Update target frame path(s)' regex name space for FrameTransformer
                    if isinstance(asset_cfg, FrameTransformerCfg):
                        updated_target_frames = []
                        for target_frame in asset_cfg.target_frames:
                            target_frame.prim_path = target_frame.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                            updated_target_frames.append(target_frame)
                        asset_cfg.target_frames = updated_target_frames
                    elif isinstance(asset_cfg, ContactSensorCfg):
                        updated_filter_prim_paths_expr = []
                        for filter_prim_path in asset_cfg.filter_prim_paths_expr:
                            updated_filter_prim_paths_expr.append(filter_prim_path.format(ENV_REGEX_NS=self.env_regex_ns))
                        asset_cfg.filter_prim_paths_expr = updated_filter_prim_paths_expr

                    self._sensors[asset_name] = asset_cfg.class_type(asset_cfg)
                elif isinstance(asset_cfg, AssetBaseCfg):
                    # manually spawn asset
                    if asset_cfg.spawn is not None:
                        asset_cfg.spawn.func(
                            asset_cfg.prim_path,
                            asset_cfg.spawn,
                            translation=asset_cfg.init_state.pos,
                            orientation=asset_cfg.init_state.rot,
                        )
                    # store xform prim view corresponding to this asset
                    # all prims in the scene are Xform prims (i.e. have a transform component)
                    self._extras[asset_name] = XFormPrim(asset_cfg.prim_path, reset_xform_properties=False)
                elif isinstance(asset_cfg, str) or isinstance(asset_cfg, int) or isinstance(asset_cfg, float) or isinstance(asset_cfg, dict) or isinstance(asset_cfg, bool):
                    print(f"[Info] Skipping {asset_name} with value {asset_cfg}")
                else:
                    raise ValueError(f"Unknown asset config type for {asset_name}: {asset_cfg}")
                # store global collision paths
                if hasattr(asset_cfg, "collision_group") and asset_cfg.collision_group == -1:
                    asset_paths = sim_utils.find_matching_prim_paths(asset_cfg.prim_path)
                    self._global_prim_paths += asset_paths
                
    def reset(self, env_ids: Sequence[int] | None = None):
        
        super(UrbanScene, self).reset(env_ids)
        
        # self.count = 0
        # print(f'[INFO] UrbanScene reset called, count set to {self.count}')

    def update(self, dt):
        
        super(UrbanScene, self).update(dt)
        
    def _remove_entities_from_cfg(self):
        """Remove scene entities from the config."""
        # parse the entire scene config and resolve regex
        for asset_name, asset_cfg in self.cfg.__dict__.items():
            # for procedural generation
            if isinstance(asset_cfg, list):
                for sub_cfg in asset_cfg:
                    for sub_cfg_key, sub_cfg_value in sub_cfg.items():
                        if isinstance(sub_cfg_value, RigidObjectCfg):
                            sub_cfg_value.prim_path = sub_cfg_value.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                            prims_utils.delete_prim(sub_cfg_value.prim_path)
                continue
            if isinstance(asset_cfg, TerrainImporterCfg):
                prims_utils.delete_prim(asset_cfg.prim_path)
    def write_data_to_sim(self):
        super().write_data_to_sim()
        if self.use_dynamic_pedestrians:
            self.write_dynamic_asset_state_to_sim()
    
    def write_dynamic_asset_state_to_sim(self):
        from isaacsim.core.prims import XFormPrim
        if not hasattr(self, "ped_step_index"):
            self.ped_step_index = 0
            self.ped_direction = 1
            print(f'[INFO] Pedestrian step index initialized to {self.ped_step_index}: Forward')
            self.human_prim_expr = XFormPrim(f'/World/envs/env_.*/Dynamic_*')

        if self.count % self.cfg.pg_config.get('ped_forward_inteval', 10) == 0:
            self.ped_step_index += self.ped_direction
            moving_max_t = self.cfg.pg_config.get('moving_max_t', 10)
            if self.ped_step_index >= moving_max_t:
                self.ped_step_index = moving_max_t
                self.ped_direction = -1  
            elif self.ped_step_index <= 0:
                self.ped_step_index = 0
                self.ped_direction = 1
            
            if self.ped_direction == 1:
                start_xy = [xyzs[0] for xyzs in self.pedestrian_forward_start_end_pos_list]
                end_xy = [xyzs[1] for xyzs in self.pedestrian_forward_start_end_pos_list]
                heading = [wxyzs[0] for wxyzs in self.pedestrian_forward_backward_heading_list]
            elif self.ped_direction == -1:
                start_xy = [xyzs[0] for xyzs in self.pedestrian_forward_start_end_pos_list]
                end_xy = [xyzs[1] for xyzs in self.pedestrian_forward_start_end_pos_list]
                heading = [wxyzs[1] for wxyzs in self.pedestrian_forward_backward_heading_list]
            start_xy_tensor = torch.tensor(start_xy, device=self.device, dtype=torch.float32).reshape(-1, 3)
            end_xy_tensor = torch.tensor(end_xy, device=self.device, dtype=torch.float32).reshape(-1, 3)
            wxyz_tensor = torch.tensor(heading, device=self.device, dtype=torch.float32).reshape(-1, 4)
            current_xy_tensor = start_xy_tensor + (end_xy_tensor - start_xy_tensor) * (self.ped_step_index / moving_max_t)
            current_xy_tensor = current_xy_tensor.reshape(self.num_envs, -1, 3) + self._default_env_origins[..., :3].reshape(self.num_envs, 1, 3)
            current_xy_tensor = current_xy_tensor.reshape(-1, 3)
            self.human_prim_expr.set_world_poses(
                positions=current_xy_tensor,
                orientations=wxyz_tensor,
            )
        
        self.count += 1
        
    def dynamic_asset_animatable_state(self):
        add_reference_to_stage(usd_path="assets/ped_actions/synbody_idle426.fbx/synbody_idle426.fbx.usd", prim_path="/World/run")
        add_reference_to_stage(usd_path="assets/ped_actions/synbody_walking426.fbx/synbody_walking426.fbx.usd", prim_path="/World/walk")
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_start_time(0)
        timeline.set_end_time(1.1)
        timeline.set_target_framerate(40)
        timeline.set_looping(True)
        
    def generate_scene(self, remove_current_scene: bool = False):
        """
        Args:
            remove_current_scene (bool, optional): _description_. Defaults to False.
        """
        if self.cfg.scenario_generation_method in ['async procedural generation', 'sync procedural generation']:
            assert hasattr(self.cfg, 'pg_config'), "pg_config is required for procedural generation."
            print(f'[INFO] configuration for procedural generation: {self.cfg.pg_config}')
        
        if self.cfg.scenario_generation_method == "predefined":
            self.generate_predefined_scene()
        elif self.cfg.scenario_generation_method == "async procedural generation":
            self.generate_async_procedural_scene()
        elif self.cfg.scenario_generation_method == "sync procedural generation":
            self.generate_sync_procedural_scene()
        elif self.cfg.scenario_generation_method == "limited async procedural generation":
            self.only_create_assets = False
            self.generate_limited_async_procedural_scene()
        elif self.cfg.scenario_generation_method == "limited sync procedural generation":
            self.generate_limited_sync_procedural_scene()
        else:
            raise ValueError(f"Invalid method: {self.cfg.scenario_generation_method}")
    
    def generate_predefined_scene(self):
        return
    
    def generate_limited_sync_procedural_scene(self):
        generation_cfg = self.cfg.pg_config
        area_size = generation_cfg['map_region']
        torch.manual_seed(generation_cfg['seed'])
        np.random.seed(generation_cfg['seed'])
        random.seed(generation_cfg['seed'])
        # ground plane
        tmp_origin = self._default_env_origins[..., :2].reshape(self.num_envs, 2).cpu().numpy()
        mesh_block_height = 0.01
        if generation_cfg['type'] == 'clean' or \
           generation_cfg['type'] == 'static' or \
           generation_cfg['type'] == 'dynamic':
            area_size = [area_size, area_size]
            
            polygon_points = []
            walkable_region_polygon_list = [[] for _ in range(len(self.cfg.terrain_importer_walkable_list))]
            all_region_polygon_list = [[] for _ in range(len(self.cfg.terrain_non_walkable_list))]
            polyline_points_list = []
            
            for env_idx in range(self.num_envs):
                buffer_width = generation_cfg['buffer_width']
                x, y = generate_random_road(area_size)
                x, y = np.array(x), np.array(y)
                
                x += tmp_origin[env_idx, 0]
                y += tmp_origin[env_idx, 1]
                x = x.clip(tmp_origin[env_idx, 0], tmp_origin[env_idx, 0] + area_size[0])
                y = y.clip(tmp_origin[env_idx, 1], tmp_origin[env_idx, 1] + area_size[1])
                
                # walkable regions
                mesh, boundary_points, polyline_points = get_road_trimesh(x, y, area_size, boundary=(tmp_origin[env_idx, 1], tmp_origin[env_idx, 1] + area_size[1]), height=mesh_block_height+0.02)
                polyline_points_list.append(polyline_points)
                polygon_points.append([x, y, boundary_points[0], boundary_points[1]])
                
                # non walkable regions
                area_polygon = np.array([(0, 0), (0, area_size[1] + buffer_width), (area_size[0] + buffer_width, area_size[1] + buffer_width), (area_size[0] + buffer_width, 0)]).astype(float)
                area_polygon[:, 0] += tmp_origin[env_idx, 0]
                area_polygon[:, 1] += tmp_origin[env_idx, 1]
                area_mesh = trimesh.creation.extrude_polygon(Polygon(area_polygon), height=mesh_block_height)
                
                # boundary polygons
                if generation_cfg.get('with_boundary', False):
                    boundary = np.array([(0, -0.5), (0, 0.0), (area_size[0] + 0.5, 0.0), (area_size[0] + 0.5, -0.5)]).astype(float)
                    boundary[:, 0] += tmp_origin[env_idx, 0]
                    boundary[:, 1] += tmp_origin[env_idx, 1]
                    boundary_mesh = trimesh.creation.extrude_polygon(Polygon(boundary), height=mesh_block_height * 10)
                    all_region_polygon_list[generation_cfg['non_walkable_seed']].append(boundary_mesh)
                    boundary = np.array([(0, area_size[1]), (0, area_size[1] + 0.5), (area_size[0] + 0.5, area_size[1] + 0.5), (area_size[0] + 0.5, area_size[1])]).astype(float)
                    boundary[:, 0] += tmp_origin[env_idx, 0]
                    boundary[:, 1] += tmp_origin[env_idx, 1]
                    boundary_mesh = trimesh.creation.extrude_polygon(Polygon(boundary), height=mesh_block_height * 10)
                    all_region_polygon_list[generation_cfg['non_walkable_seed']].append(boundary_mesh)
                    
                    boundary = np.array([(-0.5, 0.0), (0, 0.0), (0.0, area_size[1] + 0.5), (-0.5, area_size[1] + 0.5)]).astype(float)
                    boundary[:, 0] += tmp_origin[env_idx, 0]
                    boundary[:, 1] += tmp_origin[env_idx, 1]
                    boundary_mesh = trimesh.creation.extrude_polygon(Polygon(boundary), height=mesh_block_height * 10)
                    all_region_polygon_list[generation_cfg['non_walkable_seed']].append(boundary_mesh)
                    boundary = np.array([(area_size[0], 0.0), (area_size[0] + 0.5, 0.0), (area_size[0] + 0.5, area_size[1] + 0.5), (area_size[0], area_size[1] + 0.5)]).astype(float)
                    boundary[:, 0] += tmp_origin[env_idx, 0]
                    boundary[:, 1] += tmp_origin[env_idx, 1]
                    boundary_mesh = trimesh.creation.extrude_polygon(Polygon(boundary), height=mesh_block_height * 10)
                    all_region_polygon_list[generation_cfg['non_walkable_seed']].append(boundary_mesh)

                walkable_region_polygon_list[generation_cfg['walkable_seed']].append(mesh)
                all_region_polygon_list[generation_cfg['non_walkable_seed']].append(area_mesh)

            self.polygon_points = polygon_points
            self.polylines_of_all_walkable_regions = torch.from_numpy(np.stack(polyline_points_list)).float().to(self.device)
            for i in range(len(walkable_region_polygon_list)):
                mesh_list = walkable_region_polygon_list[i]
                if len(mesh_list) == 0:
                    continue
                combined_mesh = trimesh.util.concatenate(mesh_list)
                
                # uv for texturing
                combined_mesh = uv_texturing(combined_mesh, scale=UV_SCLAE)
                self.walkable_terrain_list[i].import_mesh('mesh', combined_mesh)
                sim_utils.bind_visual_material(f'/World/Walkable_{i:03d}', f'/World/Looks/terrain_walkable_material_list_{i:03d}')
                sim_utils.bind_physics_material(f'/World/Walkable_{i:03d}', f'/World/Looks/terrain_non_walkable_material_list_{i:03d}')
                stage = omni.usd.get_context().get_stage()
                prim = stage.GetPrimAtPath(Sdf.Path(f'/World/Walkable_{i:03d}/Environment'))
                prim.SetActive(False)
                
            for i in range(len(all_region_polygon_list)):
                mesh_list = all_region_polygon_list[i]
                if len(mesh_list) == 0:
                    continue
                combined_mesh = trimesh.util.concatenate(mesh_list)
                
                # uv for texturing
                combined_mesh = uv_texturing(combined_mesh, scale=UV_SCLAE)
                self.all_region_list[i].import_mesh('mesh', combined_mesh)
                sim_utils.bind_visual_material(f'/World/NonWalkable_{i:03d}', f'/World/Looks/terrain_non_walkable_material_list_{i:03d}')
                sim_utils.bind_physics_material(f'/World/NonWalkable_{i:03d}', f'/World/Looks/terrain_non_walkable_material_list_{i:03d}')
                prim = stage.GetPrimAtPath(Sdf.Path(f'/World/NonWalkable_{i:03d}/Environment'))
                prim.SetActive(False)
            
            # Remove ground plane
            for i in range(max(len(self.all_region_list), len(self.walkable_terrain_list))):
                try:
                    prim = stage.GetPrimAtPath(Sdf.Path(f'/World/NonWalkable_{i:03d}/Environment'))
                    prim.SetActive(False)
                except:
                    pass
                
                try:
                    prim = stage.GetPrimAtPath(Sdf.Path(f'/World/Walkable_{i:03d}/Environment'))
                    prim.SetActive(False)
                except:
                    pass
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/ground/Environment'))
            prim.SetActive(False)
            
        # static objects
        if generation_cfg['type'] == 'static' or  generation_cfg['type'] == 'dynamic':
            num_objects = generation_cfg['num_object']
            # register dataset path
            prims_utils.create_prim("/World/Dataset", "Scope")
            proto_prim_paths = list()
            # get all assets from dataset
            asset_root_path = 'assets/usds/'
            all_assets = os.listdir(asset_root_path)
            all_assets = [i for i in all_assets if 'non_metric' not in i]
            import yaml
            with open('./assets/asset_config.yaml', "r") as file:
                asset_config = yaml.safe_load(file)
            asset_types = asset_config['type']
            asset_types_mapping = {}
            for k, v in asset_types.items():
                for sub_k in v.keys():
                    asset_types_mapping[sub_k] = k + '_' + sub_k
            valid_assets = os.listdir('./assets/adj_parameter_folder/')
            # spawn all static objects in one dataset
            asset_buildings = [a for a in all_assets if 'building' in a.lower()]
            asset_not_buildings = [a for a in all_assets if a not in asset_buildings]
            asset_position_list = []
            for asset_path in all_assets:
                usd_path = copy.deepcopy(asset_path)
                asset_valid = False
                for k, v in asset_types_mapping.items():
                    if k in asset_path:
                        asset_path = asset_path.replace(k, v)
                        asset_path = asset_path.replace('usd', 'json')
                        asset_path = v + '-' + asset_path.split('_')[-1]
                        if asset_path in valid_assets:
                            asset_valid = True
                        break
                if not asset_valid:
                    continue
                param_path = './assets/adj_parameter_folder/' + asset_path
                obj_info = json.load(open(param_path, 'rb'))
                prim_path = asset_path[:-5].replace('-', '_')
                proto_prim_path = f"/World/Dataset/Object_{prim_path}"
                proto_asset_config = sim_utils.UsdFileCfg(
                    scale=(obj_info['scale'],obj_info['scale'],obj_info['scale']),
                    usd_path=f"assets/usds/{usd_path}",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=min(obj_info.get('mass', 1000), 1000)),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                )
                prim = proto_asset_config.func(proto_prim_path, proto_asset_config)
                prim.SetInstanceable(True)
                for child in prim.GetChildren():
                    child.SetInstanceable(True)
                prim.SetActive(True)
                # save the proto prim path
                proto_prim_paths.append([proto_prim_path, obj_info])
                # set the prim visibility
                if hasattr(proto_asset_config, "visible"):
                    imageable = UsdGeom.Imageable(prim)
                    if proto_asset_config.visible:
                        imageable.MakeVisible()
                    else:
                        imageable.MakeInvisible()
                # set the semantic annotations
                if hasattr(proto_asset_config, "semantic_tags") and proto_asset_config.semantic_tags is not None:
                    # note: taken from replicator scripts.utils.utils.py
                    for semantic_type, semantic_value in proto_asset_config.semantic_tags:
                        # deal with spaces by replacing them with underscores
                        semantic_type_sanitized = semantic_type.replace(" ", "_")
                        semantic_value_sanitized = semantic_value.replace(" ", "_")
                        # set the semantic API for the instance
                        instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                        sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                        # create semantic type and data attributes
                        sem.CreateSemanticTypeAttr()
                        sem.CreateSemanticDataAttr()
                        sem.GetSemanticTypeAttr().Set(semantic_type)
                        sem.GetSemanticDataAttr().Set(semantic_value)
                # activate rigid body contact sensors
                if hasattr(proto_asset_config, "activate_contact_sensors") and proto_asset_config.activate_contact_sensors:
                    sim_utils.activate_contact_sensors(proto_prim_path, proto_asset_config.activate_contact_sensors)
            obj_prim_path_list = []
            prim_path_list = []     
            for env_idx in range(self.num_envs):
                obj_position_list = []
                polygon_xy_in_world = polygon_points[env_idx]
                x, y, upper, lower = polygon_xy_in_world[0], polygon_xy_in_world[1], polygon_xy_in_world[2], polygon_xy_in_world[3]
                buffer_width = 1.
                polyline_boundary = np.concatenate([np.column_stack((polygon_xy_in_world[0], upper + buffer_width)), np.column_stack((x[::-1], lower[::-1] - buffer_width))])
                polygon_boundary = Polygon(polyline_boundary)
                
                # PG: buildings & objects
                buildings = []
                objects = []
                object_positions = []
                if generation_cfg['type'] == 'dynamic':
                    print(f'[INFO] Using dynamic pedestrians in the scene, the objects will not be placed in centeric regions.')
                for n_obj in range(num_objects):
                    try_times = 0
                    while True:
                        try_times += 1
                        if try_times > 20:
                            print(f"[Warning] Failed to generate object after {try_times} tries.")
                            break
                        random_x = np.random.uniform(tmp_origin[env_idx, 0] + 1, tmp_origin[env_idx, 0] + area_size[0])
                        random_y = np.random.uniform(tmp_origin[env_idx, 1] + 1, tmp_origin[env_idx, 1] + area_size[1])
                        if generation_cfg['type'] == 'dynamic':
                            if np.random.rand() < 0.5:
                                random_x = np.random.uniform(tmp_origin[env_idx, 0] + 1, tmp_origin[env_idx, 0] + area_size[0])
                                random_y = np.random.uniform(tmp_origin[env_idx, 1] + 1, tmp_origin[env_idx, 1] + area_size[1] / 4)
                            else:
                                random_x = np.random.uniform(tmp_origin[env_idx, 0] + 1, tmp_origin[env_idx, 0] + area_size[0])
                                random_y = np.random.uniform(tmp_origin[env_idx, 1] + area_size[1] * 3 / 4, tmp_origin[env_idx, 1] + area_size[1] - 1)
                        point = Point(random_x, random_y)
                        if polygon_boundary.contains(point):
                            flag = False
                            if len(objects) > 0:
                                for obj_cur in objects:
                                    obj_position = [random_x, random_y, 0]
                                    obj = trimesh.primitives.Sphere(radius=buffer_width, center=obj_position)
                                    if not obj.intersection(obj_cur).vertices.shape[0] and not obj_cur.intersection(obj).vertices.shape[0]:
                                        objects.append(obj)
                                        object_positions.append(obj_position)
                                        flag = True
                                        break
                            else:
                                obj_position = [random_x, random_y, 0]
                                obj = trimesh.primitives.Sphere(radius=buffer_width, center=obj_position)
                                objects.append(obj)
                                object_positions.append(obj_position)
                                flag = True
                            if flag:
                                break
                
                if not hasattr(self, 'tmp_object_positions'):
                    self.tmp_object_positions = copy.deepcopy([
                        [
                            pos[0] - tmp_origin[env_idx, 0],
                            pos[1] - tmp_origin[env_idx, 1],
                            0
                        ] for pos in object_positions
                    ])
                else:
                    object_positions = copy.deepcopy([
                        [
                            pos[0] + tmp_origin[env_idx, 0],
                            pos[1] + tmp_origin[env_idx, 1],
                            0
                        ] for pos in self.tmp_object_positions
                    ])

                del objects
                del buildings
                proto_prim_paths_no_building = [p for p in proto_prim_paths if 'building' not in p[0].lower()]
                proto_prim_paths_no_building = [p for p in proto_prim_paths_no_building if 'tree' not in p[0].lower()]
                proto_prim_paths_no_building = [p for p in proto_prim_paths_no_building if 'wall' not in p[0].lower()]
                for obj_idx, pos in enumerate(object_positions):
                    proto_prim_path_i = np.random.choice([i for i in range(len(proto_prim_paths_no_building))])
                    if not hasattr(self, 'random_p_list'):
                        self.random_p_list = [
                            [
                                np.random.choice([i for i in range(len(proto_prim_paths_no_building))]) for _ in range(len(object_positions))
                            ] for _ in range(1)
                        ]
                    proto_prim_path_i = self.random_p_list[env_idx % 1][obj_idx]
                    proto_prim_path = proto_prim_paths_no_building[proto_prim_path_i]
                    prim_path = proto_prim_path[0].replace('/World/Dataset/Object_', '')
                    obj_info = proto_prim_path[1]
                    prim_path=f"/World/envs/env_{env_idx}/" + f"Object_{prim_path}" + f'{obj_idx:04d}'
                    obj_prim_path_list.append([prim_path, (pos[0] - tmp_origin[env_idx, 0] + obj_info['pos0'],pos[1] - tmp_origin[env_idx, 1] + obj_info['pos1'], mesh_block_height + obj_info['pos2']), (0.707, 0.707,0.0,0.0)])
                    prim_path_list.append(prim_path)
                    obj_position_list.append(
                        [
                            pos[0] - tmp_origin[env_idx, 0] + obj_info['pos0'],pos[1] - tmp_origin[env_idx, 1] + obj_info['pos1']
                        ]
                    )
                asset_position_list.append(obj_position_list)
            all_rigid_objects_config = RigidObjectCfg(
                prim_path="/World/envs/env_*/Object_*",
                spawn=DiverseAssetCfg(
                    assets_cfg=obj_prim_path_list
                )
            )
            self.obj_prim_path_list = obj_prim_path_list
            all_rigid_objects_config.prim_path = all_rigid_objects_config.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
            if hasattr(all_rigid_objects_config, "collision_group") and all_rigid_objects_config.collision_group == -1:
                asset_paths = sim_utils.find_matching_prim_paths(all_rigid_objects_config.prim_path)
                self._global_prim_paths += asset_paths
            self._rigid_objects['all_rigid_objects_config'] = all_rigid_objects_config.class_type(all_rigid_objects_config)
            self.asset_position = torch.tensor(asset_position_list).to(self.device).reshape(self.num_envs, -1, 2)   
    
        # dynamic objects
        if generation_cfg['type'] == 'dynamic':
            self.use_dynamic_pedestrians = True
            print(f'[INFO] use_dynamic_pedestrians->True: Using dynamic pedestrians in the scene.')
            self.dynamic_asset_animatable_state()
            print('[INFO] dynamic asset animatable state is set.')
            
            # register the Pedestrian dataset cache
            prims_utils.create_prim("/World/DatasetDynamic", "Scope")
            dynamic_proto_prim_paths = list()

            for human_id, human_info in enumerate(self.unique_dynamic_asset_path):
                proto_prim_path = f"/World/DatasetDynamic/Human_{human_id:04d}"
                proto_asset_config = sim_utils.UsdFileCfg(
                    scale=(0.01, 0.01, 0.01),
                    usd_path=human_info,
                )
                prim = proto_asset_config.func(proto_prim_path, proto_asset_config)
                # save the proto prim path
                dynamic_proto_prim_paths.append(proto_prim_path)
                # set the prim visibility
                if hasattr(proto_asset_config, "visible"):
                    imageable = UsdGeom.Imageable(prim)
                    if proto_asset_config.visible:
                        imageable.MakeVisible()
                    else:
                        imageable.MakeInvisible()
                # set the semantic annotations
                if hasattr(proto_asset_config, "semantic_tags") and proto_asset_config.semantic_tags is not None:
                    # note: taken from replicator scripts.utils.utils.py
                    for semantic_type, semantic_value in proto_asset_config.semantic_tags:
                        # deal with spaces by replacing them with underscores
                        semantic_type_sanitized = semantic_type.replace(" ", "_")
                        semantic_value_sanitized = semantic_value.replace(" ", "_")
                        # set the semantic API for the instance
                        instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                        sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                        # create semantic type and data attributes
                        sem.CreateSemanticTypeAttr()
                        sem.CreateSemanticDataAttr()
                        sem.GetSemanticTypeAttr().Set(semantic_type)
                        sem.GetSemanticDataAttr().Set(semantic_value)
                # activate rigid body contact sensors
                if hasattr(proto_asset_config, "activate_contact_sensors") and proto_asset_config.activate_contact_sensors:
                    sim_utils.activate_contact_sensors(proto_prim_path, proto_asset_config.activate_contact_sensors)
            
            human_prim_path_list = []
            # split to int(sqrt(n) + 1) y-direction regions, for example:  n = 16
            # [1/4, 3/8, 1/2, 5/8, 3/4]
            # split to int(sqrt(n) + 1)  x-direction regions, for example:  n = 16
            # [0, 1/4, 1/2, 3/4, 1]
            # p <= 0.5: -> towards x
            num_pedestrian = generation_cfg['num_pedestrian']
            grid_split = int(math.sqrt(num_pedestrian) + 0.5) + 1

            x_bins = np.linspace(0, 1, grid_split)  # [0, 0.25, 0.5, 0.75, 1.0]
            x_start_left = x_bins[:-1]  # [0, 0.25, 0.5, 0.75]
            y_bins = np.linspace(0.25, 0.75, grid_split)  # [0, 0.25, 0.5, 0.75, 1.0]
            y_start_left = y_bins[:-1]  # [0.25, 0.5, 0.75, 1.0]
            pedestrian_forward_backward_heading_list = []
            pedestrain_forward_start_end_pos_list = []
            for env_idx in range(self.num_envs):
                for human_idx in range(num_pedestrian):
                    prim_path=f"/World/envs/env_{env_idx}/" + f"Dynamic_" + f'{human_idx:04d}'
                    grid_i = human_idx // (grid_split - 1)
                    grid_j = human_idx % (grid_split - 1)
                    grid_i = min(grid_i, len(x_start_left) - 1)
                    grid_j = min(grid_j, len(y_start_left) - 1)
                    start_x = x_start_left[grid_i] * area_size[0]
                    start_y = y_start_left[grid_j] * area_size[1]
                    
                    from scipy.spatial.transform import Rotation as R
                    if np.random.rand() < 0.5:
                        direction = 'y+'
                        delta_rot = np.pi
                    else:
                        direction = 'x+'
                        delta_rot = np.pi / 2
                    delta_rotation = R.from_euler('y', delta_rot)
                    default_quat = [0.5, 0.5, 0.5, 0.5] 
                    rotation = R.from_quat(default_quat)
                    # Apply the heading rotation to the current rotation
                    new_rotation =  rotation * delta_rotation
                    new_quat = new_rotation.as_quat()
                    qw, qx, qy, qz = new_quat[3], new_quat[0], new_quat[1], new_quat[2]
                    
                    delta_rotation = R.from_euler('y', delta_rot - np.pi)
                    # Apply the heading rotation to the current rotation
                    new_rotation =  rotation * delta_rotation
                    new_quat = new_rotation.as_quat()
                    qw_inv, qx_inv, qy_inv, qz_inv = new_quat[3], new_quat[0], new_quat[1], new_quat[2]
                    pedestrian_forward_backward_heading_list.append(
                        [
                            [qw, qx, qy, qz],  # forward heading
                            [qw_inv, qx_inv, qy_inv, qz_inv]  # backward heading
                        ]
                    )
                    if direction == 'y+':
                        pedestrain_forward_start_end_pos_list.append(
                            [
                                [start_x, start_y, 1.30],  # forward start position
                                [start_x, start_y + area_size[1] * (y_start_left[1] - y_start_left[0]) * 0.7, 1.30]  # forward end position
                            ]
                        )
                    elif direction == 'x+':
                        pedestrain_forward_start_end_pos_list.append(
                            [
                                [start_x, start_y, 1.30],  # forward start position
                                [start_x + area_size[0] * (x_start_left[1] - x_start_left[0]) * 0.7, start_y, 1.30]  # forward end position
                            ]
                        )
                    
                    human_prim_path_list.append(
                            [
                                np.random.choice(dynamic_proto_prim_paths), prim_path, [start_x, start_y, 1.30], (qw, qx, qy, qz)
                            ]
                        )
            all_human_config = RigidObjectCfg(
                prim_path="/World/envs/env_*/Dynamic_*",
                spawn=DiversHumanCfg(
                    assets_cfg=human_prim_path_list
                )
            )
            all_human_config.prim_path = all_human_config.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
            if hasattr(all_human_config, "collision_group") and all_human_config.collision_group == -1:
                asset_paths = sim_utils.find_matching_prim_paths(all_human_config.prim_path)
                self._global_prim_paths += asset_paths
            self._rigid_objects['all_human_config'] = all_human_config.class_type(all_human_config)
            
            # buffer
            self.pedestrian_forward_backward_heading_list = pedestrian_forward_backward_heading_list
            self.pedestrian_forward_start_end_pos_list = pedestrain_forward_start_end_pos_list
            self.human_prim_path_list = human_prim_path_list
            
            for env_idx in range(self.num_envs):
                for human_idx in range(generation_cfg['num_pedestrian']):
                    prim_path=f"/World/envs/env_{env_idx}/" + f"Dynamic_" + f'{human_idx:04d}'
                    mesh_prim = stage.GetPrimAtPath(prim_path + '/root/pelvis0/Skeleton')
                    UsdSkel.BindingAPI.Apply(mesh_prim)
                    mesh_binding_api = UsdSkel.BindingAPI(mesh_prim)
                    rel =  mesh_binding_api.CreateAnimationSourceRel()
                    rel.ClearTargets(True)
                    rel.AddTarget("/World/walk/SMPLX_neutral/root/pelvis0/SMPLX_neutral_Scene")
        
        # deactivate some prims
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(Sdf.Path(f'/World/ground/Environment'))
        prim.SetActive(False)
        prim = stage.GetPrimAtPath(Sdf.Path(f'/World/Obstacle_terrain/Environment'))
        prim.SetActive(False)
        
    def generate_limited_async_procedural_scene(self):
        generation_cfg = self.cfg.pg_config
        area_size = generation_cfg['map_region']
        torch.manual_seed(generation_cfg['seed'])
        np.random.seed(generation_cfg['seed'])
        random.seed(generation_cfg['seed'])
        # ground plane
        tmp_origin = self._default_env_origins[..., :2].reshape(self.num_envs, 2).cpu().numpy()
        mesh_block_height = 0.01
        if generation_cfg['type'] == 'clean' or \
           generation_cfg['type'] == 'static' or \
           generation_cfg['type'] == 'dynamic':
            area_size = [area_size, area_size]
            
            polygon_points = []
            walkable_region_polygon_list = [[] for _ in range(len(self.cfg.terrain_importer_walkable_list))]
            all_region_polygon_list = [[] for _ in range(len(self.cfg.terrain_non_walkable_list))]
            polyline_points_list = []
            
            for env_idx in range(self.num_envs):
                buffer_width = generation_cfg['buffer_width']
                x, y = generate_random_road(area_size)
                x, y = np.array(x), np.array(y)
                
                x += tmp_origin[env_idx, 0]
                y += tmp_origin[env_idx, 1]
                x = x.clip(tmp_origin[env_idx, 0], tmp_origin[env_idx, 0] + area_size[0])
                y = y.clip(tmp_origin[env_idx, 1], tmp_origin[env_idx, 1] + area_size[1])
                
                # walkable regions
                mesh, boundary_points, polyline_points = get_road_trimesh(x, y, area_size, boundary=(tmp_origin[env_idx, 1], tmp_origin[env_idx, 1] + area_size[1]), height=mesh_block_height+0.02)
                polyline_points_list.append(polyline_points)
                polygon_points.append([x, y, boundary_points[0], boundary_points[1]])
                
                # non walkable regions
                area_polygon = np.array([(0, 0), (0, area_size[1] + buffer_width), (area_size[0] + buffer_width, area_size[1] + buffer_width), (area_size[0] + buffer_width, 0)]).astype(float)
                area_polygon[:, 0] += tmp_origin[env_idx, 0]
                area_polygon[:, 1] += tmp_origin[env_idx, 1]
                area_mesh = trimesh.creation.extrude_polygon(Polygon(area_polygon), height=mesh_block_height)
                
                # boundary polygons
                if generation_cfg.get('with_boundary', False):
                    boundary = np.array([(0, -0.5), (0, 0.0), (area_size[0] + 0.5, 0.0), (area_size[0] + 0.5, -0.5)]).astype(float)
                    boundary[:, 0] += tmp_origin[env_idx, 0]
                    boundary[:, 1] += tmp_origin[env_idx, 1]
                    boundary_mesh = trimesh.creation.extrude_polygon(Polygon(boundary), height=mesh_block_height * 10)
                    all_region_polygon_list[(generation_cfg['non_walkable_seed'] + env_idx) % len(all_region_polygon_list)].append(boundary_mesh)
                    boundary = np.array([(0, area_size[1]), (0, area_size[1] + 0.5), (area_size[0] + 0.5, area_size[1] + 0.5), (area_size[0] + 0.5, area_size[1])]).astype(float)
                    boundary[:, 0] += tmp_origin[env_idx, 0]
                    boundary[:, 1] += tmp_origin[env_idx, 1]
                    boundary_mesh = trimesh.creation.extrude_polygon(Polygon(boundary), height=mesh_block_height * 10)
                    all_region_polygon_list[(generation_cfg['non_walkable_seed'] + env_idx) % len(all_region_polygon_list)].append(boundary_mesh)
                    
                    boundary = np.array([(-0.5, 0.0), (0, 0.0), (0.0, area_size[1] + 0.5), (-0.5, area_size[1] + 0.5)]).astype(float)
                    boundary[:, 0] += tmp_origin[env_idx, 0]
                    boundary[:, 1] += tmp_origin[env_idx, 1]
                    boundary_mesh = trimesh.creation.extrude_polygon(Polygon(boundary), height=mesh_block_height * 10)
                    all_region_polygon_list[(generation_cfg['non_walkable_seed'] + env_idx) % len(all_region_polygon_list)].append(boundary_mesh)
                    boundary = np.array([(area_size[0], 0.0), (area_size[0] + 0.5, 0.0), (area_size[0] + 0.5, area_size[1] + 0.5), (area_size[0], area_size[1] + 0.5)]).astype(float)
                    boundary[:, 0] += tmp_origin[env_idx, 0]
                    boundary[:, 1] += tmp_origin[env_idx, 1]
                    boundary_mesh = trimesh.creation.extrude_polygon(Polygon(boundary), height=mesh_block_height * 10)
                    all_region_polygon_list[(generation_cfg['non_walkable_seed'] + env_idx) % len(all_region_polygon_list)].append(boundary_mesh)

                walkable_region_polygon_list[(generation_cfg['walkable_seed'] + env_idx) % len(walkable_region_polygon_list)].append(mesh)
                all_region_polygon_list[(generation_cfg['non_walkable_seed'] + env_idx) % len(all_region_polygon_list)].append(area_mesh)

            self.polygon_points = polygon_points
            self.polylines_of_all_walkable_regions = torch.from_numpy(np.stack(polyline_points_list)).float().to(self.device)
            for i in range(len(walkable_region_polygon_list)):
                mesh_list = walkable_region_polygon_list[i]
                if len(mesh_list) == 0:
                    continue
                combined_mesh = trimesh.util.concatenate(mesh_list)
                
                # uv for texturing
                combined_mesh = uv_texturing(combined_mesh, scale=UV_SCLAE)
                self.walkable_terrain_list[i].import_mesh('mesh', combined_mesh)
                sim_utils.bind_visual_material(f'/World/Walkable_{i:03d}', f'/World/Looks/terrain_walkable_material_list_{i:03d}')
                sim_utils.bind_physics_material(f'/World/Walkable_{i:03d}', f'/World/Looks/terrain_non_walkable_material_list_{i:03d}')
                stage = omni.usd.get_context().get_stage()
                prim = stage.GetPrimAtPath(Sdf.Path(f'/World/Walkable_{i:03d}/Environment'))
                prim.SetActive(False)
                
            for i in range(len(all_region_polygon_list)):
                mesh_list = all_region_polygon_list[i]
                if len(mesh_list) == 0:
                    continue
                combined_mesh = trimesh.util.concatenate(mesh_list)
                
                # uv for texturing
                combined_mesh = uv_texturing(combined_mesh, scale=UV_SCLAE)
                self.all_region_list[i].import_mesh('mesh', combined_mesh)
                sim_utils.bind_visual_material(f'/World/NonWalkable_{i:03d}', f'/World/Looks/terrain_non_walkable_material_list_{i:03d}')
                sim_utils.bind_physics_material(f'/World/NonWalkable_{i:03d}', f'/World/Looks/terrain_non_walkable_material_list_{i:03d}')
                prim = stage.GetPrimAtPath(Sdf.Path(f'/World/NonWalkable_{i:03d}/Environment'))
                prim.SetActive(False)
            
            # Remove ground plane
            for i in range(max(len(self.all_region_list), len(self.walkable_terrain_list))):
                try:
                    prim = stage.GetPrimAtPath(Sdf.Path(f'/World/NonWalkable_{i:03d}/Environment'))
                    prim.SetActive(False)
                except:
                    pass
                
                try:
                    prim = stage.GetPrimAtPath(Sdf.Path(f'/World/Walkable_{i:03d}/Environment'))
                    prim.SetActive(False)
                except:
                    pass
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/ground/Environment'))
            prim.SetActive(False)
            
        # static objects
        if generation_cfg['type'] == 'static' or  generation_cfg['type'] == 'dynamic':
            num_objects = generation_cfg['num_object']
            # register dataset path
            prims_utils.create_prim("/World/Dataset", "Scope")
            proto_prim_paths = list()
            # get all assets from dataset
            asset_root_path = 'assets/usds/'
            all_assets = os.listdir(asset_root_path)
            all_assets = [i for i in all_assets if 'non_metric' not in i]
            import yaml
            with open('./assets/asset_config.yaml', "r") as file:
                asset_config = yaml.safe_load(file)
            asset_types = asset_config['type']
            asset_types_mapping = {}
            for k, v in asset_types.items():
                for sub_k in v.keys():
                    asset_types_mapping[sub_k] = k + '_' + sub_k
            valid_assets = os.listdir('./assets/adj_parameter_folder/')
            # spawn all static objects in one dataset
            asset_buildings = [a for a in all_assets if 'building' in a.lower()]
            asset_not_buildings = [a for a in all_assets if a not in asset_buildings]
            asset_position_list = []
            for asset_path in all_assets:
                usd_path = copy.deepcopy(asset_path)
                asset_valid = False
                for k, v in asset_types_mapping.items():
                    if k in asset_path:
                        asset_path = asset_path.replace(k, v)
                        asset_path = asset_path.replace('usd', 'json')
                        asset_path = v + '-' + asset_path.split('_')[-1]
                        if asset_path in valid_assets:
                            asset_valid = True
                        break
                if not asset_valid:
                    continue
                param_path = './assets/adj_parameter_folder/' + asset_path
                obj_info = json.load(open(param_path, 'rb'))
                prim_path = asset_path[:-5].replace('-', '_')
                proto_prim_path = f"/World/Dataset/Object_{prim_path}"
                proto_asset_config = sim_utils.UsdFileCfg(
                    scale=(obj_info['scale'],obj_info['scale'],obj_info['scale']),
                    usd_path=f"assets/usds/{usd_path}",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=min(obj_info.get('mass', 1000), 1000)),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                )
                prim = proto_asset_config.func(proto_prim_path, proto_asset_config)
                prim.SetInstanceable(True)
                for child in prim.GetChildren():
                    child.SetInstanceable(True)
                prim.SetActive(True)
                # save the proto prim path
                proto_prim_paths.append([proto_prim_path, obj_info])
                # set the prim visibility
                if hasattr(proto_asset_config, "visible"):
                    imageable = UsdGeom.Imageable(prim)
                    if proto_asset_config.visible:
                        imageable.MakeVisible()
                    else:
                        imageable.MakeInvisible()
                # set the semantic annotations
                if hasattr(proto_asset_config, "semantic_tags") and proto_asset_config.semantic_tags is not None:
                    # note: taken from replicator scripts.utils.utils.py
                    for semantic_type, semantic_value in proto_asset_config.semantic_tags:
                        # deal with spaces by replacing them with underscores
                        semantic_type_sanitized = semantic_type.replace(" ", "_")
                        semantic_value_sanitized = semantic_value.replace(" ", "_")
                        # set the semantic API for the instance
                        instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                        sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                        # create semantic type and data attributes
                        sem.CreateSemanticTypeAttr()
                        sem.CreateSemanticDataAttr()
                        sem.GetSemanticTypeAttr().Set(semantic_type)
                        sem.GetSemanticDataAttr().Set(semantic_value)
                # activate rigid body contact sensors
                if hasattr(proto_asset_config, "activate_contact_sensors") and proto_asset_config.activate_contact_sensors:
                    sim_utils.activate_contact_sensors(proto_prim_path, proto_asset_config.activate_contact_sensors)
            obj_prim_path_list = []
            prim_path_list = []   
            for env_idx in range(self.num_envs):
                obj_position_list = []
                object_positions = []  
                polygon_xy_in_world = polygon_points[env_idx]
                x, y, upper, lower = polygon_xy_in_world[0], polygon_xy_in_world[1], polygon_xy_in_world[2], polygon_xy_in_world[3]
                buffer_width = 1.
                polyline_boundary = np.concatenate([np.column_stack((polygon_xy_in_world[0], upper + buffer_width)), np.column_stack((x[::-1], lower[::-1] - buffer_width))])
                polygon_boundary = Polygon(polyline_boundary)
                
                # PG: buildings & objects
                buildings = []
                objects = []
                if generation_cfg['type'] == 'dynamic':
                    print(f'[INFO] Using dynamic pedestrians in the scene, the objects will not be placed in centeric regions.')
                for n_obj in range(num_objects):
                    try_times = 0
                    while True:
                        try_times += 1
                        if try_times > 20:
                            print(f"[Warning] Failed to generate object after {try_times} tries.")
                            break
                        random_x = np.random.uniform(tmp_origin[env_idx, 0] + 1, tmp_origin[env_idx, 0] + area_size[0])
                        random_y = np.random.uniform(tmp_origin[env_idx, 1] + 1, tmp_origin[env_idx, 1] + area_size[1])
                        if generation_cfg['type'] == 'dynamic':
                            if np.random.rand() < 0.5:
                                random_x = np.random.uniform(tmp_origin[env_idx, 0] + 1, tmp_origin[env_idx, 0] + area_size[0])
                                random_y = np.random.uniform(tmp_origin[env_idx, 1] + 1, tmp_origin[env_idx, 1] + area_size[1] / 4)
                            else:
                                random_x = np.random.uniform(tmp_origin[env_idx, 0] + 1, tmp_origin[env_idx, 0] + area_size[0])
                                random_y = np.random.uniform(tmp_origin[env_idx, 1] + area_size[1] * 3 / 4, tmp_origin[env_idx, 1] + area_size[1] - 1)
                        point = Point(random_x, random_y)
                        if polygon_boundary.contains(point):
                            flag = False
                            if len(objects) > 0:
                                for obj_cur in objects:
                                    obj_position = [random_x, random_y, 0]
                                    obj = trimesh.primitives.Sphere(radius=buffer_width, center=obj_position)
                                    if not obj.intersection(obj_cur).vertices.shape[0] and not obj_cur.intersection(obj).vertices.shape[0]:
                                        objects.append(obj)
                                        object_positions.append(obj_position)
                                        flag = True
                                        break
                            else:
                                obj_position = [random_x, random_y, 0]
                                obj = trimesh.primitives.Sphere(radius=buffer_width, center=obj_position)
                                objects.append(obj)
                                object_positions.append(obj_position)
                                flag = True
                            if flag:
                                break
                
                if not hasattr(self, 'tmp_object_positions'):
                    self.tmp_object_positions = [copy.deepcopy([
                        [
                            pos[0] - tmp_origin[env_idx, 0],
                            pos[1] - tmp_origin[env_idx, 1],
                            0
                        ] for pos in object_positions
                    ])]
                elif len(self.tmp_object_positions) < generation_cfg['unique_env_num']:
                    self.tmp_object_positions.append(copy.deepcopy([
                        [
                            pos[0] - tmp_origin[env_idx, 0],
                            pos[1] - tmp_origin[env_idx, 1],
                            0
                        ] for pos in object_positions
                    ]))
                else:
                    object_positions = copy.deepcopy([
                        [
                            pos[0] + tmp_origin[env_idx, 0],
                            pos[1] + tmp_origin[env_idx, 1],
                            0
                        ] for pos in self.tmp_object_positions[np.random.choice([i for i in range(generation_cfg['unique_env_num'])])]
                    ])

                del objects
                del buildings
                proto_prim_paths_no_building = [p for p in proto_prim_paths if 'building' not in p[0].lower()]
                proto_prim_paths_no_building = [p for p in proto_prim_paths_no_building if 'tree' not in p[0].lower()]
                proto_prim_paths_no_building = [p for p in proto_prim_paths_no_building if 'wall' not in p[0].lower()]
                for obj_idx, pos in enumerate(object_positions):
                    proto_prim_path_i = np.random.choice([i for i in range(len(proto_prim_paths_no_building))])
                    if not hasattr(self, 'random_p_list'):
                        self.random_p_list = [
                            [
                                np.random.choice([i for i in range(len(proto_prim_paths_no_building))]) for _ in range(len(object_positions))
                            ] for _ in range(generation_cfg['unique_env_num'])
                        ]
                    proto_prim_path_i = self.random_p_list[env_idx % generation_cfg['unique_env_num']][obj_idx]
                    proto_prim_path = proto_prim_paths_no_building[proto_prim_path_i]
                    prim_path = proto_prim_path[0].replace('/World/Dataset/Object_', '')
                    obj_info = proto_prim_path[1]
                    prim_path=f"/World/envs/env_{env_idx}/" + f"Object_{prim_path}" + f'{obj_idx:04d}'
                    obj_prim_path_list.append([prim_path, (pos[0] - tmp_origin[env_idx, 0] + obj_info['pos0'],pos[1] - tmp_origin[env_idx, 1] + obj_info['pos1'], mesh_block_height + obj_info['pos2']), (0.707, 0.707,0.0,0.0)])
                    prim_path_list.append(prim_path)
                    obj_position_list.append(
                        [
                            pos[0] - tmp_origin[env_idx, 0] + obj_info['pos0'],pos[1] - tmp_origin[env_idx, 1] + obj_info['pos1']
                        ]
                    )
                asset_position_list.append(obj_position_list)
            all_rigid_objects_config = RigidObjectCfg(
                prim_path="/World/envs/env_*/Object_*",
                spawn=DiverseAssetCfg(
                    assets_cfg=obj_prim_path_list
                )
            )
            self.obj_prim_path_list = obj_prim_path_list
            all_rigid_objects_config.prim_path = all_rigid_objects_config.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
            if hasattr(all_rigid_objects_config, "collision_group") and all_rigid_objects_config.collision_group == -1:
                asset_paths = sim_utils.find_matching_prim_paths(all_rigid_objects_config.prim_path)
                self._global_prim_paths += asset_paths
            self._rigid_objects['all_rigid_objects_config'] = all_rigid_objects_config.class_type(all_rigid_objects_config)
            self.asset_position = torch.tensor(asset_position_list).to(self.device).reshape(self.num_envs, -1, 2)   
    
        # dynamic objects
        if generation_cfg['type'] == 'dynamic':
            self.use_dynamic_pedestrians = True
            print(f'[INFO] use_dynamic_pedestrians->True: Using dynamic pedestrians in the scene.')
            self.dynamic_asset_animatable_state()
            print('[INFO] dynamic asset animatable state is set.')
            
            # register the Pedestrian dataset cache
            prims_utils.create_prim("/World/DatasetDynamic", "Scope")
            dynamic_proto_prim_paths = list()

            for human_id, human_info in enumerate(self.unique_dynamic_asset_path):
                proto_prim_path = f"/World/DatasetDynamic/Human_{human_id:04d}"
                proto_asset_config = sim_utils.UsdFileCfg(
                    scale=(0.01, 0.01, 0.01),
                    usd_path=human_info,
                )
                prim = proto_asset_config.func(proto_prim_path, proto_asset_config)
                # save the proto prim path
                dynamic_proto_prim_paths.append(proto_prim_path)
                # set the prim visibility
                if hasattr(proto_asset_config, "visible"):
                    imageable = UsdGeom.Imageable(prim)
                    if proto_asset_config.visible:
                        imageable.MakeVisible()
                    else:
                        imageable.MakeInvisible()
                # set the semantic annotations
                if hasattr(proto_asset_config, "semantic_tags") and proto_asset_config.semantic_tags is not None:
                    # note: taken from replicator scripts.utils.utils.py
                    for semantic_type, semantic_value in proto_asset_config.semantic_tags:
                        # deal with spaces by replacing them with underscores
                        semantic_type_sanitized = semantic_type.replace(" ", "_")
                        semantic_value_sanitized = semantic_value.replace(" ", "_")
                        # set the semantic API for the instance
                        instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                        sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                        # create semantic type and data attributes
                        sem.CreateSemanticTypeAttr()
                        sem.CreateSemanticDataAttr()
                        sem.GetSemanticTypeAttr().Set(semantic_type)
                        sem.GetSemanticDataAttr().Set(semantic_value)
                # activate rigid body contact sensors
                if hasattr(proto_asset_config, "activate_contact_sensors") and proto_asset_config.activate_contact_sensors:
                    sim_utils.activate_contact_sensors(proto_prim_path, proto_asset_config.activate_contact_sensors)
            
            human_prim_path_list = []
            # split to int(sqrt(n) + 1) y-direction regions, for example:  n = 16
            # [1/4, 3/8, 1/2, 5/8, 3/4]
            # split to int(sqrt(n) + 1)  x-direction regions, for example:  n = 16
            # [0, 1/4, 1/2, 3/4, 1]
            # p <= 0.5: -> towards x
            num_pedestrian = generation_cfg['num_pedestrian']
            grid_split = int(math.sqrt(num_pedestrian) + 0.5) + 1

            x_bins = np.linspace(0, 1, grid_split)  # [0, 0.25, 0.5, 0.75, 1.0]
            x_start_left = x_bins[:-1]  # [0, 0.25, 0.5, 0.75]
            y_bins = np.linspace(0.25, 0.75, grid_split)  # [0, 0.25, 0.5, 0.75, 1.0]
            y_start_left = y_bins[:-1]  # [0.25, 0.5, 0.75, 1.0]
            pedestrian_forward_backward_heading_list = []
            pedestrain_forward_start_end_pos_list = []
            for env_idx in range(self.num_envs):
                np.random.seed(env_idx)
                for human_idx in range(num_pedestrian):
                    prim_path=f"/World/envs/env_{env_idx}/" + f"Dynamic_" + f'{human_idx:04d}'
                    grid_i = human_idx // (grid_split - 1)
                    grid_j = human_idx % (grid_split - 1)
                    grid_i = min(grid_i, len(x_start_left) - 1)
                    grid_j = min(grid_j, len(y_start_left) - 1)
                    start_x = x_start_left[grid_i] * area_size[0]
                    start_y = y_start_left[grid_j] * area_size[1]
                    
                    from scipy.spatial.transform import Rotation as R
                    if np.random.rand() < 0.5:
                        direction = 'y+'
                        delta_rot = np.pi
                    else:
                        direction = 'x+'
                        delta_rot = np.pi / 2
                    delta_rotation = R.from_euler('y', delta_rot)
                    default_quat = [0.5, 0.5, 0.5, 0.5] 
                    rotation = R.from_quat(default_quat)
                    # Apply the heading rotation to the current rotation
                    new_rotation =  rotation * delta_rotation
                    new_quat = new_rotation.as_quat()
                    qw, qx, qy, qz = new_quat[3], new_quat[0], new_quat[1], new_quat[2]
                    
                    delta_rotation = R.from_euler('y', delta_rot - np.pi)
                    # Apply the heading rotation to the current rotation
                    new_rotation =  rotation * delta_rotation
                    new_quat = new_rotation.as_quat()
                    qw_inv, qx_inv, qy_inv, qz_inv = new_quat[3], new_quat[0], new_quat[1], new_quat[2]
                    pedestrian_forward_backward_heading_list.append(
                        [
                            [qw, qx, qy, qz],  # forward heading
                            [qw_inv, qx_inv, qy_inv, qz_inv]  # backward heading
                        ]
                    )
                    if direction == 'y+':
                        pedestrain_forward_start_end_pos_list.append(
                            [
                                [start_x, start_y, 1.30],  # forward start position
                                [start_x, start_y + area_size[1] * (y_start_left[1] - y_start_left[0]) * 0.8, 1.30]  # forward end position
                            ]
                        )
                    elif direction == 'x+':
                        pedestrain_forward_start_end_pos_list.append(
                            [
                                [start_x, start_y, 1.30],  # forward start position
                                [start_x + area_size[0] * (x_start_left[1] - x_start_left[0]) * 0.8, start_y, 1.30]  # forward end position
                            ]
                        )
                    
                    human_prim_path_list.append(
                            [
                                np.random.choice(dynamic_proto_prim_paths), prim_path, [start_x, start_y, 1.30], (qw, qx, qy, qz)
                            ]
                        )
            all_human_config = RigidObjectCfg(
                prim_path="/World/envs/env_*/Dynamic_*",
                spawn=DiversHumanCfg(
                    assets_cfg=human_prim_path_list
                )
            )
            all_human_config.prim_path = all_human_config.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
            if hasattr(all_human_config, "collision_group") and all_human_config.collision_group == -1:
                asset_paths = sim_utils.find_matching_prim_paths(all_human_config.prim_path)
                self._global_prim_paths += asset_paths
            self._rigid_objects['all_human_config'] = all_human_config.class_type(all_human_config)
            
            # buffer
            self.pedestrian_forward_backward_heading_list = pedestrian_forward_backward_heading_list
            self.pedestrian_forward_start_end_pos_list = pedestrain_forward_start_end_pos_list
            self.human_prim_path_list = human_prim_path_list
            
            for env_idx in range(self.num_envs):
                for human_idx in range(generation_cfg['num_pedestrian']):
                    prim_path=f"/World/envs/env_{env_idx}/" + f"Dynamic_" + f'{human_idx:04d}'
                    mesh_prim = stage.GetPrimAtPath(prim_path + '/root/pelvis0/Skeleton')
                    UsdSkel.BindingAPI.Apply(mesh_prim)
                    mesh_binding_api = UsdSkel.BindingAPI(mesh_prim)
                    rel =  mesh_binding_api.CreateAnimationSourceRel()
                    rel.ClearTargets(True)
                    rel.AddTarget("/World/walk/SMPLX_neutral/root/pelvis0/SMPLX_neutral_Scene")
        
        # deactivate some prims
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(Sdf.Path(f'/World/ground/Environment'))
        prim.SetActive(False)
        prim = stage.GetPrimAtPath(Sdf.Path(f'/World/Obstacle_terrain/Environment'))
        prim.SetActive(False)
        
    def generate_sync_procedural_scene(self):
        from metaurban.manager.traffic_manager import TrafficMode
        from metaurban.manager.sidewalk_manager import AssetManager
        from metaurban.manager.humanoid_manager import PGBackgroundSidewalkAssetsManager

        from metaurban.utils import clip, Config
        from metaurban.engine.engine_utils import set_global_random_seed
        from metaurban.component.map.pg_map import parse_map_config, MapGenerateMethod

        
        from metaurban.engine.base_engine import BaseEngine
        from urbansim.utils import BASE_DEFAULT_CONFIG
        from urbansim.utils.map_manager import PGMapManager
        
        # configure the logical engine
        config = Config(BASE_DEFAULT_CONFIG)
        config.register_type("map", str, int)
        config["map_config"].register_type("config", None)
        config.update(dict(
            use_render=False, image_observation=False, interface_panel=False,
            
            # traffic related
            show_mid_block_map=False,
            traffic_mode=TrafficMode.Trigger,
            random_traffic=False,
            traffic_density=0.0,
            traffic_vehicle_config=dict(
            show_navi_mark=False,
            show_dest_mark=False,
            enable_reverse=False,
            show_lidar=False,
            show_lane_line_detector=False,
            show_side_detector=False,),
            
             # pedestrian related
            spawn_human_num=1,
            spawn_wheelchairman_num=0,
            spawn_edog_num = 0,
            spawn_erobot_num=0,
            spawn_drobot_num=0,
            max_actor_num=1,))
        
        default_config_copy = Config(config, unchangeable=True)
        
        merged_config = config.update(self.cfg.pg_config, True, ["agent_configs", "sensors"])
        merged_config["map_config"] = parse_map_config(
            easy_map_config=config["map"], new_map_config=config["map_config"], default_config=default_config_copy
        )
        
        PG_CONFIG = Config(merged_config, unchangeable=False)
        
        cls = BaseEngine
        cls.singleton = BaseEngine(PG_CONFIG)
        self.logical_engine = cls.singleton
        self.logical_engine.register_manager('map_manager', PGMapManager())
        if self.cfg.pg_config['type'] != 'clean':
            self.logical_engine.register_manager('asset_manager', AssetManager())
        # if self.cfg.pg_config['type'] == 'dynamic':
        #     self.logical_engine.register_manager('human_manager', PGBackgroundSidewalkAssetsManager())
        print(f'[INFO] Logical engine is created.')
        
        # reset engine
        engine = self.logical_engine
        set_global_random_seed((engine.global_config['seed'] + engine.gets_start_index(engine.global_config)) % engine.global_config['num_scenarios'])
        try:
            engine.reset()
        except:
            set_global_random_seed(engine.gets_start_index(engine.global_config) % engine.global_config['num_scenarios'])
            engine.reset()
        print(f'[INFO] Logical engine is reset.')
        
        self.setup_object_dict()
        print(f'[INFO] Object cache is not setup because sync simulation does not need caching besides omniverse mechanism.')
        
        # initialize sidewalk
        self.set_sidewalk(engine.global_config['sidewalk_type'])
        
        # initialize polygon
        polyline_polygon_dict = self.set_polyline_polygon()
        self.polygons_of_lane = [polyline_polygon_dict['polygons'] for _ in range(self.num_envs)]
        self.white_line_polygons = [polyline_polygon_dict['white_line_polygons'] for _ in range(self.num_envs)]
        self.yellow_line_polygons = [polyline_polygon_dict['lane_yellow_line_polygons'] for _ in range(self.num_envs)]
        self.nb_polygons = [polyline_polygon_dict['near_road_buffer_polygons'] for _ in range(self.num_envs)]
        self.n_polygons = [polyline_polygon_dict['near_road_sidewalk_polygons'] for _ in range(self.num_envs)]
        self.s_polygons = [polyline_polygon_dict['sidewalk_polygons'] for _ in range(self.num_envs)]
        self.fb_polygons = [polyline_polygon_dict['far_road_buffer_polygons'] for _ in range(self.num_envs)]
        self.f_polygons = [polyline_polygon_dict['far_road_sidewalk_polygons'] for _ in range(self.num_envs)]
        self.h_polygons = [polyline_polygon_dict['house_region_polygons'] for _ in range(self.num_envs)]
        
        # initialize objects
        updated_cfg = copy.deepcopy(self.cfg)
        all_json_files = os.listdir('./assets/adj_parameter_folder/')
        usd_json_dict = {}
        for file in all_json_files:
            json_state = json.load(open('./assets/adj_parameter_folder/' + file, 'rb'))
            try:
                usd_name = json_state['filename'].replace('.glb', '.usd').replace('-', '_').replace(" ", "")
            except:
                usd_name = json_state['MODEL_PATH'].replace('.usd', '.usd').replace('-', '_').replace(" ", "").split('/')[-1]
            usd_json_dict[usd_name] = json_state
        if engine.global_config['object_density'] > 0 and self.cfg.pg_config['type'] != 'clean':
            self.sync_obejct_dict = self.set_sync_objects()
            import pickle
            with open('./types.pkl', 'rb') as f:
                types = pickle.load(f)
            for idx, primpath_usd_pos_rot_scale in enumerate(self.sync_obejct_dict['obj_prim_path_list']):
                try:
                    type_selected = [t for t in types if t in primpath_usd_pos_rot_scale[1]][0]
                except:
                    type_selected = 'unknown'
            
                rigid_cfg = RigidObjectCfg(
                    prim_path=primpath_usd_pos_rot_scale[0],
                    spawn=sim_utils.UsdFileCfg(
                        scale=(primpath_usd_pos_rot_scale[4], primpath_usd_pos_rot_scale[4], primpath_usd_pos_rot_scale[4]),
                        usd_path=f'assets/usds/{primpath_usd_pos_rot_scale[1]}',
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1e10),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        semantic_tags=[("class", type_selected)],
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(rot=primpath_usd_pos_rot_scale[3], pos=primpath_usd_pos_rot_scale[2]),
                )
                setattr(updated_cfg, f'object_cfg_list_{idx:04d}', rigid_cfg)
        
        self.cfg = updated_cfg
        # generate and spawn scene
        if self._is_scene_setup_from_cfg():
            # add entities from config
            self._add_entities_from_cfg(procedural_generation=True)

            sim_utils.bind_visual_material('/World/Lane', '/World/Looks/LaneMaterial')
            sim_utils.bind_visual_material('/World/WhiteLine', '/World/Looks/LaneWhiteLineSurfaceMaterial')
            sim_utils.bind_visual_material('/World/YellowLine', '/World/Looks/LaneYellowLineSurfaceMaterial')
            sim_utils.bind_visual_material('/World/Sidewalk', '/World/Looks/SidewalkMaterial')
            sim_utils.bind_visual_material('/World/NearRoad', '/World/Looks/SidewalkNMaterial')
            sim_utils.bind_visual_material('/World/NearBuffer', '/World/Looks/SidewalkNBMaterial')
            sim_utils.bind_visual_material('/World/FarFromBuffer', '/World/Looks/SidewalkFBMaterial')
            sim_utils.bind_visual_material('/World/FarFromRoad', '/World/Looks/SidewalkFMaterial')
            sim_utils.bind_visual_material('/World/HouseRegion', '/World/Looks/SidewalkHMaterial')
            
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/ground/Environment'))
            prim.SetActive(False)
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/Lane/Environment'))
            prim.SetActive(False)
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/WhiteLine/Environment'))
            prim.SetActive(False)
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/YellowLine/Environment'))
            prim.SetActive(False)
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/Sidewalk/Environment'))
            prim.SetActive(False)
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/NearRoad/Environment'))
            prim.SetActive(False)
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/NearBuffer/Environment'))
            prim.SetActive(False)
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/FarFromBuffer/Environment'))
            prim.SetActive(False)
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/FarFromRoad/Environment'))
            prim.SetActive(False)
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/HouseRegion/Environment'))
            prim.SetActive(False)
         
    def generate_async_procedural_scene(self):
        from metaurban.manager.traffic_manager import TrafficMode
        from metaurban.manager.sidewalk_manager import AssetManager
        from metaurban.manager.humanoid_manager import PGBackgroundSidewalkAssetsManager

        from metaurban.utils import clip, Config
        from metaurban.engine.engine_utils import set_global_random_seed
        from metaurban.component.map.pg_map import parse_map_config, MapGenerateMethod

        
        from metaurban.engine.base_engine import BaseEngine
        from urbansim.utils import BASE_DEFAULT_CONFIG
        from urbansim.utils.map_manager import PGMapManager
        
        # configure the logical engine
        config = Config(BASE_DEFAULT_CONFIG)
        config.register_type("map", str, int)
        config["map_config"].register_type("config", None)
        config.update(dict(
            use_render=False, image_observation=False, interface_panel=False,
            
            # traffic related
            show_mid_block_map=False,
            traffic_mode=TrafficMode.Trigger,
            random_traffic=False,
            traffic_density=0.0,
            traffic_vehicle_config=dict(
            show_navi_mark=False,
            show_dest_mark=False,
            enable_reverse=False,
            show_lidar=False,
            show_lane_line_detector=False,
            show_side_detector=False,),
            
             # pedestrian related
            spawn_human_num=1,
            spawn_wheelchairman_num=0,
            spawn_edog_num = 0,
            spawn_erobot_num=0,
            spawn_drobot_num=0,
            max_actor_num=1,))
        
        default_config_copy = Config(config, unchangeable=True)
        
        merged_config = config.update(self.cfg.pg_config, True, ["agent_configs", "sensors"])
        merged_config["map_config"] = parse_map_config(
            easy_map_config=config["map"], new_map_config=config["map_config"], default_config=default_config_copy
        )
        
        PG_CONFIG = Config(merged_config, unchangeable=False)
        
        cls = BaseEngine
        cls.singleton = BaseEngine(PG_CONFIG)
        self.logical_engine = cls.singleton
        self.logical_engine.register_manager('map_manager', PGMapManager())
        if self.cfg.pg_config['type'] != 'clean':
            self.logical_engine.register_manager('asset_manager', AssetManager())
        # if self.cfg.pg_config['type'] == 'dynamic':
        #     self.logical_engine.register_manager('human_manager', PGBackgroundSidewalkAssetsManager())
        print(f'[INFO] Logical engine is created.')
        # reset engine
        engine = self.logical_engine
        set_global_random_seed((engine.global_config['seed'] + engine.gets_start_index(engine.global_config)) % engine.global_config['num_scenarios'])
        try:
            engine.reset()
        except:
            engine.seed(((0) % PG_CONFIG['num_scenarios']) + engine.global_config['start_seed'])
            engine.reset()
        print(f'[INFO] Logical engine is reset.')
        
        # setup object cache
        self.setup_object_cache()
        print(f'[INFO] Object cache is setup.')
        
        self.polygons_of_lane = []
        self.white_line_polygons = []
        self.yellow_line_polygons = []
        self.nb_polygons = []
        self.n_polygons = []
        self.s_polygons = []
        self.fb_polygons = []
        self.f_polygons = []
        self.h_polygons = []
        self.obj_prim = []
        for idx in range(self.num_envs):
            self.set_sidewalk(engine.global_config['sidewalk_type'])
            # initialize polygon
            polyline_polygon_dict = self.set_polyline_polygon()
            self.polygons_of_lane += [polyline_polygon_dict['polygons']]
            self.white_line_polygons += [polyline_polygon_dict['white_line_polygons']]
            self.yellow_line_polygons += [polyline_polygon_dict['lane_yellow_line_polygons']]
            self.nb_polygons += [polyline_polygon_dict['near_road_buffer_polygons']]
            self.n_polygons += [polyline_polygon_dict['near_road_sidewalk_polygons']]
            self.s_polygons += [polyline_polygon_dict['sidewalk_polygons']]
            self.fb_polygons += [polyline_polygon_dict['far_road_buffer_polygons']]
            self.f_polygons += [polyline_polygon_dict['far_road_sidewalk_polygons']]
            self.h_polygons += [polyline_polygon_dict['house_region_polygons']]
            
            # generate objects
            obj_info_dict = self.set_async_objects(idx)
            obj_prim = obj_info_dict['obj_prim_path_list']
            self.obj_prim += obj_prim
            
            engine.seed(((engine.current_seed + 1) % PG_CONFIG['num_scenarios']) + engine.global_config['start_seed'])
            try:
                engine.reset()
            except Exception as e:
                engine.seed(((0) % PG_CONFIG['num_scenarios']) + engine.global_config['start_seed'])
                engine.reset()
        all_rigid_objects_config = RigidObjectCfg(
            prim_path="/World/envs/env_*/Object_*",
            spawn=DiverseAssetCfg(
                assets_cfg=self.obj_prim
            )
        )

        updated_cfg = copy.deepcopy(self.cfg)
        updated_cfg.object_cfg_list = all_rigid_objects_config
        self.cfg = updated_cfg
        
        # generate and spawn scene
        if self._is_scene_setup_from_cfg():
            # add entities from config
            self._add_entities_from_cfg(procedural_generation=True)
            
            sim_utils.bind_visual_material('/World/Lane', '/World/Looks/LaneMaterial')
            sim_utils.bind_visual_material('/World/WhiteLine', '/World/Looks/LaneWhiteLineSurfaceMaterial')
            sim_utils.bind_visual_material('/World/YellowLine', '/World/Looks/LaneYellowLineSurfaceMaterial')
            sim_utils.bind_visual_material('/World/Sidewalk', '/World/Looks/SidewalkMaterial')
            sim_utils.bind_visual_material('/World/NearRoad', '/World/Looks/SidewalkNMaterial')
            sim_utils.bind_visual_material('/World/NearBuffer', '/World/Looks/SidewalkNBMaterial')
            sim_utils.bind_visual_material('/World/FarFromBuffer', '/World/Looks/SidewalkFBMaterial')
            sim_utils.bind_visual_material('/World/FarFromRoad', '/World/Looks/SidewalkFMaterial')
            sim_utils.bind_visual_material('/World/HouseRegion', '/World/Looks/SidewalkHMaterial')
            
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/ground/Environment'))
            prim.SetActive(False)
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/Lane/Environment'))
            prim.SetActive(False)
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/WhiteLine/Environment'))
            prim.SetActive(False)
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/YellowLine/Environment'))
            prim.SetActive(False)
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/Sidewalk/Environment'))
            prim.SetActive(False)
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/NearRoad/Environment'))
            prim.SetActive(False)
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/NearBuffer/Environment'))
            prim.SetActive(False)
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/FarFromBuffer/Environment'))
            prim.SetActive(False)
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/FarFromRoad/Environment'))
            prim.SetActive(False)
            prim = stage.GetPrimAtPath(Sdf.Path(f'/World/HouseRegion/Environment'))
            prim.SetActive(False)
    
    def setup_object_dict(self):
        if self._is_scene_setup_from_cfg():
            self._remove_entities_from_cfg()
            
        # register dataset path
        prims_utils.create_prim("/World/Dataset", "Scope")
        proto_prim_paths = list()
        # get all assets from dataset
        asset_root_path = 'assets/usds/'
        all_assets = os.listdir(asset_root_path)
        all_assets = [i for i in all_assets if 'non_metric' not in i]
        import yaml
        with open('./assets//asset_config.yaml', "r") as file:
            asset_config = yaml.safe_load(file)
        asset_types = asset_config['type']
        asset_types_mapping = {}
        for k, v in asset_types.items():
            for sub_k in v.keys():
                asset_types_mapping[sub_k] = k + '_' + sub_k
        valid_assets = os.listdir('./assets/adj_parameter_folder/')
        self.asset_types_mapping = asset_types_mapping
    
    def setup_object_cache(self):
        if self._is_scene_setup_from_cfg():
            self._remove_entities_from_cfg()
            
        # register dataset path
        prims_utils.create_prim("/World/Dataset", "Scope")
        proto_prim_paths = list()
        # get all assets from dataset
        asset_root_path = 'assets/usds/'
        all_assets = os.listdir(asset_root_path)
        all_assets = [i for i in all_assets if 'non_metric' not in i]
        import yaml
        with open('./assets//asset_config.yaml', "r") as file:
            asset_config = yaml.safe_load(file)
        asset_types = asset_config['type']
        asset_types_mapping = {}
        for k, v in asset_types.items():
            for sub_k in v.keys():
                asset_types_mapping[sub_k] = k + '_' + sub_k
        valid_assets = os.listdir('./assets/adj_parameter_folder/')
        self.asset_types_mapping = asset_types_mapping
        # spawn all static objects in one dataset
        for asset_path in all_assets:
            usd_path = copy.deepcopy(asset_path)
            asset_valid = False
            for k, v in asset_types_mapping.items():
                if k in asset_path:
                    asset_path = asset_path.replace(k, v)
                    asset_path = asset_path.replace('usd', 'json')
                    asset_path = v + '-' + asset_path.split('_')[-1]
                    if asset_path in valid_assets:
                        asset_valid = True
                    break
            if not asset_valid:
                continue
            param_path = './assets/adj_parameter_folder/' + asset_path
            obj_info = json.load(open(param_path, 'rb'))
            prim_path = asset_path[:-5].replace('-', '_')
            proto_prim_path = f"/World/Dataset/Object_{prim_path}"
            proto_asset_config = sim_utils.UsdFileCfg(
                scale=(obj_info['scale'],obj_info['scale'],obj_info['scale']),
                usd_path=f"assets/usds/{usd_path}",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=min(obj_info.get('mass', 1000), 1000)),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                
                # init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., ),rot=(0.707, 0.707,0.0,0.0))
            )
            prim = proto_asset_config.func(proto_prim_path, proto_asset_config)
            prim.SetInstanceable(True)
            for child in prim.GetChildren():
                child.SetInstanceable(True)
            prim.SetActive(True)
            # save the proto prim path
            proto_prim_paths.append(proto_prim_path)
            # set the prim visibility
            if hasattr(proto_asset_config, "visible"):
                imageable = UsdGeom.Imageable(prim)
                if proto_asset_config.visible:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()
            # set the semantic annotations
            if hasattr(proto_asset_config, "semantic_tags") and proto_asset_config.semantic_tags is not None:
                # note: taken from replicator scripts.utils.utils.py
                for semantic_type, semantic_value in proto_asset_config.semantic_tags:
                    # deal with spaces by replacing them with underscores
                    semantic_type_sanitized = semantic_type.replace(" ", "_")
                    semantic_value_sanitized = semantic_value.replace(" ", "_")
                    # set the semantic API for the instance
                    instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                    sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                    # create semantic type and data attributes
                    sem.CreateSemanticTypeAttr()
                    sem.CreateSemanticDataAttr()
                    sem.GetSemanticTypeAttr().Set(semantic_type)
                    sem.GetSemanticDataAttr().Set(semantic_value)
            # activate rigid body contact sensors
            if hasattr(proto_asset_config, "activate_contact_sensors") and proto_asset_config.activate_contact_sensors:
                sim_utils.activate_contact_sensors(proto_prim_path, proto_asset_config.activate_contact_sensors)
    
    def set_sidewalk(self, sidewalk_type):
        self.sidewalk_type = sidewalk_type
        if self.sidewalk_type == 'Narrow Sidewalk':
            self.near_road_width = None
            self.near_road_buffer_width = PGDrivableAreaProperty.NARROW_SIDEWALK_NEAR_ROAD_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.NARROW_SIDEWALK_NEAR_ROAD_MAX_WIDTH - PGDrivableAreaProperty.NARROW_SIDEWALK_NEAR_ROAD_MIN_WIDTH)
            self.main_width = PGDrivableAreaProperty.NARROW_SIDEWALK_MAIN_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.NARROW_SIDEWALK_MAIN_MAX_WIDTH - PGDrivableAreaProperty.NARROW_SIDEWALK_MAIN_MIN_WIDTH)
            self.far_from_buffer_width = None
            self.far_from_width = None
            self.valid_house_width = PGDrivableAreaProperty.HOUSE_WIDTH
        elif self.sidewalk_type == 'Narrow Sidewalk with Trees':
            self.near_road_width = PGDrivableAreaProperty.NARROWT_SIDEWALK_NEAR_ROAD_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.NARROWT_SIDEWALK_NEAR_ROAD_MAX_WIDTH - PGDrivableAreaProperty.NARROWT_SIDEWALK_NEAR_ROAD_MIN_WIDTH)
            self.near_road_buffer_width = None
            self.main_width = PGDrivableAreaProperty.NARROWT_SIDEWALK_MAIN_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.NARROWT_SIDEWALK_MAIN_MAX_WIDTH - PGDrivableAreaProperty.NARROWT_SIDEWALK_MAIN_MIN_WIDTH)
            self.far_from_buffer_width = None
            self.far_from_width = None
            self.valid_house_width = PGDrivableAreaProperty.HOUSE_WIDTH
        elif self.sidewalk_type == 'Ribbon Sidewalk':
            self.near_road_width = PGDrivableAreaProperty.RIBBON_SIDEWALK_NEAR_ROAD_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.RIBBON_SIDEWALK_NEAR_ROAD_MAX_WIDTH - PGDrivableAreaProperty.RIBBON_SIDEWALK_NEAR_ROAD_MIN_WIDTH)
            self.near_road_buffer_width = None
            self.main_width = PGDrivableAreaProperty.RIBBON_SIDEWALK_MAIN_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.RIBBON_SIDEWALK_MAIN_MAX_WIDTH - PGDrivableAreaProperty.RIBBON_SIDEWALK_MAIN_MIN_WIDTH)
            self.far_from_buffer_width = None
            self.far_from_width = PGDrivableAreaProperty.RIBBON_SIDEWALK_FAR_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.RIBBON_SIDEWALK_FAR_MAX_WIDTH - PGDrivableAreaProperty.RIBBON_SIDEWALK_FAR_MIN_WIDTH)
            self.valid_house_width = PGDrivableAreaProperty.HOUSE_WIDTH
        elif self.sidewalk_type == 'Neighborhood 1':
            self.near_road_width = PGDrivableAreaProperty.NEIGHBORHOOD_SIDEWALK_NEAR_ROAD_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.NEIGHBORHOOD_SIDEWALK_NEAR_ROAD_MAX_WIDTH - PGDrivableAreaProperty.NEIGHBORHOOD_SIDEWALK_NEAR_ROAD_MIN_WIDTH)
            self.near_road_buffer_width = PGDrivableAreaProperty.NEIGHBORHOOD_SIDEWALK_BUFFER_NEAR_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.NEIGHBORHOOD_SIDEWALK_BUFFER_NEAR_MAX_WIDTH - PGDrivableAreaProperty.NEIGHBORHOOD_SIDEWALK_BUFFER_NEAR_MIN_WIDTH)
            self.main_width = PGDrivableAreaProperty.NEIGHBORHOOD_SIDEWALK_MAIN_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.NEIGHBORHOOD_SIDEWALK_MAIN_MAX_WIDTH - PGDrivableAreaProperty.NEIGHBORHOOD_SIDEWALK_MAIN_MIN_WIDTH)
            self.far_from_buffer_width = None
            self.far_from_width = None
            self.valid_house_width = PGDrivableAreaProperty.HOUSE_WIDTH
        elif self.sidewalk_type == 'Neighborhood 2':
            self.near_road_width = PGDrivableAreaProperty.NEIGHBORHOOD2_SIDEWALK_NEAR_ROAD_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.NEIGHBORHOOD2_SIDEWALK_NEAR_ROAD_MAX_WIDTH - PGDrivableAreaProperty.NEIGHBORHOOD2_SIDEWALK_NEAR_ROAD_MIN_WIDTH)
            self.near_road_buffer_width = None
            self.main_width = PGDrivableAreaProperty.NEIGHBORHOOD2_SIDEWALK_MAIN_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.NEIGHBORHOOD2_SIDEWALK_MAIN_MAX_WIDTH - PGDrivableAreaProperty.NEIGHBORHOOD2_SIDEWALK_MAIN_MIN_WIDTH)
            self.far_from_buffer_width = None
            self.far_from_width = PGDrivableAreaProperty.NEIGHBORHOOD2_SIDEWALK_BUFFER_FAR_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.NEIGHBORHOOD2_SIDEWALK_BUFFER_FAR_MAX_WIDTH - PGDrivableAreaProperty.NEIGHBORHOOD2_SIDEWALK_BUFFER_FAR_MIN_WIDTH)
            self.valid_house_width = PGDrivableAreaProperty.HOUSE_WIDTH
        elif self.sidewalk_type == 'Medium Commercial':
            self.near_road_width = PGDrivableAreaProperty.MediumCommercial_SIDEWALK_NEAR_ROAD_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.MediumCommercial_SIDEWALK_NEAR_ROAD_MAX_WIDTH - PGDrivableAreaProperty.MediumCommercial_SIDEWALK_NEAR_ROAD_MIN_WIDTH)
            self.near_road_buffer_width = None
            self.main_width = PGDrivableAreaProperty.MediumCommercial_SIDEWALK_MAIN_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.MediumCommercial_SIDEWALK_MAIN_MAX_WIDTH - PGDrivableAreaProperty.MediumCommercial_SIDEWALK_MAIN_MIN_WIDTH)
            self.far_from_buffer_width = None
            self.far_from_width = PGDrivableAreaProperty.MediumCommercial_SIDEWALK_FAR_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.MediumCommercial_SIDEWALK_FAR_MAX_WIDTH - PGDrivableAreaProperty.MediumCommercial_SIDEWALK_FAR_MIN_WIDTH)
            self.valid_house_width = PGDrivableAreaProperty.HOUSE_WIDTH
        elif self.sidewalk_type == 'Wide Commercial':
            self.near_road_width = PGDrivableAreaProperty.WideCommercial_SIDEWALK_NEAR_ROAD_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.WideCommercial_SIDEWALK_NEAR_ROAD_MAX_WIDTH - PGDrivableAreaProperty.WideCommercial_SIDEWALK_NEAR_ROAD_MIN_WIDTH)
            self.near_road_buffer_width = None
            self.main_width = PGDrivableAreaProperty.WideCommercial_SIDEWALK_MAIN_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.WideCommercial_SIDEWALK_MAIN_MAX_WIDTH - PGDrivableAreaProperty.WideCommercial_SIDEWALK_MAIN_MIN_WIDTH)
            self.far_from_buffer_width = PGDrivableAreaProperty.WideCommercial_SIDEWALK_MAIN_BUFFER_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.WideCommercial_SIDEWALK_MAIN_BUFFER_MAX_WIDTH - PGDrivableAreaProperty.WideCommercial_SIDEWALK_MAIN_BUFFER_MIN_WIDTH)
            self.far_from_width = PGDrivableAreaProperty.WideCommercial_SIDEWALK_FAR_MIN_WIDTH + np.random.uniform(0, 1) *  (PGDrivableAreaProperty.WideCommercial_SIDEWALK_FAR_MAX_WIDTH - PGDrivableAreaProperty.WideCommercial_SIDEWALK_FAR_MIN_WIDTH)
            self.valid_house_width = PGDrivableAreaProperty.HOUSE_WIDTH
            
    def construct_sidewalk_region(self, lane, direction, start_lat):
        nb_polygons = []
        n_polygons = []
        sidewalk_polygons = []
        fb_polygons = []
        f_polygons = []
        h_polygons = []
        if self.near_road_buffer_width is not None:
            nb_polygon = construct_continuous_polygon(lane, start_lat=start_lat, end_lat=start_lat + (direction * self.near_road_buffer_width))
            nb_polygons.extend(nb_polygon)
            start_lat = start_lat + (direction * self.near_road_buffer_width)
        if self.near_road_width is not None:
            n_polygon = construct_continuous_polygon(lane, start_lat=start_lat, end_lat=start_lat + (direction * self.near_road_width))
            n_polygons.extend(n_polygon)
            start_lat = start_lat + (direction * self.near_road_width)
        if self.main_width is not None:
            s_polygon = construct_continuous_polygon(lane, start_lat=start_lat, end_lat=start_lat + (direction * self.main_width))
            sidewalk_polygons.extend(s_polygon)
            start_lat = start_lat + (direction * self.main_width)
        if self.far_from_buffer_width is not None:
            fb_polygon = construct_continuous_polygon(lane, start_lat=start_lat, end_lat=start_lat + (direction * self.far_from_buffer_width))
            fb_polygons.extend(fb_polygon)
            start_lat = start_lat + (direction * self.far_from_buffer_width)
        if self.far_from_width is not None:
            f_polygon = construct_continuous_polygon(lane, start_lat=start_lat, end_lat=start_lat + (direction * self.far_from_width))
            f_polygons.extend(f_polygon)
            start_lat = start_lat + (direction * self.far_from_width)
        if self.valid_house_width is not None:
            h_polygon = construct_continuous_polygon(lane, start_lat=start_lat, end_lat=start_lat + (direction * self.valid_house_width))
            h_polygons.extend(h_polygon)
            start_lat = start_lat + (direction * self.valid_house_width)
            
        return nb_polygons, n_polygons, sidewalk_polygons, fb_polygons, f_polygons, h_polygons
    
    def set_polyline_polygon(self):
        from metaurban.utils.math import norm
        from metaurban.constants import TerrainProperty
        from metaurban.constants import MetaUrbanType, CamMask, PGLineType, PGLineColor, PGDrivableAreaProperty
        from metaurban.component.road_network import Road
        
        engine = self.logical_engine
        
        # lane polygons
        polygons = []
        for block in engine.map_manager.current_map.blocks:
            graph = block.block_network.graph
            for _from, to_dict in graph.items():
                for _to, lanes in to_dict.items():
                    for _id, lane in enumerate(lanes):
                        polygons_of_lane = construct_lane(lane, (_from, _to, _id))
                        polygons = polygons + polygons_of_lane
                        
        # lane lines
        white_line_polygons = []
        lane_yellow_line_polygons = []
        
        # sidewalk polygons
        nb_polygons = []
        n_polygons = []
        s_polygons = []
        f_polygons = []
        fb_polygons = []
        h_polygons = []
        
        for block in engine.map_manager.current_map.blocks:
            graph = block.block_network.graph
            for _from, to_dict in graph.items():
                for _to, lanes in to_dict.items():
                    for _id, lane in enumerate(lanes):
                        choose_side = [True, True] if _id == len(lanes) - 1 else [True, False]
                        if Road(_from, _to).is_negative_road() and _id == 0:
                            # draw center line with positive road
                            choose_side = [False, False]
                            
                        for idx, line_type, line_color, need, in zip([-1, 1], lane.line_types, lane.line_colors, choose_side):
                            if not need:
                                continue
                            lateral = idx * lane.width_at(0) / 2
                            if line_type == PGLineType.CONTINUOUS:
                                polygon_line = construct_continuous_line_polygon(lane, lateral, line_color, line_type)
                            elif line_type == PGLineType.BROKEN:
                                polygon_line = construct_broken_line_polygon(lane, lateral, line_color, line_type)
                            elif line_type == PGLineType.SIDE:
                                polygon_line = construct_continuous_line_polygon(lane, lateral, line_color, line_type)
                                tmp_nb_polygons, tmp_n_polygons, tmp_s_polygons, tmp_fb_polygons, tmp_f_polygons, tmp_h_polygons = self.construct_sidewalk_region(lane, direction=idx, start_lat=lateral)
                                nb_polygons = nb_polygons + tmp_nb_polygons
                                n_polygons = n_polygons + tmp_n_polygons
                                s_polygons = s_polygons + tmp_s_polygons
                                fb_polygons = fb_polygons + tmp_fb_polygons
                                f_polygons = f_polygons + tmp_f_polygons
                                h_polygons = h_polygons + tmp_h_polygons
                            elif line_type == PGLineType.GUARDRAIL:
                                polygon_line = construct_continuous_line_polygon(lane, lateral, line_color, line_type)
                                tmp_nb_polygons, tmp_n_polygons, tmp_s_polygons, tmp_fb_polygons, tmp_f_polygons, tmp_h_polygons = self.construct_sidewalk_region(lane, direction=idx, start_lat=lateral)
                                nb_polygons = nb_polygons + tmp_nb_polygons
                                n_polygons = n_polygons + tmp_n_polygons
                                s_polygons = s_polygons + tmp_s_polygons
                                fb_polygons = fb_polygons + tmp_fb_polygons
                                f_polygons = f_polygons + tmp_f_polygons
                                h_polygons = h_polygons + tmp_h_polygons
                            polygon_line = [np.stack([p[..., 0], p[..., 1]], axis=1) for p in polygon_line]
                            if line_color[1] == 200 / 255:#MetaUrbanType.is_yellow_line(line_type):
                                lane_yellow_line_polygons.extend(polygon_line)
                            elif MetaUrbanType.is_white_line(line_type) or MetaUrbanType.is_road_boundary_line(line_type):
                                white_line_polygons.extend(polygon_line)
        return {
            'polygons': polygons,
            'white_line_polygons': white_line_polygons,
            'lane_yellow_line_polygons': lane_yellow_line_polygons,
            'near_road_sidewalk_polygons': n_polygons,
            'near_road_buffer_polygons': nb_polygons,
            'sidewalk_polygons': s_polygons,
            'far_road_buffer_polygons': fb_polygons,
            'far_road_sidewalk_polygons': f_polygons,
            'house_region_polygons': h_polygons
        }
    
    def set_sync_objects(self):
        engine = self.logical_engine
        
        obj_position_list = []
        obj_prim_path_list = []
        prim_path_list = []
            
        valid_assets = os.listdir('./assets/adj_parameter_folder/')
        for obj_idx, (lane, lane_position, obj) in enumerate(engine.asset_manager.generated_objs):
            position = [lane_position[0], lane_position[1]] 
            usd_path = obj['filename'].replace('-', '_').replace(" ", "").split('.')[0] + '.usd'
            asset_path = usd_path
            asset_valid = False
            for k, v in self.asset_types_mapping.items():
                if k in asset_path:
                    asset_path = asset_path.replace(k, v)
                    asset_path = asset_path.replace('usd', 'json')
                    asset_path = v + '-' + asset_path.split('_')[-1]
                    if asset_path in valid_assets:
                        asset_valid = True
                    break
            if not asset_valid:
                continue
            prim_path = asset_path[:-5].replace('-', '_')
            prim_path=f"/World/envs/env_.*/" + f"Object_{prim_path}" + f'{obj_idx:04d}'
            # prim path in the sub env & init pos & init rot
            obj_prim_path_list.append([prim_path, usd_path, (position[0]+obj['pos0'],position[1]+obj['pos1'], LANE_HEIGHT + 0.02), (0.707, 0.707,0.0,0.0), obj['scale'], (position[0],position[1], LANE_HEIGHT + 0.02)])
            prim_path_list.append(prim_path)
            obj_position_list.append(
                [
                    position[0]+obj['pos0'],position[1]+obj['pos1']
                ]
            )
            
        return {
            'obj_position_list': obj_position_list,
            'obj_prim_path_list': obj_prim_path_list,
            'prim_path_list': prim_path_list
        }
        
    def set_async_objects(self, engine_idx):
        engine = self.logical_engine
        
        obj_position_list = []
        obj_prim_path_list = []
        prim_path_list = []
            
        valid_assets = os.listdir('./assets/adj_parameter_folder/')
        for obj_idx, (lane, lane_position, obj) in enumerate(engine.asset_manager.generated_objs):
            position = [lane_position[0], lane_position[1]] 
            usd_path = obj['filename'].replace('-', '_').replace(" ", "").split('.')[0] + '.usd'
            asset_path = usd_path
            asset_valid = False
            for k, v in self.asset_types_mapping.items():
                if k in asset_path:
                    asset_path = asset_path.replace(k, v)
                    asset_path = asset_path.replace('usd', 'json')
                    asset_path = v + '-' + asset_path.split('_')[-1]
                    if asset_path in valid_assets:
                        asset_valid = True
                    break
            if not asset_valid:
                continue
            prim_path = asset_path[:-5].replace('-', '_')
            prim_path=f"/World/envs/env_{engine_idx}/" + f"Object_{prim_path}" + f'{obj_idx:04d}'
            # prim path in the sub env & init pos & init rot
            obj_prim_path_list.append([prim_path, (position[0]+obj['pos0'],position[1]+obj['pos1'], LANE_HEIGHT + 0.02), (0.707, 0.707,0.0,0.0)])
            prim_path_list.append(prim_path)
            obj_position_list.append(
                [
                    position[0]+obj['pos0'],position[1]+obj['pos1']
                ]
            )
            
        return {
            'obj_position_list': obj_position_list,
            'obj_prim_path_list': obj_prim_path_list,
            'prim_path_list': prim_path_list
        }