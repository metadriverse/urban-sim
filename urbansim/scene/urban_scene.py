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

from .urban_scene_cfg import UrbanSceneCfg
from .utils import *

# from metaurban.engine.base_engine import BaseEngine
# from urbansim.utils.map_manager import PGMapManager
# from urbansim.utils.geometry import (construct_lane, 
#                                      construct_continuous_line_polygon, 
#                                      construct_continuous_polygon, 
#                                      construct_broken_line_polygon,
#                                      generate_random_road,
#                                      get_road_trimesh)
# from metaurban.manager.traffic_manager import TrafficMode
# from metaurban.manager.sidewalk_manager import AssetManager
# from metaurban.manager.humanoid_manager import PGBackgroundSidewalkAssetsManager

# from metaurban.utils import clip, Config
# from metaurban.engine.engine_utils import set_global_random_seed
# from urbansim.utils import BASE_DEFAULT_CONFIG
# from metaurban.component.map.pg_map import parse_map_config, MapGenerateMethod
# from metaurban.utils.math import norm
# from metaurban.constants import TerrainProperty
# from metaurban.constants import MetaUrbanType, CamMask, PGLineType, PGLineColor, PGDrivableAreaProperty
# from metaurban.component.road_network import Road

from urbansim.assets.rigid_object import RigidObject
from urbansim.assets.rigid_object_cfg import RigidObjectCfg

URBANSIM_PATH = os.environ.get('URBANSIM_PATH', './')

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR, check_file_path, read_file

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
                                                       'predefined', 
                                                       '3dgs',
                                                       'urban cousion'], "Invalid scenario type not in ['async procedural generation', 'sync procedural generation', 'limited async procedural generation', 'limited sync procedural generation', 'predefined', '3dgs']"
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
            
            # if asset_name == 'terrain_importer':
            #     self.terrain_importer = asset_cfg.class_type(asset_cfg)
            #     continue
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
                                    lane_mesh = trimesh.creation.extrude_polygon(polygon=Polygon(np.stack([polygon[..., 0] + center[0], polygon[..., 1] + center[1]], axis=1)), height=2.21, engine="triangle")
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
                                # combined_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, image='/opt/nvidia/mdl/vMaterials_2/Concrete/textures/mortar_diff.jpg')
                                self.procedural_generated_terrains[asset_name].import_mesh(f'lane_{idx:04d}', combined_mesh)
                        if 'whiteline' in asset_name:
                            for idx, center in enumerate(self.terrain_offsets):
                                mesh_list = []
                                for po_idx, polygon in enumerate(self.white_line_polygons[idx]):
                                    polygon = np.array(polygon).reshape(len(polygon), 2)
                                    lane_mesh = trimesh.creation.extrude_polygon(polygon=Polygon(np.stack([polygon[..., 0] + center[0], polygon[..., 1] + center[1]], axis=1)), height=2.23, engine="triangle")
                                    mesh_list.append(lane_mesh)
                                combined_mesh = trimesh.util.concatenate(mesh_list)
                                self.procedural_generated_terrains[asset_name].import_mesh(f'whiteline_{idx:04d}', combined_mesh)
                        if 'yellowline' in asset_name:
                            for idx, center in enumerate(self.terrain_offsets):
                                mesh_list = []
                                for po_idx, polygon in enumerate(self.yellow_line_polygons[idx]):
                                    polygon = np.array(polygon).reshape(len(polygon), 2)
                                    lane_mesh = trimesh.creation.extrude_polygon(polygon=Polygon(np.stack([polygon[..., 0] + center[0], polygon[..., 1] + center[1]], axis=1)), height=2.23, engine="triangle")
                                    mesh_list.append(lane_mesh)
                                combined_mesh = trimesh.util.concatenate(mesh_list)
                                self.procedural_generated_terrains[asset_name].import_mesh(f'yellowline_{idx:04d}', combined_mesh)
                                    
                        if 'sidewalk' in asset_name:
                            for idx, center in enumerate(self.terrain_offsets):
                                mesh_list = []
                                for po_idx, polygon in enumerate(self.s_polygons[idx]):
                                    polygon = np.array(polygon).reshape(len(polygon), 2)
                                    polygon = Polygon(np.stack([polygon[..., 0] + center[0], polygon[..., 1] + center[1]], axis=1))
                                    lane_mesh = trimesh.creation.extrude_polygon(polygon=polygon, height=2.21)
                                    uv = []
                                    mesh = lane_mesh
                                    for vertex in lane_mesh.vertices:
                                        u = (vertex[0] - np.min(mesh.vertices[:, 0])) / (np.max(mesh.vertices[:, 0]) - np.min(mesh.vertices[:, 0]))
                                        v = (vertex[1] - np.min(mesh.vertices[:, 1])) / (np.max(mesh.vertices[:, 1]) - np.min(mesh.vertices[:, 1]))
                                        uv.append([u, v])
                                    uv = np.array(uv)

                                    lane_mesh.visual.uv = uv
                                    mesh_list.append(lane_mesh)
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
                                    polygon = np.array(polygon).reshape(len(polygon), 2)
                                    lane_mesh = trimesh.creation.extrude_polygon(polygon=Polygon(np.stack([polygon[..., 0] + center[0], polygon[..., 1] + center[1]], axis=1)), height=2.21, engine="triangle")
                                    mesh_list.append(lane_mesh)
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
                                    polygon = np.array(polygon).reshape(len(polygon), 2)
                                    lane_mesh = trimesh.creation.extrude_polygon(polygon=Polygon(np.stack([polygon[..., 0] + center[0], polygon[..., 1] + center[1]], axis=1)), height=2.21, engine="triangle")
                                    mesh_list.append(lane_mesh)
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
                                    polygon = np.array(polygon).reshape(len(polygon), 2)
                                    lane_mesh = trimesh.creation.extrude_polygon(polygon=Polygon(np.stack([polygon[..., 0] + center[0], polygon[..., 1] + center[1]], axis=1)), height=2.21, engine="triangle")
                                    mesh_list.append(lane_mesh)
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
                                    polygon = np.array(polygon).reshape(len(polygon), 2)
                                    lane_mesh = trimesh.creation.extrude_polygon(polygon=Polygon(np.stack([polygon[..., 0] + center[0], polygon[..., 1] + center[1]], axis=1)), height=2.21, engine="triangle")
                                    mesh_list.append(lane_mesh)
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
                                    polygon = np.array(polygon).reshape(len(polygon), 2)
                                    lane_mesh = trimesh.creation.extrude_polygon(polygon=Polygon(np.stack([polygon[..., 0] + center[0], polygon[..., 1] + center[1]], axis=1)), height=2.21 if po_idx < len(self.h_polygons[idx]) -1 else 2.21)
                                    mesh_list.append(lane_mesh)
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
        # self.only_create_assets = True
        # self.generate_limited_async_procedural_scene()
        stage = omni.usd.get_context().get_stage()
        # if not hasattr(self, 'light_index'):
        #     self.light_index = 0
        # else:
        #     self.light_index += 1
        # if self.light_index == 11:
        #     self.light_index = 0
        # for i in range(11):
        #     if i == self.light_index:
        #         prim = stage.GetPrimAtPath(Sdf.Path(f'/World/skyLight_{i:04d}'))
        #         prim.SetActive(True)
        #     else:
        #         prim = stage.GetPrimAtPath(Sdf.Path(f'/World/skyLight_{i:04d}'))
        #         prim.SetActive(False)

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
        
    def write_dynamic_asset_state_to_sim(self):
        pass
    
    def dynamic_asset_animatable_state(self):
        pass
    
    def generate_and_spawn_map(self):
        pass
    
    def generate_and_spawn_terrains(self):
        pass
    
    def generate_and_spawn_static_objects(self):
        pass
    
    def generate_and_spawn_dynamic_objects(self):
        pass
    
    def generate_scene(self, remove_current_scene: bool = False):
        """
        Args:
            remove_current_scene (bool, optional): _description_. Defaults to False.
        """
        if self.cfg.scenario_generation_method in ['async procedural generation', 'sync procedural generation']:
            assert hasattr(self.cfg, 'pg_config'), "pg_config is required for procedural generation."
            print(f'[INFO] configuration for procedural generation: {self.cfg.pg_config}')
        
        if self.cfg.scenario_generation_method == "3dgs":
            self.generate_3dgs_scene()
        elif self.cfg.scenario_generation_method == "predefined":
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
        elif self.cfg.scenario_generation_method == "urban cousion":
            self.generate_urban_cousion_scene()
        else:
            raise ValueError(f"Invalid method: {self.cfg.scenario_generation_method}")
    
    def generate_predefined_scene(self):
        # terrain_usd_path = '/home/hollis/projects/temp_folder_for_isaacurban/terrain_tmp_mesh.obj'
        # # terrain_usd_path = '/home/sethzhao/Desktop/basic_terrains/intermediate_terrain.obj'
        # mesh = trimesh.load(terrain_usd_path)
        # mesh.vertices[:, 1] *= 1.2
        # mesh.vertices[:, 0] *= 1.2
        # mesh.vertices[:, 2] = np.median(mesh.vertices[:, 2]) 
        # mesh.vertices[:, 2] += 0.9
        
        # stage = omni.usd.get_context().get_stage()
        # prim = stage.GetPrimAtPath(Sdf.Path(f'/World/ground'))
        # prim.SetActive(False)
        
        # # mesh_stair_03 = trimesh.load("/home/sethzhao/Desktop/basic_terrains/stair_0.3.obj")
        # # mesh_stair_04 = trimesh.load("/home/sethzhao/Desktop/basic_terrains/stair_0.4.obj")
        # # mesh_stair_05 = trimesh.load("/home/sethzhao/Desktop/basic_terrains/stair_0.5.obj")
        # # mesh_stair_inv_04 = trimesh.load("/home/sethzhao/Desktop/basic_terrains/")
        # # mesh_rough_03 = trimesh.load("/home/hollis/projects/temp_folder_for_isaacurban/basic_terrains/rough_0.3.obj")
        # # mesh_rough_04 = copy.deepcopy(mesh_rough_03)
        # # mesh_rough_05 = copy.deepcopy(mesh_rough_03)
        # # mesh_rough_045 = trimesh.load("/home/sethzhao/Desktop/basic_terrains/rough_0.45.obj")
        # # mesh_roughgrid_02 = trimesh.load("/home/sethzhao/Desktop/basic_terrains/rough_grid_0.2.obj")
        # mesh_roughrandom_01 = trimesh.load("/home/hollis/projects/temp_folder_for_isaacurban/basic_terrains/rough_random01.obj")
        # mesh_roughrandom_02 = copy.deepcopy(mesh_roughrandom_01)
        # mesh_roughrandom_03 = copy.deepcopy(mesh_roughrandom_02)
        # # mesh_roughrandom_015 = trimesh.load("/home/sethzhao/Desktop/basic_terrains/rough_random_015.obj")
        
        # mixed_type_02 = trimesh.load("/home/hollis/projects/temp_folder_for_isaacurban/mixed_type_terrains/mixed_type_07.obj")
        # mixed_type_02.vertices[:, 0] *= 0.5
        # mixed_type_02.vertices[:, 1] *= 0.5
        # mixed_type_02.vertices[:, 2] *= 0.5
        # mixed_type_02.vertices[:, 0] += 16 
        # mixed_type_02.vertices[:, 1] += 3
        # mixed_type_02.vertices[:, 2] -= 0.3
        
        # mixed_type_03 = trimesh.load("/home/hollis/projects/temp_folder_for_isaacurban/mixed_type_terrains/mixed_type_05.obj")
        # mixed_type_03.vertices[:, 0] *= 0.6
        # mixed_type_03.vertices[:, 1] *= 0.6
        # mixed_type_03.vertices[:, 2] *= 0.6
        # mixed_type_03.vertices[:, 0] += -5.5 
        # mixed_type_03.vertices[:, 1] += -5
        # mixed_type_03.vertices[:, 2] -= -0.2
        
        # prim_type_01 = trimesh.load("/home/hollis/projects/temp_folder_for_isaacurban/mixed_type_terrains/prim_type_02.obj")
        # prim_type_01.vertices[:, 0] *= 0.6
        # prim_type_01.vertices[:, 1] *= 0.6
        # prim_type_01.vertices[:, 2] *= 0.6
        # prim_type_01.vertices[:, 0] += 2
        # prim_type_01.vertices[:, 1] += 8
        # prim_type_01.vertices[:, 2] -= -0.1
    
        # mesh_list = [mesh,mixed_type_02, mixed_type_03, prim_type_01]#, prim_type_01, prim_type_02]
        # prims_mesh_list_rand = [mesh_roughrandom_02, mesh_roughrandom_03]
        
        # # mesh_rough_05.vertices[:, 0] *= 0.3
        # # mesh_rough_05.vertices[:, 1] *= 0.2
        # # mesh_rough_05.vertices[:, 2] *= 0.3
        # # mesh_rough_05.vertices[:, 0] += 2
        # # mesh_rough_05.vertices[:, 1] += 6.25
        
        # mesh_roughrandom_01.vertices[:, 0] *= 0.3
        # mesh_roughrandom_01.vertices[:, 1] *= 0.2
        # mesh_roughrandom_01.vertices[:, 2] *= 0.3
        # mesh_roughrandom_01.vertices[:, 0] += -10
        # mesh_roughrandom_01.vertices[:, 1] += 8.15
        
        # prims_mesh_list = [mesh_roughrandom_01]

        # width = [8, 0, -8]
        # length = [5, -5, 8]
        # import itertools
        # combinations = list(itertools.product(width, length))
        # random.shuffle(combinations)
        # for i, each_mesh in enumerate(prims_mesh_list_rand):
        #     random_scale_x = random.uniform(0.1, 0.4)
        #     random_scale_y = random.uniform(0.1, 0.4)
        #     dy, dx = combinations[i]
        #     each_mesh.vertices[:, 0] *= random_scale_x
        #     each_mesh.vertices[:, 1] *= random_scale_y 
        #     each_mesh.vertices[:, 2] *= random_scale_x
            
        #     each_mesh.vertices[:, 0] += dx
        #     each_mesh.vertices[:, 1] += dy

        # mesh_list += prims_mesh_list
        # mesh_list += prims_mesh_list_rand
        
        # # mesh_list += assets_mesh_list
        # # mesh_list = [mesh]
        # mesh_v = trimesh.util.concatenate(mesh_list)

        # self.terrain_importer.import_mesh('mesh', mesh_v)
        
        # prims_mesh_list_dummy = copy.deepcopy(prims_mesh_list_rand + prims_mesh_list)
        # for mesh in prims_mesh_list_dummy:
        #     mesh.vertices[..., 2] += 0.001
        #     uvs = []
        #     for vertex in mesh.vertices:
        #         uv = [vertex[0], vertex[1]]
        #         uvs.append(uv)
        #     uvs = np.array(uvs)
        #     uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
        #     uvs *= 20
        #     mesh.visual.uvs = uvs
        # self.terrain_importer.import_mesh('mesh1', trimesh.util.concatenate(prims_mesh_list_dummy))
        # sim_utils.bind_visual_material(f'/World/input_terrain/mesh1', f'/World/Looks/material_tmp')
        
        # base_mesh_dummy = copy.deepcopy(mesh_list[0])
        # base_mesh_plane = copy.deepcopy(base_mesh_dummy)
        # base_mesh_dummy.vertices[..., 2] += 0.002
        # uvs = []
        # for vertex in base_mesh_dummy.vertices:
        #     uv = [vertex[0], vertex[1]]
        #     uvs.append(uv)
        # uvs = np.array(uvs)
        # uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
        # uvs *= 20
        # base_mesh_dummy.visual.uvs = uvs
        # self.terrain_importer.import_mesh('mesh_base_dummy', base_mesh_dummy)
        # sim_utils.bind_visual_material('/World/input_terrain/mesh_base_dummy'.format(i), f'/World/Looks/material_non_walkable_base')
        # # self._terrain_input.import_mesh('scene', mesh_asset_car)
        
        # base_mesh_plane.vertices[..., 2] = np.median(base_mesh_dummy.vertices[..., 2]) + 0.002
        # uvs = []
        # for vertex in base_mesh_plane.vertices:
        #     uv = [vertex[0], vertex[1]]
        #     uvs.append(uv)
        # uvs = np.array(uvs)
        # uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
        # uvs *= 20
        # base_mesh_plane.visual.uvs = uvs
        # self.terrain_importer.import_mesh('mesh_base_plane_dummy', base_mesh_plane)
        # sim_utils.bind_visual_material('/World/input_terrain/mesh_base_plane_dummy'.format(i), f'/World/Looks/material_walkable_base_plane')
        
        # other_mesh_list_dummy = copy.deepcopy(mesh_list[1:])
        # for i, mesh in enumerate(other_mesh_list_dummy):
        #     mesh.vertices[..., 2] += 0.001
        #     uvs = []
        #     for vertex in mesh.vertices:
        #         uv = [vertex[0], vertex[1]]
        #         uvs.append(uv)
        #     uvs = np.array(uvs)
        #     uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
        #     uvs *= 20
        #     mesh.visual.uvs = uvs
        #     self.terrain_importer.import_mesh('other_mesh_{}'.format(i), mesh)
        #     if i % 2 == 0:   
        #         sim_utils.bind_visual_material(f'/World/input_terrain/other_mesh_{i}', f'/World/Looks/material_walkable_stair_base')
        #     else:
        #         sim_utils.bind_visual_material(f'/World/input_terrain/other_mesh_{i}', f'/World/Looks/material_walkable_stair_base2')
        

        # stage = omni.usd.get_context().get_stage()
        # prim = stage.GetPrimAtPath(Sdf.Path(f'/World/input_terrain/Environment'))
        # prim.SetActive(False)
        return
    
    def generate_limited_sync_procedural_scene(self):
        generation_cfg = self.cfg.pg_config
        area_size = generation_cfg['map_region']
        torch.manual_seed(generation_cfg['seed'])
        np.random.seed(generation_cfg['seed'])
        random.seed(generation_cfg['seed'])
        # ground plane
        tmp_origin = self._default_env_origins[..., :2].reshape(self.num_envs, 2).cpu().numpy()
        if generation_cfg['type'] == 'map' or  generation_cfg['type'] == 'static' or  generation_cfg['type'] == 'dynamic':
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
                mesh, boundary_points, polyline_points = get_road_trimesh(x, y, area_size, boundary=(tmp_origin[env_idx, 1], tmp_origin[env_idx, 1] + area_size[1]))
                polyline_points_list.append(polyline_points)
                polygon_points.append([x, y, boundary_points[0], boundary_points[1]])
                area_polygon = np.array([(0, 0), (0, area_size[1] + buffer_width), (area_size[0] + buffer_width, area_size[1] + buffer_width), (area_size[0] + buffer_width, 0)]).astype(float)
                area_polygon[:, 0] += tmp_origin[env_idx, 0]
                area_polygon[:, 1] += tmp_origin[env_idx, 1]
                area_mesh = trimesh.creation.extrude_polygon(Polygon(area_polygon), height=2.21)
                area_mesh.fix_normals()
                mesh.fix_normals()
                uvs = []
                for vertex in mesh.vertices:
                    uv = [vertex[0], vertex[1]]
                    uvs.append(uv)
                uvs = np.array(uvs)
                uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
                uvs *= 20
                mesh.visual.uv = uvs
                uvs = []
                for vertex in area_mesh.vertices:
                    uv = [vertex[0], vertex[1]]
                    uvs.append(uv)
                uvs = np.array(uvs)
                uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
                uvs *= 20
                area_mesh.visual.uv = uvs

                walkable_region_polygon_list[generation_cfg['walkable_seed']].append(mesh)
                all_region_polygon_list[generation_cfg['non_walkable_seed']].append(area_mesh)

            self.polygon_points = polygon_points
            self.polylines_of_all_walkable_regions = torch.from_numpy(np.stack(polyline_points_list)).float().to(self.device)
            for i in range(len(walkable_region_polygon_list)):
                mesh_list = walkable_region_polygon_list[i]
                if len(mesh_list) == 0:
                    continue
                combined_mesh = trimesh.util.concatenate(mesh_list)
                uvs = []
                for vertex in combined_mesh.vertices:
                    uv = [vertex[0], vertex[1]]
                    uvs.append(uv)
                uvs = np.array(uvs)
                uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
                uvs *= 20
                combined_mesh.visual.uvs = uvs
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
                uvs = []
                for vertex in combined_mesh.vertices:
                    uv = [vertex[0], vertex[1]]
                    uvs.append(uv)
                uvs = np.array(uvs)
                uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
                uvs *= 20
                combined_mesh.visual.uvs = uvs
                self.all_region_list[i].import_mesh('mesh', combined_mesh)
                sim_utils.bind_visual_material(f'/World/NonWalkable_{i:03d}', f'/World/Looks/terrain_non_walkable_material_list_{i:03d}')
                sim_utils.bind_physics_material(f'/World/NonWalkable_{i:03d}', f'/World/Looks/terrain_non_walkable_material_list_{i:03d}')
                prim = stage.GetPrimAtPath(Sdf.Path(f'/World/NonWalkable_{i:03d}/Environment'))
                prim.SetActive(False)
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
            with open('metaurban_modified/asset_config.yaml', "r") as file:
                asset_config = yaml.safe_load(file)
            asset_types = asset_config['type']
            asset_types_mapping = {}
            for k, v in asset_types.items():
                for sub_k in v.keys():
                    asset_types_mapping[sub_k] = k + '_' + sub_k
            valid_assets = os.listdir('metaurban_modified/metaurban/assets/adj_parameter_folder/')
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
                param_path = 'metaurban_modified/metaurban/assets/adj_parameter_folder/' + asset_path
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
                building_positions = []
                objects = []
                object_positions = []
                for _ in range(num_objects * 1):
                    while True:
                        random_x = np.random.uniform(tmp_origin[env_idx, 0] + 1, tmp_origin[env_idx, 0] + area_size[0])
                        random_y = np.random.uniform(tmp_origin[env_idx, 1] + 1, tmp_origin[env_idx, 1] + area_size[1])
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
                    obj_prim_path_list.append([prim_path, (pos[0] - tmp_origin[env_idx, 0] + obj_info['pos0'],pos[1] - tmp_origin[env_idx, 1] + obj_info['pos1'], 2.11 + obj_info['pos2']), (0.707, 0.707,0.0,0.0)])
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
            add_reference_to_stage(usd_path="actions/synbody_idle426.fbx/synbody_idle426.fbx.usd", prim_path="/World/run")
            add_reference_to_stage(usd_path="actions/synbody_walking426.fbx/synbody_walking426.fbx.usd", prim_path="/World/walk")
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
            for human_idx in range(generation_cfg['num_pedestrian']):
                prim_path=f"/World/envs/env_{env_idx}/" + f"Dynamic_" + f'{human_idx:04d}'
                random_x = np.random.uniform(tmp_origin[env_idx, 0] + 1, tmp_origin[env_idx, 0] + area_size[0])
                random_y = np.random.uniform(tmp_origin[env_idx, 1] + 1, tmp_origin[env_idx, 1] + area_size[1])
                human_prim_path_list.append(
                        [
                            np.random.choice(dynamic_proto_prim_paths), prim_path, [random_x, random_y, 2.11 + 1.25], (0.5, 0.5, 0.5, 0.5)
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
            
            timeline = omni.timeline.get_timeline_interface()
            timeline.set_start_time(0)
            timeline.set_end_time(1.1)
            timeline.set_target_framerate(15)
            timeline.set_looping(True)
            for human_idx in range(generation_cfg['num_pedestrian']):
                prim_path=f"/World/envs/env_0/" + f"Dynamic_" + f'{human_idx:04d}'
                mesh_prim = stage.GetPrimAtPath(prim_path)
                UsdSkel.BindingAPI.Apply(mesh_prim)
                mesh_binding_api = UsdSkel.BindingAPI(mesh_prim)
                rel =  mesh_binding_api.CreateAnimationSourceRel()
                rel.ClearTargets(True)
                if np.random.rand() > 0.5:
                    rel.AddTarget("/World/run/SMPLX_neutral/root/pelvis0/SMPLX_neutral_Scene")
                else:
                    rel.AddTarget("/World/walk/SMPLX_neutral/root/pelvis0/SMPLX_neutral_Scene")
        
        # deactivate some prim
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
        if (generation_cfg['type'] == 'map' or  generation_cfg['type'] == 'static' or  generation_cfg['type'] == 'dynamic') and not self.only_create_assets:
            area_size = [area_size, area_size]
            
            polygon_points = []
            walkable_region_polygon_list = [[] for _ in range(len(self.cfg.terrain_importer_walkable_list))]
            all_region_polygon_list = [[] for _ in range(len(self.cfg.terrain_non_walkable_list))]
            polyline_points_list = []
            
            for env_idx in range(self.num_envs):
                torch.manual_seed(generation_cfg['seed'] + env_idx)
                np.random.seed(generation_cfg['seed'] + env_idx)
                random.seed(generation_cfg['seed'] + env_idx)
                        
                buffer_width = generation_cfg['buffer_width']
                x, y = generate_random_road(area_size)
                x, y = np.array(x), np.array(y)
                x += tmp_origin[env_idx, 0]
                y += tmp_origin[env_idx, 1]
                x = x.clip(tmp_origin[env_idx, 0], tmp_origin[env_idx, 0] + area_size[0])
                y = y.clip(tmp_origin[env_idx, 1], tmp_origin[env_idx, 1] + area_size[1])
                mesh, boundary_points, polyline_points = get_road_trimesh(x, y, area_size, boundary=(tmp_origin[env_idx, 1], tmp_origin[env_idx, 1] + area_size[1]), height=0.03)
                polyline_points_list.append(polyline_points)
                polygon_points.append([x, y, boundary_points[0], boundary_points[1]])
                area_polygon = np.array([(0, 0), (0, area_size[1] + buffer_width), (area_size[0] + buffer_width, area_size[1] + buffer_width), (area_size[0] + buffer_width, 0)]).astype(float)
                area_polygon[:, 0] += tmp_origin[env_idx, 0]
                area_polygon[:, 1] += tmp_origin[env_idx, 1]
                area_mesh = trimesh.creation.extrude_polygon(Polygon(area_polygon), height=0.01)

                walkable_region_polygon_list[(generation_cfg['walkable_seed'] + env_idx) % len(walkable_region_polygon_list)].append(mesh)
                all_region_polygon_list[(generation_cfg['non_walkable_seed'] + env_idx) % len(all_region_polygon_list)].append(area_mesh)

            self.polygon_points = polygon_points
            self.polylines_of_all_walkable_regions = torch.from_numpy(np.stack(polyline_points_list)).float().to(self.device)
            for i in range(len(walkable_region_polygon_list)):
                mesh_list = walkable_region_polygon_list[i]
                if len(mesh_list) == 0:
                    continue
                combined_mesh = trimesh.util.concatenate(mesh_list)
                uvs = []
                for vertex in combined_mesh.vertices:
                    uv = [vertex[0], vertex[1]]
                    uvs.append(uv)
                uvs = np.array(uvs)
                uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
                uvs *= 20
                combined_mesh.visual.uvs = uvs
                self.walkable_terrain_list[i].import_mesh('mesh', combined_mesh)
                sim_utils.bind_visual_material(f'/World/Walkable_{i:03d}', f'/World/Looks/terrain_walkable_material_list_{i:03d}')
                stage = omni.usd.get_context().get_stage()
                prim = stage.GetPrimAtPath(Sdf.Path(f'/World/Walkable_{i:03d}/Environment'))
                prim.SetActive(False)
            for i in range(len(all_region_polygon_list)):
                mesh_list = all_region_polygon_list[i]
                if len(mesh_list) == 0:
                    continue
                combined_mesh = trimesh.util.concatenate(mesh_list)
                uvs = []
                for vertex in combined_mesh.vertices:
                    uv = [vertex[0], vertex[1]]
                    uvs.append(uv)
                uvs = np.array(uvs)
                uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
                uvs *= 20
                combined_mesh.visual.uvs = uvs
                self.all_region_list[i].import_mesh('mesh', combined_mesh)
                sim_utils.bind_visual_material(f'/World/NonWalkable_{i:03d}', f'/World/Looks/terrain_non_walkable_material_list_{i:03d}')
                prim = stage.GetPrimAtPath(Sdf.Path(f'/World/NonWalkable_{i:03d}/Environment'))
                prim.SetActive(False)
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
            with open('metaurban_modified/asset_config.yaml', "r") as file:
                asset_config = yaml.safe_load(file)
            asset_types = asset_config['type']
            asset_types_mapping = {}
            for k, v in asset_types.items():
                for sub_k in v.keys():
                    asset_types_mapping[sub_k] = k + '_' + sub_k
            valid_assets = os.listdir('metaurban_modified/metaurban/assets/adj_parameter_folder/')
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
                param_path = 'metaurban_modified/metaurban/assets/adj_parameter_folder/' + asset_path
                obj_info = json.load(open(param_path, 'rb'))
                prim_path = asset_path[:-5].replace('-', '_')
                proto_prim_path = f"/World/Dataset/Object_{prim_path}"
                proto_asset_config = sim_utils.UsdFileCfg(
                    scale=(obj_info['scale'],obj_info['scale'],obj_info['scale']),
                    usd_path=f"assets/usds/{usd_path}",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
                    mass_props=sim_utils.MassPropertiesCfg(mass=min(obj_info.get('mass', 2), 2)),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    
                    # init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., ),rot=(0.707, 0.707,0.0,0.0))
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
                torch.manual_seed(generation_cfg['seed'] + env_idx)
                np.random.seed(generation_cfg['seed'] + env_idx)
                random.seed(generation_cfg['seed'] + env_idx)
                
                obj_position_list = []
                polygon_xy_in_world = self.polygon_points[env_idx]
                if not isinstance(area_size, list):
                    area_size = [area_size, area_size]
                x, y, upper, lower = polygon_xy_in_world[0], polygon_xy_in_world[1], polygon_xy_in_world[2], polygon_xy_in_world[3]
                buffer_width = 1.
                polyline_boundary = np.concatenate([np.column_stack((polygon_xy_in_world[0], upper + buffer_width)), np.column_stack((x[::-1], lower[::-1] - buffer_width))])
                polygon_boundary = Polygon(polyline_boundary)
                
                # PG: buildings & objects
                buildings = []
                building_positions = []
                objects = []
                object_positions = []
                for ni in range(num_objects * 1):
                    while True:
                        random_x = np.random.uniform(tmp_origin[env_idx, 0] + 3, tmp_origin[env_idx, 0] + area_size[0])
                        random_y = np.random.uniform(tmp_origin[env_idx, 1] + 3, tmp_origin[env_idx, 1] + area_size[1])
                        # point = Point(random_x, random_y)
                        # if polygon_boundary.contains(point):
                        #     flag = False
                        #     if len(objects) > 0:
                        #         for obj_cur in objects:
                        #             obj_position = [random_x, random_y, 0]
                        #             obj = trimesh.primitives.Sphere(radius=buffer_width, center=obj_position)
                        #             if not obj.intersection(obj_cur).vertices.shape[0] and not obj_cur.intersection(obj).vertices.shape[0]:
                        #                 objects.append(obj)
                        #                 object_positions.append(obj_position)
                        #                 flag = True
                        #                 break
                        #     else:
                        #         obj_position = [random_x, random_y, 0]
                        #         obj = trimesh.primitives.Sphere(radius=buffer_width, center=obj_position)
                        #         objects.append(obj)
                        #         object_positions.append(obj_position)
                        #         flag = True
                        #     if flag:
                        #         break
                        if int(area_size[0]) == 30 and num_objects == 30:
                            if ni < 10:
                                random_x = tmp_origin[env_idx, 0] + 5.0 + ni * 2.5 # 2.
                                random_y = np.random.uniform(tmp_origin[env_idx, 1] + 10, tmp_origin[env_idx, 1] + 20) # 15 17.5
                            elif ni < 20:
                                random_x = tmp_origin[env_idx, 0] + 5.0 + (ni - 10) * 2.5
                                random_y = np.random.uniform(tmp_origin[env_idx, 1] + 0, tmp_origin[env_idx, 1] + 10)
                            else:
                                random_x = tmp_origin[env_idx, 0] + 5.0 + (ni - 20) * 2.5
                                random_y = np.random.uniform(tmp_origin[env_idx, 1] + 20, tmp_origin[env_idx, 1] + 30)
                        elif int(area_size[0]) == 30 and num_objects == 60:
                            if ni < 20:
                                random_x = tmp_origin[env_idx, 0] + 5.0 + ni * 1.
                                random_y = np.random.uniform(tmp_origin[env_idx, 1] + 10, tmp_origin[env_idx, 1] + 20)
                            elif ni < 40:
                                random_x = tmp_origin[env_idx, 0] + 5.0 + (ni - 20) * 1.2
                                random_y = np.random.uniform(tmp_origin[env_idx, 1] + 0, tmp_origin[env_idx, 1] + 10)
                            else:
                                random_x = tmp_origin[env_idx, 0] + 5.0 + (ni - 40) * 1.2
                                random_y = np.random.uniform(tmp_origin[env_idx, 1] + 20, tmp_origin[env_idx, 1] + 30)
                        elif int(area_size[0]) == 30:
                            if ni < 10:
                                random_x = tmp_origin[env_idx, 0] + 5.0 + ni * 2.5 # 2.
                                random_y = np.random.uniform(tmp_origin[env_idx, 1] + 0, tmp_origin[env_idx, 1] + 10) # 15 17.5
                            elif ni < 20:
                                random_x = tmp_origin[env_idx, 0] + 5.0 + (ni - 10) * 2.5
                                random_y = np.random.uniform(tmp_origin[env_idx, 1] + 20, tmp_origin[env_idx, 1] + 30)
                            else:
                                random_x = tmp_origin[env_idx, 0] + 5.0 + ((ni - 20) % 10) * 2.
                                random_y = np.random.uniform(tmp_origin[env_idx, 1] + 8, tmp_origin[env_idx, 1] + 22)
                        # if ni == 0:
                        #     random_x = np.random.uniform(tmp_origin[env_idx, 0] + 15, tmp_origin[env_idx, 0] + 25)
                        #     random_y = np.random.uniform(tmp_origin[env_idx, 1] + 15, tmp_origin[env_idx, 1] + 20)
                        # if ni == 1:
                        #     random_x = np.random.uniform(tmp_origin[env_idx, 0] + 10, tmp_origin[env_idx, 0] + 15)
                        #     random_y = np.random.uniform(tmp_origin[env_idx, 1] + 15, tmp_origin[env_idx, 1] + 25)
                        # if ni == 2:
                        #     random_x = np.random.uniform(tmp_origin[env_idx, 0] + 10, tmp_origin[env_idx, 0] + 15)
                        #     random_y = np.random.uniform(tmp_origin[env_idx, 1] + 3, tmp_origin[env_idx, 1] + 10)
                        # if ni == 3:
                        #     random_x = np.random.uniform(tmp_origin[env_idx, 0] + 15, tmp_origin[env_idx, 0] + 20)
                        #     random_y = np.random.uniform(tmp_origin[env_idx, 1] + 10, tmp_origin[env_idx, 1] + 15)
                        # if ni == 0:
                        #     random_x = np.random.uniform(tmp_origin[env_idx, 0] + 3, tmp_origin[env_idx, 0] + 5)
                        #     random_y = np.random.uniform(tmp_origin[env_idx, 1] + 4, tmp_origin[env_idx, 1] + 5)
                        # if ni == 1:
                        #     random_x = np.random.uniform(tmp_origin[env_idx, 0] + 7, tmp_origin[env_idx, 0] + 8)
                        #     random_y = np.random.uniform(tmp_origin[env_idx, 1] + 0, tmp_origin[env_idx, 1] + 2)
                        # if ni == 2:
                        #     random_x = np.random.uniform(tmp_origin[env_idx, 0] + 7, tmp_origin[env_idx, 0] + 8)
                        #     random_y = np.random.uniform(tmp_origin[env_idx, 1] + 7, tmp_origin[env_idx, 1] + 9)
                        # if ni == 3:
                        #     random_x = np.random.uniform(tmp_origin[env_idx, 0] + 9, tmp_origin[env_idx, 0] + 10)
                        #     random_y = np.random.uniform(tmp_origin[env_idx, 1] + 3, tmp_origin[env_idx, 1] + 6)
                        #point = Point(random_x, random_y)
                        if True:#polygon_boundary.contains(point):
                            flag = True#False
                            if False:#len(objects) > 0:
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
                                #obj = trimesh.primitives.Sphere(radius=buffer_width, center=obj_position)
                                #objects.append(obj)
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
                proto_prim_paths_no_building = [p for p in proto_prim_paths_no_building if 'vege' not in p[0].lower()]
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
                    obj_prim_path_list.append([prim_path, (pos[0] - tmp_origin[env_idx, 0] + obj_info['pos0'],pos[1] - tmp_origin[env_idx, 1] + obj_info['pos1'], 0.05 + obj_info['pos2']), (0.707, 0.707,0.0,0.0)])
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
            pass
        
        # deactivate some prim
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(Sdf.Path(f'/World/ground/Environment'))
        prim.SetActive(False)
        prim = stage.GetPrimAtPath(Sdf.Path(f'/World/Obstacle_terrain/Environment'))
        prim.SetActive(False)
        
    
    def generate_async_procedural_scene(self):
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
        self.logical_engine.register_manager('asset_manager', AssetManager())
        # self.logical_engine.register_manager('human_manager', PGBackgroundSidewalkAssetsManager())
        print(f'[INFO] Logical engine is created.')
        
        # reset engine
        engine = self.logical_engine
        set_global_random_seed((engine.global_config['reset_seed'] + engine.gets_start_index(engine.global_config)) % engine.global_config['num_scenarios'])
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
    
    def generate_sync_procedural_scene(self):
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
            max_actor_num=1,
            
            walk_on_all_regions=False,
            ))
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
        if self.logical_engine.global_config['object_density'] > 0:
            self.logical_engine.register_manager('asset_manager', AssetManager())
            # self.logical_engine.register_manager('human_manager', PGBackgroundSidewalkAssetsManager())
        print(f'[INFO] Logical engine is created.')
        
        # reset engine
        engine = self.logical_engine
        set_global_random_seed((engine.global_config['reset_seed'] + engine.gets_start_index(engine.global_config)) % engine.global_config['num_scenarios'])
        engine.reset()
        print(f'[INFO] Logical engine is reset.')
        
        # setup object cache
        # self.setup_object_cache()
        # print(f'[INFO] Object cache is setup.')
        self.setup_object_dict()
        print(f'[INFO] Object cache is not setup because sync simulation does not need caching but only omniverse mechanism.')
        
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
        
        # get mask
        start_x, start_y = 5., -5.
        if self.logical_engine.global_config['object_density'] > 0:
            mask = engine.walkable_regions_mask
            start_end_regions_mask = engine.start_end_regions_mask
            mask_translate = engine.mask_translate
            from metaurban.policy.get_planning import get_planning
            #start_points, end_points = random_start_and_end_points(mask[:, :, 0], mask_translate, 1, starts_init=[(start_x + mask_translate[0], start_y + mask_translate[1])])
            start_points, end_points = random_start_and_end_points(start_end_regions_mask[:, :, 0], mask_translate, 1)
            # start_points = [(start_x + mask_translate[0], start_y + mask_translate[1])]
            print(f'[INFO] Planning is started: {start_points} -> {end_points}')
            time_length_lists, nexts_list, speed_list, earliest_stop_pos_list = get_planning(
                [start_points],
                
                [mask],
                
                [end_points],
                
                [len(start_points)],
                
                1
            )
            position_list = np.array(nexts_list).reshape(-1, 2)
            position_list[..., 0] -= mask_translate[0]
            position_list[..., 1] -= mask_translate[1]
            print(f'[INFO] Planning is done.')
            print(f'[INFO] Position: {position_list[0]} -> {position_list[-1]}')

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.imshow(np.flipud(mask), origin='lower')   ######
        ax.scatter([np.array(start_points).reshape(2, )[0]], [np.array(start_points).reshape(2, )[1]], marker='o', color='red')
        ax.scatter([position_list[0, 0] + mask_translate[0]], [position_list[0, 1] + mask_translate[1]], marker='o')
        for t in range(len(position_list) // 10):
            ax.scatter([position_list[min(t * 10, len(position_list) - 1), 0] + mask_translate[0]], [position_list[min(t * 10, len(position_list) - 1), 1] + mask_translate[1]], marker='.', c='green')
        ax.scatter([position_list[-1, 0] + mask_translate[0]], [position_list[-1, 1] + mask_translate[1]], marker='x')
        ax.scatter([np.array(end_points).reshape(2, )[0]], [np.array(end_points).reshape(2, )[1]], marker='*', color='red')
        plt.show()
        # plt.savefig('./tmp.png')
        
        # initialize objects
        updated_cfg = copy.deepcopy(self.cfg)
        all_usd_files = os.listdir('assets/usds/')
        all_json_files = os.listdir('metaurban_modified/metaurban/assets/adj_parameter_folder/')
        usd_json_dict = {}
        for file in all_json_files:
            json_state = json.load(open('metaurban_modified/metaurban/assets/adj_parameter_folder/' + file, 'rb'))
            try:
                usd_name = json_state['filename'].replace('.glb', '.usd').replace('-', '_').replace(" ", "")
            except:
                # {'TIRE_RADIUS': 0.6, 'TIRE_WIDTH': 0.25, 'MASS': 1100, 'LATERAL_TIRE_TO_CENTER': 1.1953352769679304, 
                # 'FRONT_WHEELBASE': 2.7690288713910762, 'REAR_WHEELBASE': 2.087126137841352, 'CHASSIS_TO_WHEEL_AXIS': 0.0, 'TIRE_SCALE': 1.8, 'TIRE_OFFSET': -0.5801886792452831, 'MODEL_PATH': 'test/firerelated-aac70737c4f5478ebf9e14387a123214.glb', 'MODEL_SCALE': [1.1236023611051744, 1.1236023611051744, 1.1236023611051744], 'MODEL_OFFSET': [0, 0, -1.5], 'MODEL_HPR': [90.0, 0, 0], 'LENGTH': 7.570067405700684, 'HEIGHT': 3.144643008708954, 'WIDTH': 2.81737744808197, 'general': {'length': 7.570067405700684, 'width': 2.81737744808197, 'height': 3.144643008708954, 'bounding_box': [[3.907700300216675, 1.4086887836456299], [3.907700300216675, -1.4086886644363403], [-3.662367105484009, -1.4086886644363403], [-3.662367105484009, 1.4086887836456299]], 'center': [0.12266659736633301, 5.960464477539063e-08], 'color': 'Red', 'general_type': 'vehicle', 'detail_type': 'FireTruck'}}
                usd_name = json_state['MODEL_PATH'].replace('.usd', '.usd').replace('-', '_').replace(" ", "").split('/')[-1]
            usd_json_dict[usd_name] = json_state
        if engine.global_config['object_density'] > 0:
            self.sync_obejct_dict = self.set_sync_objects()
            import pickle
            with open('./types.pkl', 'rb') as f:
                types = pickle.load(f)
            for idx, primpath_usd_pos_rot_scale in enumerate(self.sync_obejct_dict['obj_prim_path_list']):
                try:
                    type_selected = [t for t in types if t in primpath_usd_pos_rot_scale[1]][0]
                except:
                    type_selected = 'unknown'
                np.random.seed(engine.global_config.get('instance_seed', 0))
                if engine.global_config.get('random_re_instance', False):
                    if 'building' in primpath_usd_pos_rot_scale[1].lower():
                        pass
                    else:
                        try:
                            files_to_be_selected = [f for f in all_usd_files if type_selected in f and 'non_metric' not in f and f != primpath_usd_pos_rot_scale[1]]
                            selected_replace_file = np.random.choice(files_to_be_selected)
                            json_state = usd_json_dict[selected_replace_file]
                            json_raw_state = usd_json_dict[primpath_usd_pos_rot_scale[1]]
                            B_orig = np.array(json_raw_state['general']['bounding_box'])
                            B_new = np.array(json_state['general']['bounding_box'])
                            W_orig = np.max(B_orig[:, 0]) - np.min(B_orig[:, 0])
                            H_orig = np.max(B_orig[:, 1]) - np.min(B_orig[:, 1])

                            W_new = np.max(B_new[:, 0]) - np.min(B_new[:, 0])
                            H_new = np.max(B_new[:, 1]) - np.min(B_new[:, 1])

                            s_x = W_orig / W_new
                            s_y = H_orig / H_new
                            
                            primpath_usd_pos_rot_scale[-1] = json_state['scale'] * s_x
                            primpath_usd_pos_rot_scale[1] = selected_replace_file
                            primpath_usd_pos_rot_scale[2] = [primpath_usd_pos_rot_scale[2][0] - json_raw_state['pos0'] + json_state['pos0'], primpath_usd_pos_rot_scale[2][1] - json_raw_state['pos1'] + json_state['pos1'], 2.11 + json_state['pos2']]
                        except:
                            pass
                # if 'tree' in primpath_usd_pos_rot_scale[1].lower():
                #     root_dir = f"{NVIDIA_NUCLEUS_DIR}/Assets/Vegetation/Trees/"
                #     with open("tree_usd.txt", "r") as f:
                #         valid_usd = [line.strip() for line in f.readlines() if line.strip()]
                #     chosen_usd = np.random.choice(valid_usd)
                #     rigid_cfg = RigidObjectCfg(
                #         prim_path=primpath_usd_pos_rot_scale[0],
                #         spawn=sim_utils.UsdFileCfg(
                #             scale=(0.01, 0.01, 0.01),
                #             usd_path=root_dir + chosen_usd,
                #             # rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                #             # mass_props=sim_utils.MassPropertiesCfg(mass=1e10),
                #             # collision_props=sim_utils.CollisionPropertiesCfg(),
                #             semantic_tags=[("class", type_selected)],
                #         ),
                #         init_state=RigidObjectCfg.InitialStateCfg(rot=(0., 0., 0., 0.), pos=primpath_usd_pos_rot_scale[5]),
                #     )
                if 'tree' not in primpath_usd_pos_rot_scale[0].lower():
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
            if engine.global_config['object_density'] > 0:
                self.position_list = position_list
                self.start_points = start_points
                self.end_points = end_points
                self.mask = mask
                self.mask_translate = mask_translate
            if engine.global_config.get('random_re_texturing', False):
                for idx, primpath_usd_pos_rot_scale in enumerate(self.sync_obejct_dict['obj_prim_path_list']):
                    np.random.seed(engine.global_config.get('instance_seed', 0))
                    chosen_material = np.random.choice(
                        [
                           f'/World/Looks/Random_{i:02d}'for i in range(20)
                        ]
                    )
                    sim_utils.bind_visual_material(primpath_usd_pos_rot_scale[0].replace('.*', '0'), chosen_material)
            # sim_utils.bind_visual_material('/World/envs/env_0/trash_bin_6', '/World/Looks/SidewalkNMaterial')
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
         
    def generate_3dgs_scene(self):
        pass
    
    def generate_urban_cousion_scene(self):
        assert hasattr(self.cfg, 'layout_from_video_path'), 'Please set layout_from_video_path in config'
        if not hasattr(self.cfg, 'use_random_cousin_instances'):
            self.cfg.use_random_cousin_instances = False
            print(f'[INFO] use_random_cousin_instances is set to False')
        if self.cfg.use_random_cousin_instances:
            if not hasattr(self.cfg, 'no_cousin_class_names'):
                self.cfg.no_cousin_class_names = []
                print(f'[INFO] no_cousin_class_names is set to []')
            scene_folder = self.cfg.scene_folder
            cname2cousin_info = {}
            for cname_folder in os.listdir(f"{scene_folder}/urban_verse_assets_seq1"):
                this_assets_info_list = []
                for asset_folder in os.listdir(f"{scene_folder}/urban_verse_assets_seq1/{cname_folder}"):
                    this_assets_info = {
                        "glb_path": f"{scene_folder}/urban_verse_assets_seq1/{cname_folder}/{asset_folder}/adjusted_asset_scaled_bottomed.glb",
                        "annotation_path": f"{scene_folder}/urban_verse_assets_seq1/{cname_folder}/{asset_folder}/annotations.json",
                    }
                    this_assets_info_list.append(this_assets_info)

                cname_key = cname_folder.lower().replace("_", " ")
                cname2cousin_info[cname_key] = this_assets_info_list
            print('[INFO] cname2cousin_info is set to:')
            print(cname2cousin_info.keys())
            print('-----*****************************-----')

            # Create an empty scene container
            scene_layout = SceneEntity()
            # Initialize the learned scene from disk
            # You can then play with this SceneEntity class
            scene_layout.load_from_serializable(self.cfg.layout_from_video_path)
            num_instances = len(scene_layout.object_list)
            for inst_id in range(num_instances):
                cname = scene_layout.object_list[inst_id]['class_name']

                if cname in self.cfg.no_cousin_class_names:
                    scene_layout.object_list[inst_id]['bound_cousins'] = []
                    continue

                bound_cousins = cname2cousin_info.get(cname, None)
                if bound_cousins is None:
                    print(f"No cousin found for {cname}")
                scene_layout.object_list[inst_id]['bound_cousins'] = bound_cousins
            # Save and Reload
            scene_layout.dump_to_serializable(f"{scene_folder}/scene_entity_bound_cousins.pkl.gz")
            scene_layout_with_cousins = SceneEntity()

            # Initialize the learned scene from disk
            # You can then play with this SceneEntity class
            scene_layout_with_cousins.load_from_serializable(f"{scene_folder}/scene_entity_bound_cousins.pkl.gz")
            num_instances = len(scene_layout_with_cousins.object_list)
            
            # analyze the scene to get the global scale
            height_list_of_the_video = []
            height_list_of_the_instance = []
            chosen_cousion = {}
            for inst_id in range(num_instances): # instance in the scenario (bicycle1, bicycle2, bus1, bus2, etc.)
                cousin_info = scene_layout_with_cousins.object_list[inst_id]['bound_cousins'] # each bicycle has a list of cousins
                not_included_keys = []
                for k, v in scene_layout_with_cousins.object_list[inst_id].items():
                    if 'open3d' in str(type(v)):
                        not_included_keys.append(k)
                bbox_center_from_video = np.mean(scene_layout_with_cousins.object_list[inst_id]['aligned_bbox_np'], axis=0)
                if len(cousin_info) > 0:
                    cousion_instance = np.random.choice(cousin_info)
                else:
                    cousion_instance = None
                if cousion_instance is not None:
                    chosen_cousion[inst_id] = [cousion_instance, bbox_center_from_video]
                    annotation_path = cousion_instance['annotation_path']
                    if 'building' in annotation_path.lower():
                        continue
                    annotation = json.load(open(annotation_path, 'rb'))
                    height_list_of_the_video.append(bbox_center_from_video[2])
                    height_list_of_the_instance.append(annotation['height'])
            assert len(height_list_of_the_video) > 0, 'No height information found in the video'
            assert len(height_list_of_the_instance) == len(height_list_of_the_video), 'Height information mismatch between video and instance'
            self.global_scale = np.mean(np.array(height_list_of_the_instance) / np.array(height_list_of_the_video)) # instance / height
            print(f'[INFO] global_scale is set to {self.global_scale}')
            updated_cfg = copy.deepcopy(self.cfg)
            
            # generate road
            road_height = 0.3
            road_mesh_list = []
            for inst_id in range(num_instances): # instance in the scenario (bicycle1, bicycle2, bus1, bus2, etc.)
                instance = scene_layout.object_list[inst_id]
                cname = instance['class_name']
                if cname in self.cfg.road_class_names:
                    bbox_center_from_video = scene_layout_with_cousins.object_list[inst_id]['aligned_bbox_np'] * self.global_scale#[:, :2]
                    bbox_center2d_from_video = bbox_center_from_video[::2, :2]
                    lane_mesh = trimesh.creation.extrude_polygon(polygon=Polygon(bbox_center2d_from_video), height=road_height, engine="triangle")
                    road_mesh_list += [lane_mesh]
            if len(road_mesh_list) > 0:
                combined_mesh = trimesh.util.concatenate(road_mesh_list)
                uvs = []
                for vertex in combined_mesh.vertices:
                    uv = [vertex[0], vertex[1]]
                    uvs.append(uv)
                uvs = np.array(uvs)
                uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
                uvs *= 20
                combined_mesh.visual.uvs = uvs
                
                chosen_lane_mesh_idx = 1
                self.all_region_list[chosen_lane_mesh_idx].import_mesh('mesh', combined_mesh)
                sim_utils.bind_visual_material(f'/World/NonWalkable_{chosen_lane_mesh_idx:03d}', f'/World/Looks/terrain_non_walkable_material_list_{chosen_lane_mesh_idx:03d}')
            # generate sidewalk
            sidewalke_height = 0.32
            sidewalke_mesh_list = []
            for inst_id in range(num_instances): # instance in the scenario (bicycle1, bicycle2, bus1, bus2, etc.)
                instance = scene_layout.object_list[inst_id]
                cname = instance['class_name']
                if cname in self.cfg.sidewalk_class_names:
                    bbox_center_from_video = scene_layout_with_cousins.object_list[inst_id]['aligned_bbox_np'] * self.global_scale#[:, :2]
                    bbox_center2d_from_video = bbox_center_from_video[::2, :2]
                    lane_mesh = trimesh.creation.extrude_polygon(polygon=Polygon(bbox_center2d_from_video), height=sidewalke_height, engine="triangle")
                    sidewalke_mesh_list += [lane_mesh]
            if len(sidewalke_mesh_list) > 0:
                combined_mesh = trimesh.util.concatenate(sidewalke_mesh_list)
                uvs = []
                for vertex in combined_mesh.vertices:
                    uv = [vertex[0], vertex[1]]
                    uvs.append(uv)
                uvs = np.array(uvs)
                uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
                uvs *= 20
                combined_mesh.visual.uvs = uvs
                
                chosen_lane_mesh_idx = 1
                self.walkable_terrain_list[chosen_lane_mesh_idx].import_mesh('mesh', combined_mesh)
                sim_utils.bind_visual_material(f'/World/Walkable_{chosen_lane_mesh_idx:03d}', f'/World/Looks/terrain_non_walkable_material_list_{chosen_lane_mesh_idx:03d}')
            stage = omni.usd.get_context().get_stage()
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
            
            for k, v in chosen_cousion.items():
                
                # HACK: Building should be alinged with edge rather than center
                
                
                # raw info from parsing
                cousin_info = v[0]
                bbox_center_from_video = v[1]
                cousin_info['pos'] = bbox_center_from_video * self.global_scale
                cousin_info['pos'][-1] = road_height
                cousin_info['rot'] = [0.707, 0.707, 0., 0.]
                cousin_info['prim_path'] = f"/World/envs/env_.*/Object_{k}"
                glb_path = cousin_info['glb_path']
                asset_name = glb_path.split('/')[-2]
                usd_path = 'adj_asset_usds/' + asset_name + '/scaled.usd'
                
                # HACK: compute additional rotation
                
                
                
                rigid_object_cfg  = RigidObjectCfg(
                        prim_path=cousin_info['prim_path'],
                        spawn=sim_utils.UsdFileCfg(
                            scale=(1., 1., 1.),
                            usd_path=usd_path,
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                            mass_props=sim_utils.MassPropertiesCfg(mass=1e10),
                            collision_props=sim_utils.CollisionPropertiesCfg(),
                        ),
                        init_state=RigidObjectCfg.InitialStateCfg(rot=cousin_info['rot'], pos=cousin_info['pos']),
                    )
                rigid_object_cfg.prim_path = rigid_object_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                if hasattr(rigid_object_cfg, "collision_group") and rigid_object_cfg.collision_group == -1:
                    asset_paths = sim_utils.find_matching_prim_paths(rigid_object_cfg.prim_path)
                    self._global_prim_paths += asset_paths
                self._rigid_objects[f'{k}'] = rigid_object_cfg.class_type(rigid_object_cfg)
        else:
            raise NotImplementedError('Video layouts without cousions are not implemented yet')
    
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
        with open('metaurban_modified/asset_config.yaml', "r") as file:
            asset_config = yaml.safe_load(file)
        asset_types = asset_config['type']
        asset_types_mapping = {}
        for k, v in asset_types.items():
            for sub_k in v.keys():
                asset_types_mapping[sub_k] = k + '_' + sub_k
        valid_assets = os.listdir('metaurban_modified/metaurban/assets/adj_parameter_folder/')
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
        with open('metaurban_modified/asset_config.yaml', "r") as file:
            asset_config = yaml.safe_load(file)
        asset_types = asset_config['type']
        asset_types_mapping = {}
        for k, v in asset_types.items():
            for sub_k in v.keys():
                asset_types_mapping[sub_k] = k + '_' + sub_k
        valid_assets = os.listdir('metaurban_modified/metaurban/assets/adj_parameter_folder/')
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
            param_path = 'metaurban_modified/metaurban/assets/adj_parameter_folder/' + asset_path
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
            
        valid_assets = os.listdir('metaurban_modified/metaurban/assets/adj_parameter_folder/')
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
            obj_prim_path_list.append([prim_path, usd_path, (position[0]+obj['pos0'],position[1]+obj['pos1'], 2.21), (0.707, 0.707,0.0,0.0), obj['scale'], (position[0],position[1], 2.21)])
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
            
        valid_assets = os.listdir('metaurban_modified/metaurban/assets/adj_parameter_folder/')
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
            obj_prim_path_list.append([prim_path, (position[0]+obj['pos0'],position[1]+obj['pos1'], 2.21), (0.707, 0.707,0.0,0.0)])
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