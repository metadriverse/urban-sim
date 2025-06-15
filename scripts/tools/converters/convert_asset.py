# Copyright (c) 2022-2025, The UrbanSim Project Developers.
# Author: Honglin He
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a OBJ/STL/FBX into USD format.

The OBJ file format is a simple data-format that represents 3D geometry alone — namely, the position
of each vertex, the UV position of each texture coordinate vertex, vertex normals, and the faces that
make each polygon defined as a list of vertices, and texture vertices.

An STL file describes a raw, unstructured triangulated surface by the unit normal and vertices (ordered
by the right-hand rule) of the triangles using a three-dimensional Cartesian coordinate system.

FBX files are a type of 3D model file created using the Autodesk FBX software. They can be designed and
modified in various modeling applications, such as Maya, 3ds Max, and Blender. Moreover, FBX files typically
contain mesh, material, texture, and skeletal animation data.
Link: https://www.autodesk.com/products/fbx/overview


This script uses the asset converter extension from Isaac Sim (``omni.kit.asset_converter``) to convert a
OBJ/STL/FBX asset into USD format. It is designed as a convenience script for command-line use.


positional arguments:
  input               The path to the input mesh (.OBJ/.STL/.FBX) file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                    Show this help message and exit
  --make-instanceable,          Make the asset instanceable for efficient cloning. (default: False)
  --collision-approximation     The method used for approximating collision mesh. Defaults to convexDecomposition.
                                Set to \"none\" to not add a collision mesh to the converted mesh. (default: convexDecomposition)
  --mass                        The mass (in kg) to assign to the converted asset. (default: None)

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a mesh file into USD format.")
parser.add_argument("--input_file_list", type=str, default=None, help="The path to the input mesh file.")
parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=False,
    help="Make the asset instanceable for efficient cloning.",
)
parser.add_argument(
    "--collision-approximation",
    type=str,
    default="convexDecomposition",
    choices=["convexDecomposition", "convexHull", "boundingCube", "boundingSphere", "meshSimplification", "none"],
    help=(
        'The method used for approximating collision mesh. Set to "none" '
        "to not add a collision mesh to the converted mesh."
    ),
)
parser.add_argument(
    "--mass",
    type=float,
    default=None,
    help="The mass (in kg) to assign to the converted asset. If not provided, then no mass is added.",
)
parser.add_argument(
    "--id",
    type=int,
    default=None,
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os
import pickle
import tqdm

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.parent
print(f"ROOT_DIR: {ROOT_DIR}")

def main():
    root_dir = os.path.join(ROOT_DIR, 'assets', 'objects')
    valid_files = os.listdir(root_dir)
    valid_files.sort()
    file_names_parent = [f for f in valid_files]
    file_names = file_names_parent
    for asset in file_names:
        if '-' not in asset and " " not in asset:
            continue
        else:
            replaced_name = asset.replace('-', '_').replace(" ", "")
            if ' ' in asset:
                os.system(f'mv {root_dir}/\'{asset}\' {root_dir}/{replaced_name}')
            else:
                os.system(f'mv {root_dir}/{asset} {root_dir}/{replaced_name}')   

    valid_files = os.listdir(root_dir)
    valid_files.sort()
    file_names_parent = [os.path.join(root_dir, f) for f in valid_files]

    file_names = [f for f in file_names_parent if '.glb' in f] 
    file_goes = [f.replace('glb', 'usd').replace('asssets/objects', 'asssets/usds') for f in file_names]
    
    if args_cli.mass is not None:
        mass_props = schemas_cfg.MassPropertiesCfg(mass=args_cli.mass)
        rigid_props = schemas_cfg.RigidBodyPropertiesCfg()
    else:
        mass_props = None
        rigid_props = None
    mass_props = schemas_cfg.MassPropertiesCfg(mass=args_cli.mass)
    rigid_props = schemas_cfg.RigidBodyPropertiesCfg(rigid_body_enabled=True,kinematic_enabled=True)
    # Collision properties
    collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=args_cli.collision_approximation != "none")
    
    mesh_converters_cfgs = []
    for mesh_path, mesh_go in zip(file_names, file_goes):
        # Create Mesh converter config
        mesh_converter_cfg = MeshConverterCfg(
            mass_props=mass_props,
            rigid_props=rigid_props,
            collision_props=collision_props,
            asset_path=mesh_path,
            force_usd_conversion=True,
            usd_dir=os.path.dirname(mesh_go),
            usd_file_name=os.path.basename(mesh_go),
            make_instanceable=args_cli.make_instanceable,
            collision_approximation=args_cli.collision_approximation,
        )
        mesh_converters_cfgs.append(mesh_converter_cfg)

    # Print info
    for mesh_converter_cfg in tqdm.tqdm(mesh_converters_cfgs):
        print('|' + "-" * 100 + '|')
        print('|' + "-" * 100 + '|')
        print("Mesh importer config:")
        print_dict(mesh_converter_cfg.to_dict(), nesting=0)
        print('|' + "-" * 100 + '|')
        print('|' + "-" * 100 + '|')

        # Create Mesh converter and import the file
        mesh_converter = MeshConverter(mesh_converter_cfg)
        
        # print output
        print("Mesh importer output:")
        print(f"Generated USD file: {mesh_converter.usd_path}")
        print('|' + "-" * 100 + '|')
        print('|' + "-" * 100 + '|')


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
