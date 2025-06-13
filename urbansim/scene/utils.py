# Copyright (c) 2022-2025, The UrbanSim Project Developers.
# Author: Honglin He
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import numpy as np
from collections.abc import Sequence
from typing import Any
import os

import carb
import omni.usd
from isaacsim.core.cloner import GridCloner
from isaacsim.core.prims import XFormPrim
from pxr import PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObject,
    DeformableObjectCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.scene.interactive_scene import InteractiveScene
from isaaclab.scene.interactive_scene_cfg import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, SensorBase, SensorBaseCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

import math


class PGDrivableAreaProperty:
    """
    Defining some properties for creating PGMap
    """
    # road network property
    ID = None  # each block must have a unique ID
    SOCKET_NUM = None

    # visualization size property
    LANE_SEGMENT_LENGTH = 4
    SI_SEGMENT_LENGTH = 1
    STRIPE_LENGTH = 1.5
    LANE_LINE_WIDTH = 0.15
    LANE_LINE_THICKNESS = 0.016
    
    # Narrow sidewalk
    scale = 2.
    NARROW_SIDEWALK_NEAR_ROAD_MIN_WIDTH = 0.6 * scale
    NARROW_SIDEWALK_NEAR_ROAD_MAX_WIDTH = 0.8 * scale
    NARROW_SIDEWALK_MAIN_MIN_WIDTH = 2.4 * scale
    NARROW_SIDEWALK_MAIN_MAX_WIDTH = 2.8 * scale
    
    # Narrow sidewalk with trees
    NARROWT_SIDEWALK_NEAR_ROAD_MIN_WIDTH = 1.5 * scale
    NARROWT_SIDEWALK_NEAR_ROAD_MAX_WIDTH = 1.8 * scale
    NARROWT_SIDEWALK_MAIN_MIN_WIDTH = 2.4 * scale
    NARROWT_SIDEWALK_MAIN_MAX_WIDTH = 2.8 * scale
    
    # Ribbon Sidewalk
    RIBBON_SIDEWALK_NEAR_ROAD_MIN_WIDTH = 1.5 * scale
    RIBBON_SIDEWALK_NEAR_ROAD_MAX_WIDTH = 1.8 * scale
    RIBBON_SIDEWALK_MAIN_MIN_WIDTH = 2.0 * scale
    RIBBON_SIDEWALK_MAIN_MAX_WIDTH = 2.4 * scale
    RIBBON_SIDEWALK_FAR_MIN_WIDTH = 0.5 * scale
    RIBBON_SIDEWALK_FAR_MAX_WIDTH = 0.8 * scale
    
    # Neighborhood Main Street 1
    NEIGHBORHOOD_SIDEWALK_NEAR_ROAD_MIN_WIDTH = 1.0 * scale
    NEIGHBORHOOD_SIDEWALK_NEAR_ROAD_MAX_WIDTH = 1.2 * scale
    NEIGHBORHOOD_SIDEWALK_BUFFER_NEAR_MIN_WIDTH = 2.1 * scale
    NEIGHBORHOOD_SIDEWALK_BUFFER_NEAR_MAX_WIDTH = 2.4 * scale
    NEIGHBORHOOD_SIDEWALK_MAIN_MIN_WIDTH = 2.4 * scale
    NEIGHBORHOOD_SIDEWALK_MAIN_MAX_WIDTH = 2.8 * scale
    
    # Neighborhood Main Street 2
    NEIGHBORHOOD2_SIDEWALK_NEAR_ROAD_MIN_WIDTH = 1.5 * scale
    NEIGHBORHOOD2_SIDEWALK_NEAR_ROAD_MAX_WIDTH = 1.8 * scale
    NEIGHBORHOOD2_SIDEWALK_MAIN_MIN_WIDTH = 3.0 * scale
    NEIGHBORHOOD2_SIDEWALK_MAIN_MAX_WIDTH = 3.3 * scale
    NEIGHBORHOOD2_SIDEWALK_BUFFER_FAR_MIN_WIDTH = 1.5 * scale
    NEIGHBORHOOD2_SIDEWALK_BUFFER_FAR_MAX_WIDTH = 1.8 * scale
    
    # Medium Commercial 
    MediumCommercial_SIDEWALK_NEAR_ROAD_MIN_WIDTH = 1.5 * scale
    MediumCommercial_SIDEWALK_NEAR_ROAD_MAX_WIDTH = 1.8 * scale
    MediumCommercial_SIDEWALK_MAIN_MIN_WIDTH = 3.0 * scale
    MediumCommercial_SIDEWALK_MAIN_MAX_WIDTH = 3.3 * scale
    MediumCommercial_SIDEWALK_FAR_MIN_WIDTH = 3.0 * scale
    MediumCommercial_SIDEWALK_FAR_MAX_WIDTH = 3.3 * scale
    
    # Wide Commercial 
    WideCommercial_SIDEWALK_NEAR_ROAD_MIN_WIDTH = 1.5 * scale
    WideCommercial_SIDEWALK_NEAR_ROAD_MAX_WIDTH = 1.8 * scale
    WideCommercial_SIDEWALK_MAIN_MIN_WIDTH = 3.0 * scale
    WideCommercial_SIDEWALK_MAIN_MAX_WIDTH = 3.3 * scale
    WideCommercial_SIDEWALK_MAIN_BUFFER_MIN_WIDTH = 0.3 * scale
    WideCommercial_SIDEWALK_MAIN_BUFFER_MAX_WIDTH = 0.5 * scale
    WideCommercial_SIDEWALK_FAR_MIN_WIDTH = 3.0 * scale
    WideCommercial_SIDEWALK_FAR_MAX_WIDTH = 3.3 * scale

    SIDEWALK_THICKNESS = 0.3
    SIDEWALK_LENGTH = 3
    SIDEWALK_SEG_LENGTH = 1
    SIDEWALK_WIDTH = 5
    CROSSWALK_WIDTH = 5
    CROSSWALK_LENGTH = 1
    SIDEWALK_LINE_DIST = 0.6
    HOUSE_WIDTH = 25.
        
    SIDEWALK_NEAR_ROAD_WIDTH = 4
    SIDEWALK_FARFROM_ROAD_WIDTH = 4
    OFF_SIDEWALK_VALID_WIDTH = 6

    GUARDRAIL_HEIGHT = 4.0

    # visualization color property
    LAND_COLOR = (0.4, 0.4, 0.4, 1)
    NAVI_COLOR = (0.709, 0.09, 0, 1)

    # for detection
    LANE_LINE_GHOST_HEIGHT = 1.0

    # for creating complex block, for example Intersection and roundabout consist of 4 part, which contain several road
    PART_IDX = 0
    ROAD_IDX = 0
    DASH = "_"

    #  when set to True, Vehicles will not generate on this block
    PROHIBIT_TRAFFIC_GENERATION = False

class UrbanType:
    """
    Following waymo style, this class defines a set of strings used to denote different types of objects.
    Those types are used within MetaUrban and might mismatch to the strings used in other dataset.

    NOTE: when add new keys, make sure class method works well for them
    """
    TEST_OBJECT = "TestObject"
    # ===== Lane, Road =====
    LANE_SURFACE_STREET = "LANE_SURFACE_STREET"
    # Unlike a set of lanes separated by broken/solid line, this includes intersection and some unstructured roads.
    LANE_SURFACE_UNSTRUCTURE = "LANE_SURFACE_UNSTRUCTURE"
    # use them as less frequent as possible, it is for waymo compatibility
    LANE_UNKNOWN = "LANE_UNKNOWN"
    LANE_FREEWAY = "LANE_FREEWAY"
    LANE_BIKE_LANE = "LANE_BIKE_LANE"

    # ===== Lane Line =====
    LINE_UNKNOWN = "UNKNOWN_LINE"
    LINE_BROKEN_SINGLE_WHITE = "ROAD_LINE_BROKEN_SINGLE_WHITE"
    LINE_SOLID_SINGLE_WHITE = "ROAD_LINE_SOLID_SINGLE_WHITE"
    LINE_SOLID_DOUBLE_WHITE = "ROAD_LINE_SOLID_DOUBLE_WHITE"
    LINE_BROKEN_SINGLE_YELLOW = "ROAD_LINE_BROKEN_SINGLE_YELLOW"
    LINE_BROKEN_DOUBLE_YELLOW = "ROAD_LINE_BROKEN_DOUBLE_YELLOW"
    LINE_SOLID_SINGLE_YELLOW = "ROAD_LINE_SOLID_SINGLE_YELLOW"
    LINE_SOLID_DOUBLE_YELLOW = "ROAD_LINE_SOLID_DOUBLE_YELLOW"
    LINE_PASSING_DOUBLE_YELLOW = "ROAD_LINE_PASSING_DOUBLE_YELLOW"

    # ===== Edge/Boundary/SideWalk/Region =====
    BOUNDARY_UNKNOWN = "UNKNOWN"  # line
    BOUNDARY_LINE = "ROAD_EDGE_BOUNDARY"  # line
    BOUNDARY_MEDIAN = "ROAD_EDGE_MEDIAN"  # line
    BOUNDARY_SIDEWALK = "ROAD_EDGE_SIDEWALK"  # polygon
    STOP_SIGN = "STOP_SIGN"
    CROSSWALK = "CROSSWALK"
    SPEED_BUMP = "SPEED_BUMP"
    DRIVEWAY = "DRIVEWAY"
    GUARDRAIL = "GUARDRAIL"  # A thickened sidewalk that doesn't allow object to penetrate.

    # ===== Traffic Light =====
    LANE_STATE_UNKNOWN = "LANE_STATE_UNKNOWN"
    LANE_STATE_ARROW_STOP = "LANE_STATE_ARROW_STOP"
    LANE_STATE_ARROW_CAUTION = "LANE_STATE_ARROW_CAUTION"
    LANE_STATE_ARROW_GO = "LANE_STATE_ARROW_GO"
    LANE_STATE_STOP = "LANE_STATE_STOP"
    LANE_STATE_CAUTION = "LANE_STATE_CAUTION"
    LANE_STATE_GO = "LANE_STATE_GO"
    LANE_STATE_FLASHING_STOP = "LANE_STATE_FLASHING_STOP"
    LANE_STATE_FLASHING_CAUTION = "LANE_STATE_FLASHING_CAUTION"

    # the light states above will be converted to the following 4 types
    LIGHT_GREEN = "TRAFFIC_LIGHT_GREEN"
    LIGHT_RED = "TRAFFIC_LIGHT_RED"
    LIGHT_YELLOW = "TRAFFIC_LIGHT_YELLOW"
    LIGHT_UNKNOWN = "TRAFFIC_LIGHT_UNKNOWN"

    # ===== Agent type =====
    UNSET = "UNSET"
    VEHICLE = "VEHICLE"
    PEDESTRIAN = "PEDESTRIAN"
    CYCLIST = "CYCLIST"
    OTHER = "OTHER"

    # ===== Object type =====
    TRAFFIC_LIGHT = "TRAFFIC_LIGHT"
    TRAFFIC_BARRIER = "TRAFFIC_BARRIER"
    TRAFFIC_CONE = "TRAFFIC_CONE"
    TRAFFIC_OBJECT = "TRAFFIC_OBJECT"
    GROUND = "GROUND"
    INVISIBLE_WALL = "INVISIBLE_WALL"
    BUILDING = "BUILDING"

    @classmethod
    def is_traffic_object(cls, type):
        return type in [cls.TRAFFIC_CONE, cls.TRAFFIC_BARRIER, cls.TRAFFIC_OBJECT]

    @classmethod
    def has_type(cls, type_string: str):
        return type_string in cls.__dict__

    @classmethod
    def from_waymo(cls, waymo_type_string: str):
        assert cls.__dict__[waymo_type_string]
        return waymo_type_string

    @classmethod
    def is_lane(cls, type):
        return type in [
            cls.LANE_SURFACE_STREET, cls.LANE_SURFACE_UNSTRUCTURE, cls.LANE_UNKNOWN, cls.LANE_BIKE_LANE,
            cls.LANE_FREEWAY
        ]

    @classmethod
    def is_road_line(cls, line):
        """
        This function relates to is_road_edge. We will have different processing when treating a line that
        is in the boundary or not.
        """
        return line in [
            cls.LINE_UNKNOWN, cls.LINE_BROKEN_SINGLE_WHITE, cls.LINE_SOLID_SINGLE_WHITE, cls.LINE_SOLID_DOUBLE_WHITE,
            cls.LINE_BROKEN_SINGLE_YELLOW, cls.LINE_BROKEN_DOUBLE_YELLOW, cls.LINE_SOLID_SINGLE_YELLOW,
            cls.LINE_SOLID_DOUBLE_YELLOW, cls.LINE_PASSING_DOUBLE_YELLOW
        ]

    @classmethod
    def is_yellow_line(cls, line):
        return line in [
            cls.LINE_SOLID_DOUBLE_YELLOW, cls.LINE_PASSING_DOUBLE_YELLOW, cls.LINE_SOLID_SINGLE_YELLOW,
            cls.LINE_BROKEN_DOUBLE_YELLOW, cls.LINE_BROKEN_SINGLE_YELLOW
        ]

    @classmethod
    def is_white_line(cls, line):
        return UrbanType.is_road_line(line) and not UrbanType.is_yellow_line(line)

    @classmethod
    def is_broken_line(cls, line):
        return line in [cls.LINE_BROKEN_DOUBLE_YELLOW, cls.LINE_BROKEN_SINGLE_YELLOW, cls.LINE_BROKEN_SINGLE_WHITE]

    @classmethod
    def is_solid_line(cls, line):
        return line in [
            cls.LINE_SOLID_DOUBLE_WHITE, cls.LINE_SOLID_DOUBLE_YELLOW, cls.LINE_SOLID_SINGLE_YELLOW,
            cls.LINE_SOLID_SINGLE_WHITE
        ]

    @classmethod
    def is_road_boundary_line(cls, edge):
        """
        This function relates to is_road_line.
        """
        return edge in [cls.BOUNDARY_UNKNOWN, cls.BOUNDARY_LINE, cls.BOUNDARY_MEDIAN]

    @classmethod
    def is_sidewalk(cls, edge):
        return edge == cls.BOUNDARY_SIDEWALK

    @classmethod
    def is_stop_sign(cls, type):
        return type == UrbanType.STOP_SIGN

    @classmethod
    def is_speed_bump(cls, type):
        return type == UrbanType.SPEED_BUMP

    @classmethod
    def is_driveway(cls, type):
        return type == UrbanType.DRIVEWAY

    @classmethod
    def is_crosswalk(cls, type):
        return type == UrbanType.CROSSWALK

    @classmethod
    def is_vehicle(cls, type):
        return type == cls.VEHICLE

    @classmethod
    def is_pedestrian(cls, type):
        return type == cls.PEDESTRIAN

    @classmethod
    def is_cyclist(cls, type):
        return type == cls.CYCLIST

    @classmethod
    def is_participant(cls, type):
        return type in (cls.CYCLIST, cls.PEDESTRIAN, cls.VEHICLE, cls.UNSET, cls.OTHER)

    @classmethod
    def is_traffic_light_in_yellow(cls, light):
        return cls.simplify_light_status(light) == cls.LIGHT_YELLOW

    @classmethod
    def is_traffic_light_in_green(cls, light):
        return cls.simplify_light_status(light) == cls.LIGHT_GREEN

    @classmethod
    def is_traffic_light_in_red(cls, light):
        return cls.simplify_light_status(light) == cls.LIGHT_RED

    @classmethod
    def is_traffic_light_unknown(cls, light):
        return cls.simplify_light_status(light) == cls.LIGHT_UNKNOWN

    @classmethod
    def parse_light_status(cls, status: str, simplifying=True):
        """
        Parse light status from ENUM to STR
        """
        # if data_source == "waymo":
        #     status = cls.LIGHT_ENUM_TO_STR[status]
        if simplifying:
            return cls.simplify_light_status(status)
        else:
            return status

    @classmethod
    def simplify_light_status(cls, status: str):
        """
        Convert status to red/yellow/green/unknown
        """
        if status in [cls.LANE_STATE_UNKNOWN, cls.LANE_STATE_FLASHING_STOP, cls.LIGHT_UNKNOWN, None]:
            return cls.LIGHT_UNKNOWN
        elif status in [cls.LANE_STATE_ARROW_STOP, cls.LANE_STATE_STOP, cls.LIGHT_RED]:
            return cls.LIGHT_RED
        elif status in [cls.LANE_STATE_ARROW_CAUTION, cls.LANE_STATE_CAUTION, cls.LANE_STATE_FLASHING_CAUTION,
                        cls.LIGHT_YELLOW]:
            return cls.LIGHT_YELLOW
        elif status in [cls.LANE_STATE_ARROW_GO, cls.LANE_STATE_GO, cls.LIGHT_GREEN]:
            return cls.LIGHT_GREEN
        else:
            return cls.LIGHT_UNKNOWN

    def __init__(self, type=None):
        # TODO extend this base class to all objects! It is only affect lane so far.
        # TODO Or people can only know the type with isinstance()
        self.urban_type = UrbanType.UNSET
        if type is not None:
            self.set_urban_type(type)

    def set_urban_type(self, type):
        if type in UrbanType.__dict__.values():
            # Do something if type matches one of the class variables
            self.urban_type = type
        else:
            raise ValueError(f"'{type}' is not a valid UrbanType.")

def _random_points_new(map_mask, num, min_dis=5, generated_position=None):
        import matplotlib.pyplot as plt
        from scipy.signal import convolve2d
        import random
        from skimage import measure
        h, _ = map_mask.shape
        import metaurban.policy.orca_planner_utils as orca_planner_utils
        mylist, h, w = orca_planner_utils.mask_to_2d_list(map_mask)
        contours = measure.find_contours(mylist, 0.5, positive_orientation='high')
        flipped_contours = []
        for contour in contours:
            contour = orca_planner_utils.find_tuning_point(contour, h)
            flipped_contours.append(contour)
        int_points = []
        for p in flipped_contours:
            for m in p:
                int_points.append((int(m[1]), int(m[0])))
        def find_walkable_area(map_mask):
            kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
            conv_result= convolve2d(map_mask/255, kernel, mode='same')
            ct_pts = np.where(conv_result==8) #8, 24
            ct_pts = list(zip(ct_pts[1], ct_pts[0]))
            ct_pts = [c for c in ct_pts if c not in int_points]
            return ct_pts
        selected_pts = []
        walkable_pts = find_walkable_area(map_mask)
        if generated_position is not None:
            dis_to_start = np.linalg.norm(np.array([(x[0], h - 1 - x[1]) for x in walkable_pts]) - generated_position, axis=1)
            walkable_pts = np.array(walkable_pts)[np.argsort(dis_to_start)[::-1]][:int(len(walkable_pts) / 10)].tolist()
        random.shuffle(walkable_pts)
        if len(walkable_pts) < num: raise ValueError(" Walkable points are less than spawn number! ")
        try_time = 0
        while len(selected_pts) < num:
            if try_time > 10000: raise ValueError("Try too many time to get valid humanoid points!")
            cur_pt = random.choice(walkable_pts)
            if all(math.dist(cur_pt, selected_pt) >= min_dis for selected_pt in selected_pts): 
                selected_pts.append(cur_pt)
            try_time+=1
        selected_pts = [(x[0], h - 1 - x[1]) for x in selected_pts]
        return selected_pts

def random_start_and_end_points(map_mask, mask_translate, num, starts_init=None):
    distance = 1.0
    import math
    iteration = 0
    while distance <  50.0:
        if starts_init is None:
            starts = _random_points_new(map_mask, num)
        else:
            starts = starts_init
        goals = _random_points_new(map_mask, num, generated_position=starts[0])
        goal_pos = goals[0] - mask_translate
        start_pos = starts[0] - mask_translate
        distance = math.sqrt((goal_pos[0] - start_pos[0]) ** 2 + (goal_pos[1] - start_pos[1]) ** 2)
        iteration += 1
        if iteration > 100:
            break
    return starts, goals

def construct_continuous_line_polygon(lane, lateral, line_color, line_type):
    segment_num = int(lane.length / PGDrivableAreaProperty.LANE_SEGMENT_LENGTH)
    polygon = []
    if segment_num == 0:
        # start = lane.position(0, lateral)
        # end = lane.position(lane.length, lateral)
        segment_polygon = []
        for i in range(int(lane.length)):
            segment_polygon.append([lane.position(i, lateral - PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
        if lane.length > int(lane.length):
            segment_polygon.append([lane.position(lane.length, lateral - PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
            segment_polygon.append([lane.position(lane.length, lateral + PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
        for i in range(int(lane.length)):
            segment_polygon.append([lane.position(int(lane.length) - i, lateral + PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
        polygon.append(np.array(segment_polygon).reshape(-1, 2))
        #node_path_list = construct_lane_line_segment_polygon(start, end, line_color, line_type)
    for segment in range(segment_num):
        if segment == segment_num - 1:
            segment_polygon = []
            for i in range(int(lane.length) - segment * PGDrivableAreaProperty.LANE_SEGMENT_LENGTH + 1):
                segment_polygon.append([lane.position(i + segment * PGDrivableAreaProperty.LANE_SEGMENT_LENGTH, lateral - PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
            if lane.length > int(lane.length):
                segment_polygon.append([lane.position(lane.length, lateral - PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
                segment_polygon.append([lane.position(lane.length, lateral + PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
            for i in range(int(lane.length) - segment * PGDrivableAreaProperty.LANE_SEGMENT_LENGTH + 1):
                segment_polygon.append([lane.position(int(lane.length) - i, lateral + PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
            polygon.append(np.array(segment_polygon).reshape(-1, 2))
        else:
            segment_polygon = []
            for i in range(int(PGDrivableAreaProperty.LANE_SEGMENT_LENGTH) + 1):
                segment_polygon.append([lane.position(i + segment * PGDrivableAreaProperty.LANE_SEGMENT_LENGTH, lateral - PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
            for i in range(int(PGDrivableAreaProperty.LANE_SEGMENT_LENGTH) + 1):
                segment_polygon.append([lane.position((segment + 1) * PGDrivableAreaProperty.LANE_SEGMENT_LENGTH - i, lateral + PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
            polygon.append(np.array(segment_polygon).reshape(-1, 2))
        
    return polygon

def construct_continuous_polygon(lane, start_lat, end_lat):
    segment_num = int(lane.length / PGDrivableAreaProperty.SI_SEGMENT_LENGTH)
    polygon = []
    if segment_num == 0:
        # start = lane.position(0, lateral)
        # end = lane.position(lane.length, lateral)
        segment_polygon = []
        for i in range(int(lane.length)):
            segment_polygon.append([lane.position(i, start_lat)])
        if lane.length > int(lane.length):
            segment_polygon.append([lane.position(lane.length, start_lat)])
            segment_polygon.append([lane.position(lane.length, end_lat)])
        for i in range(int(lane.length)):
            segment_polygon.append([lane.position(int(lane.length) - i, end_lat)])
        polygon.append(np.array(segment_polygon).reshape(-1, 2))
        #node_path_list = construct_lane_line_segment_polygon(start, end, line_color, line_type)
    for segment in range(segment_num):
        if segment == segment_num - 1:
            segment_polygon = []
            for i in range(int(lane.length) - segment * PGDrivableAreaProperty.SI_SEGMENT_LENGTH + 1):
                segment_polygon.append([lane.position(i + segment * PGDrivableAreaProperty.SI_SEGMENT_LENGTH, start_lat)])
            if lane.length > int(lane.length):
                segment_polygon.append([lane.position(lane.length, start_lat)])
                segment_polygon.append([lane.position(lane.length, end_lat)])
            for i in range(int(lane.length) - segment * PGDrivableAreaProperty.SI_SEGMENT_LENGTH + 1):
                segment_polygon.append([lane.position(int(lane.length) - i, end_lat)])
            polygon.append(np.array(segment_polygon).reshape(-1, 2))
        else:
            segment_polygon = []
            for i in range(int(PGDrivableAreaProperty.SI_SEGMENT_LENGTH) + 1):
                segment_polygon.append([lane.position(i + segment * PGDrivableAreaProperty.SI_SEGMENT_LENGTH, start_lat)])
            for i in range(int(PGDrivableAreaProperty.SI_SEGMENT_LENGTH) + 1):
                segment_polygon.append([lane.position((segment + 1) * PGDrivableAreaProperty.SI_SEGMENT_LENGTH - i, end_lat)])
            polygon.append(np.array(segment_polygon).reshape(-1, 2))
        
    return polygon

def construct_broken_line_polygon(lane, lateral, line_color, line_type):
    assert UrbanType.is_broken_line(line_type)
    points = lane.get_polyline(2, lateral)
    polygon = []
    for index in range(0, len(points) - 1, 2):
        segment_polygon = []
        if index + 1 < len(points):
              segment_polygon.append([lane.position(2 * index, lateral - PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
              segment_polygon.append([lane.position(2 * (index + 1), lateral - PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
              segment_polygon.append([lane.position(2 * (index + 1), lateral + PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
              segment_polygon.append([lane.position(2 * index, lateral + PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
              polygon.append(np.array(segment_polygon).reshape(-1, 2))
    return polygon
import trimesh
from shapely.geometry import Polygon
def generate_random_road(area_size=(100, 100), level=5):
    start = (0.25, 0.25)
    end = (np.random.uniform(area_size[0], area_size[0]), np.random.uniform(area_size[1], area_size[1]))
    
    mid_points_x = np.linspace(start[0], end[0], level + 2)
    mid_points_y = np.linspace(start[1], end[1], level + 2)
    mid_points_y[1:-1] = np.random.uniform(0, area_size[1], size=level)

    x_points = mid_points_x
    y_points = mid_points_y
    
    # from scipy.interpolate import make_interp_spline
    # x_new = np.linspace(start[0], end[0], 300)
    # spl = make_interp_spline(x_points, y_points, k=1)
    # y_smooth = spl(x_new)
    
    return x_points, y_points

def get_road_trimesh(x, y, area_size, min_road_width=15, max_road_width=18, height=0.05, boundary=[-1e10, 1e10]):
    road_width = np.random.uniform(min_road_width, max_road_width)
    
    upper_boundary = y + road_width / 2
    lower_boundary = y - road_width / 2
    upper_boundary = upper_boundary.clip(boundary[0], boundary[1])
    lower_boundary = lower_boundary.clip(boundary[0], boundary[1])
    polyline_points = np.concatenate([np.column_stack((x, upper_boundary)), np.column_stack((x[::-1], lower_boundary[::-1]))])
    
    polygon = Polygon(polyline_points)
    
    mesh = trimesh.creation.extrude_polygon(polygon, height=height)
    
    return mesh, [upper_boundary, lower_boundary], polyline_points

UV_SCLAE = 20
def uv_texturing(mesh, scale=UV_SCLAE):
    uvs = []
    for vertex in mesh.vertices:
        uv = [vertex[0], vertex[1]]
        uvs.append(uv)
    uvs = np.array(uvs)
    uvs = (uvs - np.min(uvs)) / (np.max(uvs) - np.min(uvs))
    uvs *= scale
    mesh.visual.uvs = uvs
    
    return mesh

# UrbanVerse Utils
import os
import json
import pickle
import math
import copy
import matplotlib
import matplotlib.pyplot as plt
import gzip
import random
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from collections.abc import Iterable
from collections import defaultdict, Counter

def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def to_tensor(numpy_array, device=None):
    if isinstance(numpy_array, torch.Tensor):
        return numpy_array
    if device is None:
        return torch.from_numpy(numpy_array)
    else:
        return torch.from_numpy(numpy_array).to(device)


class DetectionList(list):
    def get_values(self, key, idx: int = None):
        if idx is None:
            return [detection[key] for detection in self]
        else:
            return [detection[key][idx] for detection in self]

    def get_stacked_values_torch(self, key, idx: int = None):
        values = []
        for detection in self:
            v = detection[key]
            if idx is not None:
                v = v[idx]
            if isinstance(v, o3d.geometry.OrientedBoundingBox) or \
                    isinstance(v, o3d.geometry.AxisAlignedBoundingBox):
                v = np.asarray(v.get_box_points())
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            values.append(v)
        return torch.stack(values, dim=0)

    def get_stacked_values_numpy(self, key, idx: int = None):
        values = self.get_stacked_values_torch(key, idx)
        return to_numpy(values)

    def __add__(self, other):
        new_list = copy.deepcopy(self)
        new_list.extend(other)
        return new_list

    def __iadd__(self, other):
        self.extend(other)
        return self

    def slice_by_indices(self, index: Iterable[int]):
        '''
        Return a sublist of the current list by indexing
        '''
        new_self = type(self)()
        for i in index:
            new_self.append(self[i])
        return new_self

    def slice_by_mask(self, mask: Iterable[bool]):
        '''
        Return a sublist of the current list by masking
        '''
        new_self = type(self)()
        for i, m in enumerate(mask):
            if m:
                new_self.append(self[i])
        return new_self

    def get_most_common_class(self) -> list[int]:
        classes = []
        for d in self:
            values, counts = np.unique(np.asarray(d['class_id']), return_counts=True)
            most_common_class = values[np.argmax(counts)]
            classes.append(most_common_class)
        return classes

    def color_by_most_common_classes(self, obj_classes, color_bbox: bool = True):
        '''
        Color the point cloud of each detection by the most common class
        '''
        classes = self.get_most_common_class()
        for d, c in zip(self, classes):
            # color = obj_classes[str(c)]
            color = obj_classes.get_class_color(int(c))
            d['pcd'].paint_uniform_color(color)
            if color_bbox:
                d['bbox'].color = color

    def color_by_instance(self):
        if len(self) == 0:
            # Do nothing
            return

        if "inst_color" in self[0]:
            for d in self:
                d['pcd'].paint_uniform_color(d['inst_color'])
                d['bbox'].color = d['inst_color']
        else:
            cmap = matplotlib.colormaps.get_cmap("turbo")
            instance_colors = cmap(np.linspace(0, 1, len(self)))
            instance_colors = instance_colors[:, :3]
            for i in range(len(self)):
                self[i]['pcd'].paint_uniform_color(instance_colors[i])
                self[i]['bbox'].color = instance_colors[i]


class MapObjectList(DetectionList):
    def compute_similarities(self, new_clip_ft):
        '''
        The input feature should be of shape (D, ), a one-row vector
        This is mostly for backward compatibility
        '''
        # if it is a numpy array, make it a tensor
        new_clip_ft = to_tensor(new_clip_ft)

        # assuming cosine similarity for features
        clip_fts = self.get_stacked_values_torch('clip_ft')

        similarities = F.cosine_similarity(new_clip_ft.unsqueeze(0), clip_fts)
        # return similarities.squeeze()
        return similarities

    def to_serializable(self):
        s_obj_list = []

        for obj in self:
            s_obj_dict = copy.deepcopy(obj)

            # print(s_obj_dict["bound_cousins"])

            if 'aligned_bbox_instance' in s_obj_dict:    del s_obj_dict['aligned_bbox_instance']
            if 'aligned_bbox_2d_instance' in s_obj_dict: del s_obj_dict['aligned_bbox_2d_instance']

            s_obj_dict['pcd_np'] = np.asarray(s_obj_dict['pcd'].points)
            s_obj_dict['pcd_color_np'] = np.asarray(s_obj_dict['pcd'].colors)
            s_obj_dict['bbox_np'] = np.asarray(s_obj_dict['bbox'].get_box_points())

            del s_obj_dict['pcd']
            del s_obj_dict['bbox']

            if 'aligned_pcd' in s_obj_dict:
                s_obj_dict['aligned_pcd_np'] = np.asarray(s_obj_dict['aligned_pcd'].points)
                s_obj_dict['aligned_pcd_color_np'] = np.asarray(s_obj_dict['aligned_pcd'].colors)
                del s_obj_dict['aligned_pcd']

            if 'aligned_bbox' in s_obj_dict:
                s_obj_dict['aligned_bbox_np'] = np.asarray(s_obj_dict['aligned_bbox'].get_box_points())
                del s_obj_dict['aligned_bbox']

            if 'aligned_bbox_2d' in s_obj_dict:
                s_obj_dict['aligned_bbox_2d_np'] = np.asarray(s_obj_dict['aligned_bbox_2d'].vertices)
                del s_obj_dict['aligned_bbox_2d']

            if 'clip_ft' in s_obj_dict: s_obj_dict['clip_ft'] = to_numpy(s_obj_dict['clip_ft'])

            if 'text_ft' in s_obj_dict: s_obj_dict['text_ft'] = to_numpy(s_obj_dict['text_ft'])

            if 'dino_ft' in s_obj_dict: s_obj_dict['dino_ft'] = to_numpy(s_obj_dict['dino_ft'])

            s_obj_list.append(s_obj_dict)

        return s_obj_list

    def load_serializable(self, s_obj_list):
        assert len(self) == 0, 'MapObjectList should be empty when loading'
        for s_obj_dict in s_obj_list:
            new_obj = copy.deepcopy(s_obj_dict)

            new_obj['pcd'] = o3d.geometry.PointCloud()
            new_obj['pcd'].points = o3d.utility.Vector3dVector(new_obj['pcd_np'])
            new_obj['pcd'].colors = o3d.utility.Vector3dVector(new_obj['pcd_color_np'])

            new_obj['bbox'] = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(new_obj['bbox_np']))
            new_obj['bbox'].color = new_obj['pcd_color_np'][0]

            # del new_obj['pcd_np']
            # del new_obj['bbox_np']
            # del new_obj['pcd_color_np']

            if 'aligned_pcd_np' in new_obj:
                new_obj['aligned_pcd'] = o3d.geometry.PointCloud()
                new_obj['aligned_pcd'].points = o3d.utility.Vector3dVector(new_obj['aligned_pcd_np'])
                new_obj['aligned_pcd'].colors = o3d.utility.Vector3dVector(new_obj['aligned_pcd_color_np'])
                # del new_obj['aligned_pcd_np']

            if 'aligned_bbox_np' in new_obj:
                new_obj['aligned_bbox'] = o3d.geometry.OrientedBoundingBox.create_from_points(
                    o3d.utility.Vector3dVector(new_obj['aligned_bbox_np']))
                new_obj['aligned_bbox'].color = new_obj['aligned_pcd_color_np'][0]
                # del new_obj['aligned_bbox_np']

            if 'aligned_bbox_2d_np' in new_obj:
                new_obj['aligned_bbox_2d'] = o3d.geometry.TriangleMesh()
                new_obj['aligned_bbox_2d'].vertices = o3d.utility.Vector3dVector(new_obj['aligned_bbox_2d_np'])
                triangles = [[0, i, i + 1] for i in range(1, len(new_obj['aligned_bbox_2d_np']) - 1)]
                new_obj['aligned_bbox_2d'].triangles = o3d.utility.Vector3iVector(triangles)
                new_obj['aligned_bbox_2d'].paint_uniform_color(new_obj['aligned_pcd_color_np'][0])
                # del new_obj['aligned_bbox_2d_np']
                # del new_obj['aligned_pcd_color_np']

            if 'clip_ft' in new_obj:
                new_obj['clip_ft'] = to_tensor(new_obj['clip_ft'])

            if 'text_ft' in new_obj:
                new_obj['text_ft'] = to_tensor(new_obj['text_ft'])

            if 'dino_ft' in new_obj:
                new_obj['dino_ft'] = to_tensor(new_obj['dino_ft'])

            self.append(new_obj)


class SceneEntity(object):
    def __init__(
            self
    ):
        self.cfg_info = {}
        self.object_list = MapObjectList()

        self.class_names = []
        self.class_colors = {}
        self.instance_colors = {}

        self.num_classes = -1
        self.num_instances = -1

        self.primary_ground_class_names = []
        self.secondary_ground_class_names = []

    def populate(
            self,
            cfg_info: dict,
            object_list: MapObjectList,
            class_names: list,
            class_colors: dict,
            instance_colors: dict,
            primary_ground_class_names: list,
            secondary_ground_class_names: list,
    ):
        self.cfg_info = cfg_info
        self.object_list = object_list

        self.class_names = class_names
        self.class_colors = class_colors
        self.instance_colors = instance_colors

        self.primary_ground_class_names = primary_ground_class_names
        self.secondary_ground_class_names = secondary_ground_class_names

        self.num_classes = len(class_names)
        self.num_instances = len(object_list)

    def dump_to_serializable(
            self,
            file_path: str  # xxx.pkl.gz
    ):
        results = defaultdict()

        results['cfg_info'] = self.cfg_info
        results['objects'] = self.object_list.to_serializable()
        results['class_names'] = self.class_names
        results['class_colors'] = self.class_colors
        results['instance_colors'] = self.instance_colors
        results['primary_ground_class_names'] = self.primary_ground_class_names
        results['secondary_ground_class_names'] = self.secondary_ground_class_names

        # Save the scene entity data
        with gzip.open(file_path, "wb") as fbj:
            pickle.dump(results, fbj)

        print(f"[SceneEntity]: dumped the scene entity from {file_path}")
        return True

    def load_from_serializable(
            self,
            file_path: str  # xxx.pkl.gz
    ):
        with gzip.open(file_path, "rb") as fbj:
            results = pickle.load(fbj)

        if not isinstance(results, dict):
            raise ValueError("Results should be a dictionary!")

        self.cfg_info = results['cfg_info']

        self.object_list.load_serializable(results["objects"])

        self.class_names = results['class_names']
        self.class_colors = results['class_colors']
        self.instance_colors = results['instance_colors']

        self.primary_ground_class_names = results['primary_ground_class_names']
        self.secondary_ground_class_names = results['secondary_ground_class_names']

        self.num_classes = len(results['class_names'])
        self.num_instances = len(self.object_list)

        print(f"[SceneEntity]: loaded the scene entity from {file_path}")
        return True

    def get_object_entities(
            self
    ) -> MapObjectList:
        return self.object_list

    def get_class_names(
            self
    ) -> list:
        return self.class_names

    def get_class_colors(
            self
    ) -> dict:
        return self.class_colors

    def get_instance_colors(
            self
    ) -> dict:
        return self.instance_colors

    def get_primary_ground_class_names(
            self
    ) -> list:
        return self.primary_ground_class_names

    def get_secondary_ground_class_names(
            self
    ) -> list:
        return self.secondary_ground_class_names

    def get_num_classes(
            self
    ) -> int:
        return self.num_classes

    def get_num_instances(
            self
    ) -> int:
        return self.num_instances

    def get_cfg(
            self
    ) -> dict:
        return self.cfg_info
