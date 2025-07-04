import copy
from metaurban.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metaurban.type import MetaUrbanType
from metaurban.constants import PGLineType, PGLineColor
from typing import List
import numpy as np
from panda3d.core import NodePath

from metaurban.component.algorithm.BIG import BigGenerateMethod, BIG
from metaurban.component.map.base_map import BaseMap
from metaurban.component.pgblock.first_block import FirstPGBlock
from metaurban.component.road_network.node_road_network import NodeRoadNetwork
from metaurban.constants import Decoration
from metaurban.engine.core.physics_world import PhysicsWorld
from metaurban.utils import Config


def parse_map_config(easy_map_config, new_map_config, default_config):
    # assert isinstance(new_map_config, Config)
    # assert isinstance(default_config, Config)

    # Return the user specified config if overwritten
    if not default_config["map_config"].is_identical(new_map_config):
        new_map_config = default_config["map_config"].copy(unchangeable=False).update(new_map_config)
        assert default_config["map"] == easy_map_config
        return new_map_config

    if isinstance(easy_map_config, int):
        new_map_config[BaseMap.GENERATE_TYPE] = BigGenerateMethod.BLOCK_NUM
    elif isinstance(easy_map_config, str):
        new_map_config[BaseMap.GENERATE_TYPE] = BigGenerateMethod.BLOCK_SEQUENCE
    else:
        raise ValueError(
            "Unkown easy map config: {} and original map config: {}".format(easy_map_config, new_map_config)
        )
    new_map_config[BaseMap.GENERATE_CONFIG] = easy_map_config
    return new_map_config


class MapGenerateMethod:
    BIG_BLOCK_NUM = BigGenerateMethod.BLOCK_NUM
    BIG_BLOCK_SEQUENCE = BigGenerateMethod.BLOCK_SEQUENCE
    BIG_SINGLE_BLOCK = BigGenerateMethod.SINGLE_BLOCK
    PG_MAP_FILE = "pg_map_file"


class PGMap(BaseMap):
    def _generate(self):
        """
        We can override this function to introduce other methods!
        """
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        generate_type = self._config[self.GENERATE_TYPE]
        self.sidewalk_type_all = ['Narrow Sidewalk', 
                                  'Narrow Sidewalk with Trees', 
                                  'Ribbon Sidewalk', 
                                  'Neighborhood 1', 
                                  'Neighborhood 2', 
                                  'Medium Commercial', 
                                  'Wide Commercial']
        if 'training' in self.engine.global_config:
            training_ = self.engine.global_config['training']
            use_all = False
        else:
            training_ = False
            use_all = True
        
        if not use_all:
            print('Current mode is not [use-all], training types and validation are different')
            self.sidewalk_type_all = self.sidewalk_type_all[:-1] if training_ else self.sidewalk_type_all[-1:]
        seed = self.engine.global_random_seed
        import os, random
        import torch
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        prob = [1 / len(self.sidewalk_type_all) for _ in range(len(self.sidewalk_type_all))]
        self.sidewalk_type = np.random.choice(self.sidewalk_type_all, p=prob)
        print('Generated sidewalk type is : ', self.sidewalk_type)
        self.engine.global_config['sidewalk_type'] = self.sidewalk_type
        
        if generate_type == BigGenerateMethod.BLOCK_NUM or generate_type == BigGenerateMethod.BLOCK_SEQUENCE:
            self._big_generate(parent_node_path, physics_world)

        elif generate_type == MapGenerateMethod.PG_MAP_FILE:
            # other config such as lane width, num and seed will be valid, since they will be read from file
            blocks_config = self._config[self.GENERATE_CONFIG]
            self._config_generate(blocks_config, parent_node_path, physics_world)
        else:
            raise ValueError("Map can not be created by {}".format(generate_type))
        self.road_network.after_init()

    def _big_generate(self, parent_node_path: NodePath, physics_world: PhysicsWorld):
        big_map = BIG(
            self._config.get(self.LANE_NUM, 3), ####### 2
            self._config.get(self.LANE_WIDTH, 3.5),
            self.road_network,
            parent_node_path,
            physics_world,
            # self._config["block_type_version"],
            exit_length=self._config.get("exit_length", 50),
            random_seed=self.engine.global_random_seed,
            block_dist_config=self.engine.global_config.get("block_dist_config", PGBlockDistConfig)
        )
        big_map.generate(self._config[self.GENERATE_TYPE], self._config[self.GENERATE_CONFIG])
        self.blocks = big_map.blocks
        self.sidewalks = {}
        self.crosswalks = {}
        self.sidewalks_near_road = {}
        self.sidewalks_farfrom_road = {}
        self.sidewalks_near_road_buffer = {}
        self.sidewalks_farfrom_road_buffer = {}
        self.valid_region = {}
        self.nbs = []
        for block in self.blocks:
            self.sidewalks.update(block.sidewalks)   
            self.crosswalks.update(block.crosswalks)
            
            self.sidewalks_near_road.update(block.sidewalks_near_road)   
            self.sidewalks_farfrom_road.update(block.sidewalks_farfrom_road)
            self.sidewalks_near_road_buffer.update(block.sidewalks_near_road_buffer)   
            self.sidewalks_farfrom_road_buffer.update(block.sidewalks_farfrom_road_buffer)
            self.valid_region.update(block.valid_region)
        big_map.destroy()

    def _config_generate(self, blocks_config: List, parent_node_path: NodePath, physics_world: PhysicsWorld):
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"
        last_block = FirstPGBlock(
            global_network=self.road_network,
            lane_width=self._config.get(self.LANE_WIDTH, 3.5),
            lane_num=self._config.get(self.LANE_NUM, 3),  ###### 2
            render_root_np=parent_node_path,
            physics_world=physics_world,
            length=self._config.get("exit_length", 50),
            ignore_intersection_checking=True
        )
        self.blocks.append(last_block)
        for block_index, b in enumerate(blocks_config[1:], 1):
            block_type = self.engine.global_config["block_dist_config"].get_block(b.pop(self.BLOCK_ID))
            pre_block_socket_index = b.pop(self.PRE_BLOCK_SOCKET_INDEX)
            last_block = block_type(
                block_index,
                last_block.get_socket(pre_block_socket_index),
                self.road_network,
                random_seed=self.engine.global_random_seed,
                ignore_intersection_checking=True
            )
            last_block.construct_from_config(b, parent_node_path, physics_world)
            self.blocks.append(last_block)

    @property
    def road_network_type(self):
        return NodeRoadNetwork

    def get_meta_data(self):
        assert self.blocks is not None and len(self.blocks) > 0, "Please generate Map before saving it"
        map_config = []
        for b in self.blocks:
            b_config = b.get_config()
            json_config = b_config.get_serializable_dict()
            json_config[self.BLOCK_ID] = b.ID
            json_config[self.PRE_BLOCK_SOCKET_INDEX] = b.pre_block_socket_index
            map_config.append(json_config)

        saved_data = copy.deepcopy({self.BLOCK_SEQUENCE: map_config, "map_config": self.config.copy()})
        saved_data.update(super(PGMap, self).get_meta_data())
        return saved_data

    def show_coordinates(self):
        lanes = []
        for to_ in self.road_network.graph.values():
            for lanes_to_add in to_.values():
                lanes += lanes_to_add
        self._show_coordinates(lanes)

    def get_boundary_line_vector(self, interval):
        map = self
        ret = {}
        for _from in map.road_network.graph.keys():
            decoration = True if _from == Decoration.start else False
            for _to in map.road_network.graph[_from].keys():
                for l in map.road_network.graph[_from][_to]:
                    sides = 2 if l is map.road_network.graph[_from][_to][-1] or decoration else 1
                    for side in range(sides):
                        type = l.line_types[side]
                        if type == PGLineType.NONE:
                            continue
                        color = l.line_colors[side]
                        line_type = self.get_line_type(type, color)
                        lateral = l.width_at(0) / 2
                        if side == 0:
                            lateral *= -1
                        ret["{}_{}".format(l.index, side)] = {
                            "type": line_type,
                            "polyline": l.get_polyline(interval, lateral),
                            "speed_limit_kmh": l.speed_limit
                        }
        return ret

    def get_line_type(self, type, color):
        if type == PGLineType.CONTINUOUS and color == PGLineColor.YELLOW:
            return MetaUrbanType.LINE_SOLID_SINGLE_YELLOW
        elif type == PGLineType.BROKEN and color == PGLineColor.YELLOW:
            return MetaUrbanType.LINE_BROKEN_SINGLE_YELLOW
        elif type == PGLineType.CONTINUOUS and color == PGLineColor.GREY:
            return MetaUrbanType.LINE_SOLID_SINGLE_WHITE
        elif type == PGLineType.BROKEN and color == PGLineColor.GREY:
            return MetaUrbanType.LINE_BROKEN_SINGLE_WHITE
        elif type == PGLineType.SIDE:
            return MetaUrbanType.LINE_SOLID_SINGLE_WHITE
        else:
            # Unknown line type
            return MetaUrbanType.LINE_UNKNOWN
