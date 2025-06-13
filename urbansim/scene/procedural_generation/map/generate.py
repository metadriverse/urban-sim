class PGBlockDistConfig:
    MAX_LANE_NUM = 5
    MIN_LANE_NUM = 1

    # Register the block types here! Set their probability to 0.0 if you don't wish it appears in standard metaurban.
    BLOCK_TYPE_DISTRIBUTION_V1 = {
        "Curve": 0.5,
        "Straight": 0.1,
        "StdInterSection": 0.075,
        "Roundabout": 0.05,
        "StdTInterSection": 0.075,
        "InRampOnStraight": 0.1,
        "OutRampOnStraight": 0.1,
        "InFork": 0.00,
        "OutFork": 0.00,
        "Merge": 0.00,
        "Split": 0.00,
        "ParkingLot": 0.00,
        "TollGate": 0.00
    }

    BLOCK_TYPE_DISTRIBUTION_V2 = {
        # 0.3 for curves
        "Curve": 0.4,
        # 0.3 for straight
        "Straight": 0.2,
        "InRampOnStraight": 0.0,
        "OutRampOnStraight": 0.0,
        # 0.3 for intersection
        "StdInterSection": 0.15,
        "StdTInterSection": 0.15,
        # 0.1 for roundabout
        "Roundabout": 0.1,
        "InFork": 0.00,
        "OutFork": 0.00,
        "Merge": 0.00,
        "Split": 0.00,
        "ParkingLot": 0.00,
        "TollGate": 0.00,
        "Bidirection": 0.00
    }

    @classmethod
    def all_blocks(cls, version: str = "v2"):
        ret = list(cls._get_dist(version).keys())
        for k in ret:
            assert isinstance(k, str)
        return ret

    @classmethod
    def get_block(cls, block_id: str, version: str = "v2"):
        from urbansim.scene.procedural_generation.map.blocks.first_block import FirstBlock
        from urbansim.scene.procedural_generation.map.blocks.straight import Straight
        from urbansim.scene.procedural_generation.map.blocks.curve import Curve
        from urbansim.scene.procedural_generation.map.blocks.intersection import Intersection
        all_blocks = [
            FirstBlock,  # This is the first block, which is always a straight road.
            Straight,
            Curve,
            Intersection,
            # Add other block types here as needed.
        ]
        for block in all_blocks:
            if block.ID == block_id:
                return block
        raise ValueError("No {} block type".format(block_id))

    @classmethod
    def block_probability(cls, version: str = "v2"):
        return list(cls._get_dist(version).values())

    @classmethod
    def _get_dist(cls, version: str):
        if version == "v2":
            return cls.BLOCK_TYPE_DISTRIBUTION_V2
        else:
            raise ValueError("Unknown version: {}".format(version))


def generate_map_polygon(scene, stage, config):
    """
    Generate a map polygon for the scene.
    Args:
        scene UrbanScene: Scene instance to modify.
        stage Isaacsim stage: Stage to apply the changes.
        config dict(): Configuration dictionary for the map generation.
    """
    blocks = []
    block_sequences = 'F' + config.get('map_config', 'X')
    assert isinstance(block_sequences, str), "map_config should be a string of block sequences."
    assert len(block_sequences) > 1, "map_config should not be empty."
    print(f"Generating map with block sequences: {block_sequences}, F indicates the first block.")
    
    lane_num = config.get('lane_num', 2)
    lane_width = config.get('lane_width', 3.5)
    exit_length = config.get('exit_length', 50.0)
    
    while len(blocks) < len(block_sequences):
        blocks = sample_block(blocks, block_sequences)
    
    return

def sample_block(blocks, block_sequences):
    type_id = block_sequences[len(blocks)]
    block_type = PGBlockDistConfig.get_block(type_id)
    print(f"Sampling block type: {type_id} ({block_type.ID})")
    
    blocks.append(block_type)
    
    return blocks
