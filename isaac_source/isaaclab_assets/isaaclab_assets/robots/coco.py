# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ANYbotics robots.

The following configuration parameters are available:

* :obj:`ANYMAL_B_CFG`: The ANYmal-B robot with ANYdrives 3.0
* :obj:`ANYMAL_C_CFG`: The ANYmal-C robot with ANYdrives 3.0
* :obj:`ANYMAL_D_CFG`: The ANYmal-D robot with ANYdrives 3.0

Reference:

* https://github.com/ANYbotics/anymal_b_simple_description
* https://github.com/ANYbotics/anymal_c_simple_description
* https://github.com/ANYbotics/anymal_d_simple_description

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetLSTMCfg, DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg

##
# Configuration - Actuators.
##

ANYDRIVE_3_SIMPLE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
    stiffness={".*": 40.0},
    damping={".*": 5.0},
)
"""Configuration for ANYdrive 3.x with DC actuator model."""


ANYDRIVE_3_LSTM_ACTUATOR_CFG = ActuatorNetLSTMCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt",
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
)
"""Configuration for ANYdrive 3.0 (used on ANYmal-C) with LSTM actuator model."""


##
# Configuration - Articulation.
##

ANYMAL_B_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-B/anymal_b.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of ANYmal-B robot using actuator-net."""


ANYMAL_C_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"./Collected_anymal_c/anymal_c.usd",
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_instanceable.usd",
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.61),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of ANYmal-C robot using actuator-net."""


ANYMAL_D_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd",
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d_minimal.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of ANYmal-D robot using actuator-net.

Note:
    Since we don't have a publicly available actuator network for ANYmal-D, we use the same network as ANYmal-C.
    This may impact the sim-to-real transfer performance.
"""


# COCO_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         #mass_props=sim_utils.MassPropertiesCfg( mass =0.05),# TODO: MIGHT CAUSE PROBLEM
#         usd_path=f"assets/coco/coco-default.usd",#f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
#         # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_instanceable.usd",
#         activate_contact_sensors=True,#### might be problematic
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=10.0,
#             max_angular_velocity=180.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False,
#             solver_position_iteration_count=4,
#             solver_velocity_iteration_count=0,
#             sleep_threshold=0.005,
#             #stabilization_threshold=0.001,
#         ),
#         # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#          pos=(1.5,2.577,2.21),
#         joint_pos={
#             ".*wheel_joint*": 0.0,"base_to_front_axle_joint":0.0,
#         },
#         joint_vel={
#             ".*wheel_joint":0.0,"base_to_front_axle_joint":0.0,#TODO: WHY NOT WORKING
#         },
#     ),
#     actuators={"wheels": ImplicitActuatorCfg(
#             joint_names_expr=[".*wheel_joint"],
#             stiffness=0.0,
#             damping=174.0,
#             #velocity_limit=5.,
#         ), "axle": ImplicitActuatorCfg(
#             joint_names_expr=["base_to_front_axle_joint", ".*shock_joint"],

#             effort_limit=80.0, # why not force why not newton
#             #velocity_limit=5.,
#             stiffness=0.0,
#             damping=0.0,
#         )},
#     soft_joint_pos_limit_factor=1.0,
# )
COCO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        #mass_props=sim_utils.MassPropertiesCfg( mass =0.05),# TODO: MIGHT CAUSE PROBLEM
        usd_path=f"coco_one/coco_one.usd",
        #usd_path=f"assets/coco/coco-default.usd",#f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_instanceable.usd",
        activate_contact_sensors=True,#### might be problematic
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            #stabilization_threshold=0.001,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
         pos=(0., 0., 0.31),
        joint_pos={
            ".*wheel_joint*": 0.0,"base_to_front_axle_joint":0.0,
        },
        joint_vel={
            ".*wheel_joint":0.0,"base_to_front_axle_joint":0.0,#TODO: WHY NOT WORKING
        },
    ),
    actuators={"wheels": DelayedPDActuatorCfg(
            joint_names_expr=[".*wheel_joint"],
            #effort_limit=5.0,
            velocity_limit=100.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=4,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                ".*_wheel_joint": 0.0,
            },
            damping={".*_wheel_joint": 0.3},
            friction={
                ".*_wheel_joint": 0.0,
            },
            armature={
                ".*_wheel_joint": 0.0,
            },
            # stiffness=0.0,
            # damping=1740.0,
        ), "axle": 
            DCMotorCfg(
            joint_names_expr=["base_to_front_axle_joint"],
            saturation_effort=64.0,
            effort_limit=64.0,
            velocity_limit=20.,
            stiffness=25.0,
            damping=0.5,
            friction=0.
        ),
            # ImplicitActuatorCfg(
            # joint_names_expr=["base_to_front_axle_joint"],

            # #effort_limit=80.0,
            # #velocity_limit=7.5,
            # stiffness=1000.0,
            # damping=50.0,
        # ), 
        
    "shock": ImplicitActuatorCfg(
            joint_names_expr=[".*shock_joint"],

            #effort_limit=80.0,
            #velocity_limit=7.5,
            stiffness=0.0,
            damping=0.0,
        )
        
        },
    soft_joint_pos_limit_factor=1.0,# TODO: 0.95
)