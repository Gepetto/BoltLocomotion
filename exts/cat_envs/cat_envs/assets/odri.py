# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`BOLT_CFG`: BOLT robot
* :obj:`BOLT_MINIMAL_CFG`: BOLT robot with minimal collision bodies

Reference: Anonymous authors
"""

import isaaclab.sim as sim_utils

from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration - Actuators.
##

##
# Configuration
##
BOLT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"exts/cat_envs/cat_envs/assets/Robots/odri/bolt_description/usd/bolt_description.usd",
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
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.50),
        joint_pos={
            "FL_HAA": 0.0,
            "FL_HFE": 0.398,
            "FL_KFE": -0.691,
            "FR_HAA": 0.0,
            "FR_HFE": 0.398,
            "FR_KFE": -0.691,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=["FL_HAA","FL_HFE","FL_KFE", "FR_HAA","FR_HFE","FR_KFE"],
            armature=0.00036207,
            effort_limit=100,
            velocity_limit=100.0,
            stiffness={
                "FL_HAA": 4.0,
                "FL_HFE": 4.0,
                "FL_KFE": 4.0,
                "FR_HAA": 4.0,
                "FR_HFE": 4.0,
                "FR_KFE": 4.0,
            },
            damping={
                "FL_HAA": 0.2,
                "FL_HFE": 0.2,
                "FL_KFE": 0.2,
                "FR_HAA": 0.2,
                "FR_HFE": 0.2,
                "FR_KFE": 0.2,
            },
            min_delay=0,
            max_delay=2,
        ),
    },
)
BOLT_MINIMAL_CFG = BOLT_CFG.copy()