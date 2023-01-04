import xml.etree.ElementTree as ET
from urdfpy import URDF
import urdfpy
import yaml
from scipy.spatial.transform import Rotation as R
import numpy as np


if __name__ == '__main__':
    df3dpp_to_nmf_names = {
        'ThC_pitch': 'Coxa', 'ThC_roll': 'Coxa_roll', 'ThC_yaw': 'Coxa_yaw',
        'CTr_pitch': 'Femur', 'CTr_roll': 'Femur_roll',
        'FTi_pitch': 'Tibia', 'TiTa_pitch': 'Tarsus1'
    }
    nmf_to_df3dpp_names = {v: k for k, v in df3dpp_to_nmf_names.items()}
    
    # The default position of certain fixed joints as specified in the
    # mesh files lead to unnatural poses. They should be set to non-zero
    # fixed positions for better visualization. Counterintuitively, to
    # be set to such fixed positions, they need to be unfixed in the
    # SDF. Here's a set of such joints.
    # fixed_joints_to_unfix = {
    #     'joint_A3', 'joint_A4', 'joint_A5', 'joint_A6',
    #     'joint_LAntenna', 'joint_RAntenna',
    #     'joint_Rostrum', 'joint_Haustellum', 'joint_Head',
    #     'joint_LWing_roll', 'joint_LWing_yaw',
    #     'joint_RWing_roll', 'joint_RWing_yaw'
    # }
    
    # Load SDF
    sdf = ET.parse('/home/achard/isaacgym/python/isaacGymEnvs/isaacgymenvs/minimal-isaacgym/fly_bProject/assets/nmf_no_limits.urdf')
    root = sdf.getroot()

    
    # Load joint limits
    with open('/home/achard/isaacgym/python/isaacGymEnvs/isaacgymenvs/minimal-isaacgym/fly_bProject/assets/pose_default_rad.yaml') as f:
        initial_joints_dict = yaml.safe_load(f)

    #print(initial_joints_dict)
    action_names = ["joint_LHCoxa_roll", "joint_RHCoxa_roll", "joint_LHFemur", "joint_RHFemur", "joint_LHTibia", "joint_RHTibia",
                    "joint_LMCoxa_roll", "joint_RMCoxa_roll", "joint_LMFemur", "joint_RMFemur", "joint_LMTibia", "joint_RMTibia",
                     "joint_LFCoxa", "joint_RFCoxa", "joint_LFFemur", "joint_RFFemur", "joint_LFTibia", "joint_RFTibia"]
    
    """"
    count = 0
    for joint in robot.actuated_joints:
        if joint.name not in action_names:
            count+=1
            joint.joint_type = "fixed"
            if all(joint.axis == [1, 0, 0]):
                joint.origin[:3,:3] = urdfpy.rpy_to_matrix([initial_joints_dict[joint.name], 0, 0])
            elif all(joint.axis == [0, 1, 0]):
                joint.origin[:3,:3] = urdfpy.rpy_to_matrix([0, initial_joints_dict[joint.name], 0])
            elif all(joint.axis == [0, 0, 1]):
                joint.origin[:3,:3] = urdfpy.rpy_to_matrix([0, 0, initial_joints_dict[joint.name]])
            print(joint.origin)

    assert(count == 42-18)
    """
    
    
    
    
    # Modify joint limits in SDF
    actuated_leg_dof_count = 0
    changed_orient = 0
    for child in root:
        if child.tag != 'joint':
            continue
        joint_name_complete = child.attrib['name']
        joint_name = child.attrib['name'].replace('joint_', '')
        if 'support' in joint_name:
            continue
        is_actuated_leg_dof = ((joint_name[0] in ['L', 'R']) and
                               (joint_name[1] in ['F', 'M', 'H']) and
                               (joint_name[2:] in nmf_to_df3dpp_names))
        if is_actuated_leg_dof:
            if joint_name_complete not in action_names:
                actuated_leg_dof_count += 1
                child.set('type', 'fixed')
                child_axis = []
                for grandchild in child:
                    if grandchild.tag == "axis":
                        child_axis = grandchild.attrib["xyz"]
                for grandchild in child:
                    if grandchild.tag != "origin":
                        continue
                    value = initial_joints_dict[joint_name_complete]
                    if child_axis == "1 0 0":
                        changed_orient+=1
                        grandchild.set("rpy", f"{value} 0 0") 
                    elif child_axis== "0 1 0":
                        changed_orient+=1
                        grandchild.set("rpy", f"0 {value} 0") 
                    elif child_axis == "0 0 1":
                        changed_orient+=1
                        grandchild.set("rpy", f"0 0 {value}") 

    assert actuated_leg_dof_count == 42-18
    assert changed_orient == 42-18


    sdf.write(
        '/home/achard/isaacgym/python/isaacGymEnvs/isaacgymenvs/minimal-isaacgym/fly_bProject/assets/nmf_no_limits_limited_Dofs.urdf',
        xml_declaration=True)