import numpy as np
import yaml
from isaacgym import gymapi
from common import joints_42dof

# config
headless = False
sim_time = 1

# initialize gym
gym = gymapi.acquire_gym()

# create simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1 / 1000
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.num_threads = 4
sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = True
compute_device_id = 0
graphics_device_id = 0
physics_engine = gymapi.SIM_PHYSX
sim = gym.create_sim(compute_device_id, graphics_device_id,
                     physics_engine, sim_params)

# load pose file
pose_file = '/home/sibwang/Projects/IsaacGymPlayground/assets/pose_default.yaml'
with open(pose_file) as f:
    pose = yaml.safe_load(f)
pose['joints'] = {k: np.deg2rad(v) for k, v in pose['joints'].items()}

# create viewer
if not headless:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())

# add ground
gym.add_ground(sim, gymapi.PlaneParams())

# set up env grid
num_envs = 64
spacing = 3
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# add asset
asset_root = '/home/sibwang/Projects/IsaacGymPlayground/assets'
asset_file = 'nmf_no_limits.urdf'
asset_options = gymapi.AssetOptions()  # get default options
asset_options.fix_base_link = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
nmf_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# initial root pose
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 5.0, 0.0)
initial_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), -np.pi / 2)
envs_per_row = 8

# create environments with actors
envs = []
actors = []
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    nmf = gym.create_actor(env, nmf_asset, initial_pose, f'nmf{i}', i, 1)
    envs.append(env)
    actors.append(nmf)
    
    # set DoFs to position control
    dof_props = gym.get_actor_dof_properties(env, nmf)
    dof_props['driveMode'] = gymapi.DOF_MODE_POS
    dof_props["stiffness"].fill(10000)
    dof_props["damping"].fill(50)
    gym.set_actor_dof_properties(env, nmf, dof_props)
    
    # control by position
    for joint in joints_42dof:
        dof = gym.find_actor_dof_handle(env, nmf, joint)
        gym.set_dof_target_position(env, dof, pose['joints'][joint])
gym.prepare_sim(sim)    # required for use_gpu_pipeline=True

# simulate
if sim_time is not None:
    nsteps_to_simulate = int(sim_time / sim_params.dt)
if headless:
    for t_idx in range(nsteps_to_simulate):
        # step simulation
        gym.simulate(sim)
        gym.fetch_results(sim, True)
else:
    t_idx = 0
    while not gym.query_viewer_has_closed(viewer):
        # step simulation
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        # update viewer
        if not headless:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            # make it real time
            gym.sync_frame_time(sim)
        t_idx += 1
        if sim_time is not None and t_idx >= nsteps_to_simulate:
            break