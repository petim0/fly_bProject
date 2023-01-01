from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *

import sys
import torch
import yaml
import torchgeometry as tgm
import os

class Fly:
    def __init__(self, args):
        #TODO CHAGER LA VITESS MAX DE CHAQUE JOINT !!!
        self.args = args
        self.end = False
        self.dt = 1 / 60 #was 1000.
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        # configure sim (gravity is pointing down)
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81) #should be *1000
        sim_params.dt = self.dt
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = True

        # set simulation parameters (we use PhysX engine by default, these parameters are from the example file)
        ### I should maybe change these params, see doc here file:///Users/petimo/Desktop/IsaacGym_Preview_4_Package/isaacgym/docs/api/python/struct_py.html#isaacgym.gymapi.PhysXParams
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.rest_offset = 0.001
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.use_gpu = True
        self.i = 0
        # task-specific parameters
        self.num_act = 17 #(3 DoFs * 6 legs)
        self.num_obs = 97 + self.num_act  # See compute_fly_observations
        self.starting_height = 2.2
        #ThC pitch for the front legs (joint_RFCoxa), ThC roll (joint_LMCoxa_roll) for the middle and hind legs, and CTr pitch (joint_RFFemur) and FTi pitch (joint_LFTibia) for all leg
        self.max_episode_length = 1000  # maximum episode length
        self.render_count = 0
        self.names = ["joint_LHCoxa_roll", "joint_RHCoxa_roll", "joint_LHFemur", "joint_RHFemur", "joint_LHTibia", "joint_RHTibia",
                    "joint_LMCoxa_roll", "joint_RMCoxa_roll", "joint_LMFemur", "joint_RMFemur", "joint_LMTibia", "joint_RMTibia",
                     "joint_LFCoxa", "joint_RFCoxa", "joint_LFFemur", "joint_RFFemur", "joint_LFTibia", "joint_RFTibia"]
        
        self.names = ["joint_LHCoxa_roll", "joint_RHCoxa_roll", "joint_LHFemur", "joint_RHFemur", "joint_LHTibia", "joint_RHTibia",
                     "joint_LMCoxa_roll", "joint_RMCoxa_roll", "joint_RMFemur", "joint_LMTibia", "joint_RMTibia",
                     "joint_LFCoxa", "joint_RFCoxa", "joint_LFFemur", "joint_RFFemur", "joint_LFTibia", "joint_RFTibia"]
        
        #self.names = ["joint_LHCoxa_roll", "joint_RHCoxa_roll", "joint_LHFemur", "joint_RHFemur", "joint_LHTibia", "joint_RHTibia",
         #            "joint_LFCoxa", "joint_RFCoxa", "joint_LFFemur", "joint_RFFemur", "joint_LFTibia", "joint_RFTibia"]
        #self.names = ["joint_LHTibia", "joint_RHTibia",
         #            "joint_LMTibia", "joint_RMTibia",
          #          "joint_LFTibia", "joint_RFTibia"]
        #self.names = ["joint_LHCoxa_roll", "joint_RHCoxa_roll",
        #            "joint_LMCoxa_roll", "joint_RMCoxa_roll",
        #            "joint_LFCoxa", "joint_RFCoxa"]
        #self.names = ["joint_LFFemur"]

        self.joints_limits = {
            "joint_LHCoxa_roll" : {'lower': 0.6012615998580322, 'upper': 4.120341207989709}, 
            "joint_RHCoxa_roll": {'lower': -4.120341207989709, 'upper': -0.6012615998580322}, 
            "joint_LHFemur": {'lower': -5.553724929606129, 'upper': 1.6139985085925022}, 
            "joint_RHFemur": {'lower': -5.553724929606129, 'upper': 1.6139985085925022}, 
            "joint_LHTibia": {'lower': -3.8187837662418334, 'upper': 6.979499524663906}, 
            "joint_RHTibia": {'lower': -3.8187837662418334, 'upper': 6.979499524663906},
            "joint_LMCoxa_roll": {'lower': -0.1644733111051202, 'upper': 3.843949339634286}, 
            "joint_RMCoxa_roll": {'lower': -3.843949339634286, 'upper': 0.1644733111051202}, 
            "joint_LMFemur": {'lower': -3.8856558255692613, 'upper': 0.2503410005690172}, 
            "joint_RMFemur": {'lower': -3.8856558255692613, 'upper': 0.2503410005690172}, 
            "joint_LMTibia": {'lower': -2.5514814160669523, 'upper': 5.025832418893524}, 
            "joint_RMTibia": {'lower': -2.5514814160669523, 'upper': 5.025832418893524},
            "joint_LFCoxa": {'lower': -1.2282643976845713, 'upper': 1.4495346989023457}, 
            "joint_RFCoxa": {'lower': -1.2282643976845713, 'upper': 1.4495346989023457}, 
            "joint_LFFemur": {'lower': -4.986930927481532, 'upper': 1.4560609499793291}, 
            "joint_RFFemur": {'lower': -4.986930927481532, 'upper': 1.4560609499793291}, 
            "joint_LFTibia": {'lower': -2.362989686468837, 'upper': 4.222732123265363},
            "joint_RFTibia": {'lower': -2.362989686468837, 'upper': 4.222732123265363}
        }

        self.plane_static_friction = 100.0
        self.plane_dynamic_friction = 1.0

        #Constants for the reward function, taken from ant
        self.dof_vel_scale = 0.2
        self.heading_weight = 0.5
        self.up_weight = 0.1
        self.actions_cost_scale = 0.005
        self.energy_cost_scale = 0.05
        self.joints_at_limit_cost_scale = 0.1
        self.death_cost = -2.0
        self.termination_height = 1 #PEUT être TROP petit !!
        self.termination_height_up = 5 #A jouer avec 

        # allocate buffers
        #obs_buf size will have to change TODO 
        self.obs_buf = torch.zeros((self.args.num_envs, self.num_obs), device=self.args.sim_device) #Observation given to the NN
        self.reward_buf = torch.zeros(self.args.num_envs, device=self.args.sim_device)
        self.reset_buf = torch.ones(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)
        
        # Distance that a fly ha made since the last reset
        self.distance = torch.zeros(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)
        #Time it has spend too close to the ground
        self.timer_down = torch.zeros(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)

        # acquire gym interface
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)

        initial_joints_file = 'assets/pose_default.yaml'
        with open(initial_joints_file) as f:
            initial_joints_dict = yaml.safe_load(f)
        self.initial_joints_dict = {k: np.deg2rad(v) for k, v in initial_joints_dict['joints'].items()}
        ## ATTENTION ceci ne marche pas si on ne prends pas pose default psk pause stretch ne prends pas toutes les dofs##
        self.joints_92dof = self.initial_joints_dict.keys()
        self.num_total_dof = len(self.joints_92dof) # Not equal to num_dof because this contains all the dofs even the ones you can't moove

        #print("Nb of base positions: ", len(self.joints_92dof)) #92

        # initialise envs and state tensors
        self.envs, self.num_dof, self.action_indexes, self.action_indexes_one, self.initial_dofs, self.initial_dofs_one, self.translation, self.multiplication, self.dof_limits_lower, self.dof_limits_upper  = self.create_envs()
        self.dof_states, self.root_tensor = self.get_states_tensor()
        self.dof_pos = self.dof_states.view(self.args.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_states.view(self.args.num_envs, self.num_dof, 2)[..., 1]

        #It is always 13: 3 floats for position, 4 floats for quaternion, 3 floats for linear velocity, and 3 floats for angular velocity.
        #self.old_position = torch.zeros((self.args.num_envs, 3), device=self.args.sim_device)
        self.root_positions = self.root_tensor.view(self.args.num_envs, 13)[:, 0:3] 
        self.root_orientations = self.root_tensor.view(self.args.num_envs, 13)[:, 3:7] #THOSE ORIENTATION ARE NOT ADDING UP TO ONE !!
        self.root_linvels = self.root_tensor.view(self.args.num_envs, 13)[:, 7:10]
        self.root_angvels = self.root_tensor.view(self.args.num_envs, 13)[:, 10:13]
        
        self.origin_pos = torch.zeros((self.args.num_envs, 3), device=self.args.sim_device)
        self.origin_root_tensor = torch.zeros((self.args.num_envs, 13), device=self.args.sim_device) #self.root_tensor.clone() ne marchera pas 
        #psk on le fait avant de lancer la simulation
        self.origin_root_tensor[:,2] = self.starting_height
        self.origin_root_tensor[:,6] = 1 #the last quaterion should be one
        self.origin_root_tensor_one = self.origin_root_tensor[0]

        self.origin_orientations = self.origin_root_tensor.view(self.args.num_envs, 13)[:, 3:7]

        # generate viewer for visualisation
        self.viewer = self.create_viewer()

        cameras_initial_pos = gymapi.Vec3(30, 0.0, 10)
        if self.args.record:
            self.record_dir_name = self.args.record_dir_name
            self.time_steps_per_recorded_frame = self.args.time_steps_per_recorded_frame
            self.camera, self.record_root_dir, self.record_command = self.set_up_recording(cameras_initial_pos, self.pose)
        
        #Initialise other values for reward and obs buffer 
        self.potentials = to_torch([-1000./self.dt], device=self.args.sim_device).repeat(self.args.num_envs)
        self.prev_potentials = self.potentials.clone()
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.args.sim_device).repeat((self.args.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.args.sim_device).repeat((self.args.num_envs, 1))

        self.inv_start_rot = quat_conjugate(self.origin_orientation).repeat((self.args.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = to_torch([1000, 0, 0], device=self.args.sim_device).repeat((self.args.num_envs, 1)) #was [1000, 0, 0]
        self.target_dirs = to_torch([1, 0, 0], device=self.args.sim_device).repeat((self.args.num_envs, 1)) #was [1, 0, 0]

        
        # step simulation to initialise tensor buffers
        self.gym.prepare_sim(self.sim)
        
        #self.reset() #Ne Peux pas être là !
        ## First reset to start on smth frseh, This has to be modified, it is ugly
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.dof_pos[env_ids, :] = self.initial_dofs_one[..., 0]
        self.dof_vel[env_ids, :] = self.initial_dofs_one[..., 1]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_states),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.distance[env_ids] = 0


        #Stuff I have to init after starting the simulation
        print("velocity", self.PROP["velocity"][0])
        print("effort", self.PROP["effort"][0])
        print("stiffness", self.PROP["stiffness"][0])
        print("damping", self.PROP["damping"][0])

        
    def create_envs(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = 0.0
        plane_params.normal = gymapi.Vec3(0, 0, 1) # z-Up
        self.gym.add_ground(self.sim, plane_params)

        # define environment space (for visualisation)
        spacing = 7
        lower = gymapi.Vec3(-spacing, 0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(np.sqrt(self.args.num_envs))

        # add cartpole asset
        asset_root = 'assets'
        asset_file = 'nmf_no_limits.urdf'
        asset_options = gymapi.AssetOptions() #get default options
        asset_options.fix_base_link = False
        fly_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        num_dof = self.gym.get_asset_dof_count(fly_asset)
        #print("num_dof: ", num_dof) #42
        self.joints_42dof = self.gym.get_asset_dof_names(fly_asset) ## GIVES ONLY THE DOFS WHICH YOU CAN HAVE AN ACTION ON !!! so 42
        # print("Dof_names: ", self.joints_42dof)
        # define fly pose
        pose = gymapi.Transform()
        pose.p.z = self.starting_height   # generate the fly 3m from the ground
        self.origin_orientation = torch.tensor([pose.r.x, pose.r.y, pose.r.z, pose.r.w], device=self.args.sim_device)
        #pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi / 8) # No rotation needed 
        self.pose = pose #Degeu TODO 

        # define fly dof properties
        dof_props = self.gym.get_asset_dof_properties(fly_asset)
        dof_props['driveMode'] = gymapi.DOF_MODE_POS
        dof_props['stiffness'].fill(4) #This cannot be over a certain value idk what ... At least lower than 1000000 
        dof_props['damping'].fill(0.1)
        dof_props['velocity'].fill(50)
        dof_props['effort'].fill(3.4e+38)
        
        self.PROP = dof_props
        # generate environments
        envs = []
        actors = []
        print(f'Creating {self.args.num_envs} environments.')
        for i in range(self.args.num_envs):
            # create env
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # add fly here in each environment
            fly = self.gym.create_actor(env, fly_asset, pose, f'fly{i}', i, 1, 0)
            self.gym.set_actor_dof_properties(env, fly, dof_props)
            
            envs.append(env)
            actors.append(fly)

        
        dof_limits_lower = []
        dof_limits_upper = []
        dof_prop = self.gym.get_actor_dof_properties(envs[0], actors[0])
        for j in range(num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                dof_limits_lower.append(dof_prop['upper'][j])
                dof_limits_upper.append(dof_prop['lower'][j])
            else:
                dof_limits_lower.append(dof_prop['lower'][j])
                dof_limits_upper.append(dof_prop['upper'][j])
        
        dof_limits_lower = to_torch(dof_limits_lower, device=self.args.sim_device)
        dof_limits_upper = to_torch(dof_limits_upper, device=self.args.sim_device)



        # Find the indexes we want to modify, these indexes are relative to the sim 0 and 42*num_envs
        # It should have a size num_action*num_envs
        # We also calculate the translation and multiplication factor we want to apply, this could be done without hardcoding 
        # the values in a dictionnary using get_actor_dof_properties() !! Don't have the time to change 

        action_indexes = torch.full((self.num_act * self.args.num_envs, 1), 0, dtype=torch.long, device=self.args.sim_device)
        dof_translation = torch.full((self.num_act * self.args.num_envs, 1), 0, dtype=torch.float, device=self.args.sim_device)
        dof_multiplication = torch.full((self.num_act * self.args.num_envs, 1), 0, dtype=torch.float, device=self.args.sim_device)
        #maxU = torch.full((self.num_act * self.args.num_envs, 1), 0, dtype=torch.float, device=self.args.sim_device)

        #lower = []
        j = 0
        for i in range(self.args.num_envs):
            for name in self.names:
                action_indexes[j] = self.gym.find_actor_dof_index(envs[i], actors[i], name, gymapi.DOMAIN_SIM)
                #lower.append(self.joints_limits[name]["lower"])
                dof_translation[j] = self.find_trans(self.joints_limits[name]["lower"], self.joints_limits[name]["upper"])
                dof_multiplication[j] = self.find_mult(self.joints_limits[name]["lower"], self.joints_limits[name]["upper"])
                #maxU[j] = self.joints_limits[name]["upper"]
                j+=1
        #lower = to_torch(lower, device=self.args.sim_device)
        action_indexes, indexosef = torch.sort(action_indexes, 0)

        #action_indexes for just one fly (indexes relative to the env) 
        action_indexes_one = action_indexes[0:self.num_act]

        indexosef = indexosef.squeeze(-1)        
        dof_translation = dof_translation[indexosef]
        dof_multiplication = dof_multiplication[indexosef]
        #maxU = maxU[indexosef]

        #It's the same thing ! But the second one is easier, I'll change it when I have some time !
        #print(lower[indexosef][0:self.num_act])
        #print(self.dof_limits_lower[action_indexes_one])

        self.left_leg_indicies = [self.gym.find_actor_dof_index(envs[i], actors[i], "joint_LMCoxa_roll", gymapi.DOMAIN_ENV), 
                                self.gym.find_actor_dof_index(envs[i], actors[i], "joint_LMFemur", gymapi.DOMAIN_ENV),
                                self.gym.find_actor_dof_index(envs[i], actors[i], "joint_LMTibia", gymapi.DOMAIN_ENV)]

        print("Setting initial dof position")

        # Set initial joint angles for each dofs of the fly of all the flys, with no velocity set for all dofs, should be of size self.num_dof * self.args.num_envs
        # We keep this initial_dof, we will use it later to only moove some dofs 
        initial_dofs = torch.full((num_dof*self.args.num_envs, 2), 0, dtype=torch.float32, device=self.args.sim_device) 
        for i in range(self.args.num_envs):
            for joint_name in self.joints_42dof:
                joint_index = self.gym.find_actor_dof_index(envs[i], actors[i], joint_name, gymapi.DOMAIN_SIM)
                initial_dofs[joint_index, 0] = self.initial_joints_dict.get(joint_name, 0) # defaults to 0 for unspecified joints
        
        #Initial_dof for just one fly, useful in the reset 
        initial_dofs_one = initial_dofs[:num_dof]        
        
        return envs, num_dof, action_indexes, action_indexes_one, initial_dofs, initial_dofs_one, dof_translation, dof_multiplication, dof_limits_lower, dof_limits_upper

    #find the translation to apply to transform a value between [-1 1] to [lower upper]
    def find_trans(self, lower: float, upper: float):
        return (upper+lower)/2.0 

    def find_mult(self, lower: float, upper: float):
        return (upper-lower)/2.0


    def create_viewer(self):
        # create viewer for debugging (looking at the center of environment)
        
        self.enable_viewer_sync = True
        viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.args.headless == False:
            # subscribe to keyboard shortcuts
            viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

            self.gym.subscribe_viewer_keyboard_event(
                viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(
                viewer, gymapi.KEY_R, "Reset_everyone")
            self.gym.subscribe_viewer_keyboard_event(
                viewer, gymapi.KEY_E, "End_simulation")
            self.gym.subscribe_viewer_keyboard_event(
                viewer, gymapi.KEY_P, "print")     
            
        cam_pos = gymapi.Vec3(30, 0.0, 10)
        cam_target = gymapi.Vec3(-1, 0, 0)
        self.gym.viewer_camera_look_at(viewer, self.envs[self.args.num_envs // 2], cam_pos, cam_target)

        return viewer

    def get_states_tensor(self):
        # get dof state tensor of All the flys
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        dof_states = dof_states.view(self.args.num_envs, self.num_dof*2)

        # acquire root state tensor descriptor
        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)

        # wrap it in a PyTorch Tensor and create convenient views
        root_tensor = gymtorch.wrap_tensor(_root_tensor)
        #print("Roooot: ", root_tensor.size())

        return dof_states, root_tensor

    def get_obs(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim) 
        
        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = compute_fly_observations(
            self.obs_buf, self.root_tensor, self.targets, self.potentials,
            self.inv_start_rot, self.dof_pos, self.dof_vel,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale, 
            self.actions.view(self.args.num_envs, -1), self.dt,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)

    def get_reward(self):
        # retrieve environment observations from buffer
        nb_of_sim_step = self.progress_buf
        nb_of_sim_step[nb_of_sim_step==0] = 1

        #self.reward_buf[:], self.reset_buf[:], self.timer_down[:] = compute_fly_reward(nb_of_sim_step, self.timer_down, self.origin_pos, 
            #self.root_positions, self.reset_buf, self.max_episode_length, 250)

        self.reward_buf[:], self.reset_buf[:] = compute_fly_reward2(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.actions.view(self.args.num_envs, -1),
            self.up_weight,
            self.heading_weight,
            self.potentials,
            self.prev_potentials,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.joints_at_limit_cost_scale,
            self.termination_height,
            self.termination_height_up,
            self.death_cost,
            self.max_episode_length, 
            self.action_indexes_one, 
            self.dof_limits_upper,
            self.dof_limits_lower,
            self.args.num_envs,
            self.root_orientations,
            self.num_act
        )
        
        #self.old_position = self.root_positions
    
    ## Can be made better  
    def reset(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
       
        if len(env_ids) == 0:
            return False

        # initial dof [{pos1, vel1},{pos2, vel2},....,{posn,veln}]
        # Can be done differently, we could not pass by these dof_pos, dof_vel, and do smth like self.dof_state[env_ids, :] = smth I'll see later If I have the courage to change that
        self.dof_pos[env_ids, :] = self.initial_dofs_one[..., 0]
        self.dof_vel[env_ids, :] = self.initial_dofs_one[..., 1]
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.root_tensor[env_ids, :] = self.origin_root_tensor[env_ids, :] 

        # Reset desired environments 
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_tensor),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_states),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # clear up desired buffer states
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.timer_down[env_ids] = 0
        self.distance[env_ids] = 0
    
        #YOU CANNOT GET_OBS HERE 
        return True 

    def simulate(self):
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def render(self):
        # update viewer
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif  evt.action == "Reset_everyone" and evt.value > 0:
                    print("We reset everyone !")
                    self.reset_buf = torch.ones(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)
                elif evt.action == "End_simulation" and evt.value > 0:
                    print("We end the simulation !!!")
                    self.end = True
                elif evt.action == "print" and evt.value > 0:
                    print(self.actions[self.left_leg_indicies])
                    

            # fetch results
            if self.args.sim_device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

            else:
                self.gym.poll_viewer_events(self.viewer)

    # returns (camera, record_root_dir, record_command)
    def set_up_recording(self, cameras_initial_pos, cameras_look_at_pos):

        recorded_real_frame_rate = 1.0 / (self.time_steps_per_recorded_frame * self.dt)
        if not recorded_real_frame_rate.is_integer():
            print(f"Warning: simulation recording real frame rate is not an integer ({recorded_real_frame_rate:.2f}), script will not be able to make a video out of the recorded frames")

        # create camera
        camera = self.gym.create_camera_sensor(self.envs[0], gymapi.CameraProperties())
        self.gym.set_camera_location(camera, self.envs[0], cameras_initial_pos, gymapi.Vec3(-1, 0, 0)) #TODO 

        # set up directories
        root = "recordings"
        # create directory if it doesn't exist yet
        if not os.path.exists(root):
            os.mkdir(root)
        assert not os.path.exists(f'{root}/{self.record_dir_name}'), "You are trying to record the simulation but the directory name of your recording is in conflict with a previously saved recording."
        os.mkdir(f'{root}/{self.record_dir_name}')
        print(f"Will record the simulation in {root}/{self.record_dir_name} with real frame rate {recorded_real_frame_rate:.2f}")

        # generate command
        command = f'ffmpeg -r {int(recorded_real_frame_rate)} -start_number 0 -i {root}/{self.record_dir_name}/%d.png -vcodec libx264 -crf 18 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y {root}/{self.record_dir_name}/video_{int(recorded_real_frame_rate)}fps.mp4'
        print("Command to create video manually (if simulation crashes): ")
        print(command)
        if not recorded_real_frame_rate.is_integer():
            print("Warning: note that this command will not generate a video that runs at real speed given that the command frame rate was rounded")
        return camera, root, command
       
    def generate_video(self):
        recorded_real_frame_rate = 1.0 / (self.time_steps_per_recorded_frame * self.dt)

        if not recorded_real_frame_rate.is_integer():
            print(f"Warning: recording frame rate is not an integer ({recorded_real_frame_rate:.2f}), the script will not create a video")
            return

        print("Generating video from recorded frames...")
        if os.system(self.record_command) != 0:
            print("Error while creating the video")
            return
        print("Done")
        if os.system(f'rm ./{self.record_root_dir}/{self.record_dir_name}/*.png') != 0:
            print("Error while trying to remove the recorded frames")
            return
        print("Removed all recorded frames")
    
    def record_frame(self, t_idx):
        self.gym.render_all_camera_sensors(self.sim)
        filename = f'{self.record_root_dir}/{self.record_dir_name}/{t_idx // self.time_steps_per_recorded_frame}.png'
        self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.camera, gymapi.IMAGE_COLOR, filename)

    def exit(self):
        # close the simulator in a graceful way
        if self.args.record:
            self.generate_video()  

        if not self.args.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        
    
    def step(self, actions):
        # apply action
        # Reçois un tenseur de la taille num_actions * num_envs du coup il faut le mettre correctement pour que ça fasse num_obs * num_envs
        ##actions_tensor = torch.zeros(self.args.num_envs * self.num_dof, device=self.args.sim_device)
        ##actions_tensor[::self.num_dof] = actions.squeeze(-1) * self.max_push_effort
        actions_tensor = torch.clone(self.initial_dofs).detach()
        actions = (actions.view(self.args.num_envs * self.num_act, 1) * self.multiplication) + self.translation
        self.actions = actions.clone()
        
        #Replaces in the mask the values of the actions
        actions_tensor[..., 0][self.action_indexes] = actions
        
        #We only want position !! No velocity 
        actions_pos_only = actions_tensor[...,0].contiguous()
        #print(actions_pos_only.size())
        #Sets the target position dicted by the actions 
        positions = gymtorch.unwrap_tensor(actions_pos_only)
        targets_set = self.gym.set_dof_position_target_tensor(self.sim, positions)
        #print("Target well set ? ", targets_set)

        self.reset()

        # simulate and render
        self.simulate()
        
        if not self.args.headless :
            if self.render_count % 1 == 0:
                self.render()
        
        render_camera = self.args.record and (self.render_count % self.time_steps_per_recorded_frame == 0)
        if render_camera:
            # update camera and save file
            self.record_frame(self.render_count)

        self.render_count+=1

        #You cannot get obs if you reset 
        self.get_obs()        
        self.progress_buf += 1
        self.get_reward()


# define reward function using JIT
@torch.jit.script
def compute_fly_reward(time, timer_down, origin_pos, pos, reset_buf, max_episode_length, max_time_down):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float) -> Tuple[Tensor, Tensor, Tensor]
    

    #We only want progress in the X axis 
    dist_x = pos[...,0] - origin_pos[...,0]
    
    
    reward = dist_x / time 
    reward += pos[..., 2]

    timer_down[pos[..., 2]< 1] += 1
    
    # adjust reward for reset agents
    reward = torch.where(time > max_episode_length, torch.ones_like(reward) * -2.0, reward)
    reward = torch.where(timer_down > max_time_down, torch.ones_like(reward) * -2.0, reward)

    reset = torch.where(time > max_episode_length, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(timer_down > max_time_down, torch.ones_like(reset), reset)
    #reset = torch.where(rotation < rot or rotation < rot2 or rotation < rot3 , torch.ones_like(reset), reset)
    
    return reward, reset, timer_down

@torch.jit.script
def compute_fly_reward2(
    obs_buf,
    reset_buf,
    progress_buf,
    actions,
    up_weight,
    heading_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    joints_at_limit_cost_scale,
    termination_height,
    termination_height_up,
    death_cost,
    max_episode_length, 
    action_indicies_one,
    upper_limit_of_actions, 
    lower_limit_of_actions, 
    num_env,
    orientation,
    num_actions 
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, float, float, float, Tensor, Tensor, Tensor, int, Tensor, int) -> Tuple[Tensor, Tensor]

    # reward from direction headed
    heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    heading_reward = torch.where(obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8)

    # aligning up axis of ant and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(obs_buf[:, 10] > 1.3, up_reward + up_weight, up_reward)

    # Aligned with origin orientation 
    # put heading_weigth there just because I think orientation should be as important as heading
    # works because base orientation is 1 + 0i + 0j + 0k and to be on the same plane reel^2 + k^2 should be = 1
    orient_reward = torch.zeros_like(up_reward)
    orient_reward = torch.where(torch.square(orientation[:, 2])  + torch.square(orientation[:, 3]) > 0.92, orient_reward + heading_weight, orient_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions ** 2, dim=-1)
    #Plus la diff de mouvement est grande plus ça coute  
    electricity_cost = torch.sum(torch.abs(actions - obs_buf[:, 96:(96 + num_actions)]), dim=-1)
   
    #Be at the the extremities costs  
    #print(obs_buf[:, 96:114].size(), upper_limit_of_actions[action_indicies_one].squeeze(-1).repeat((num_env, 1)).size(), upper_limit_of_actions[action_indicies_one].squeeze(-1).repeat((10, 1))) 
    dof_at_limit_cost = torch.sum(obs_buf[:, 96:(96 + num_actions)] > upper_limit_of_actions[action_indicies_one].squeeze(-1).repeat((num_env, 1)) * 0.9, dim=-1)
    dof_at_limit_cost += torch.sum(obs_buf[:, 96:(96 + num_actions)] < lower_limit_of_actions[action_indicies_one].squeeze(-1).repeat((num_env, 1)) * 0.9, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * 0.5
    progress_reward = potentials - prev_potentials

    total_reward = progress_reward * 3 + alive_reward + up_reward + heading_reward - \
        actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost * joints_at_limit_cost_scale + orient_reward

    # adjust reward for fallen agents
    total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward)
    total_reward = torch.where(obs_buf[:, 0] > termination_height_up, torch.ones_like(total_reward) * death_cost, total_reward)

    # reset agents
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(obs_buf[:, 0] > termination_height_up, torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return total_reward, reset


@torch.jit.script
def compute_fly_observations(obs_buf, root_states, targets, potentials,
                             inv_start_rot, dof_pos, dof_vel,
                             dof_limits_lower, dof_limits_upper, dof_vel_scale, actions, dt,
                             basis_vec0, basis_vec1, up_axis_idx):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, float, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)
    
    # ATTENTION DOF POSITION ET ACTION EST UNE Répétition du coup ici 
    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs(42), num_dofs(42), num_actions(18), 1 = 115
    obs = torch.cat((torso_position[:, up_axis_idx].view(-1, 1), vel_loc, angvel_loc,
                     yaw.unsqueeze(-1), roll.unsqueeze(-1), angle_to_target.unsqueeze(-1),
                     up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1), dof_pos_scaled,
                     dof_vel * dof_vel_scale, actions, pitch.unsqueeze(-1)), dim=-1)

    return obs, potentials, prev_potentials_new, up_vec, heading_vec
