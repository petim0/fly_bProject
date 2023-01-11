from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *

import sys
import torch
import yaml
import os

class Fly:
    def __init__(self, args):
        self.print_once = True #TODO
        self.args = args
        self.end = False
        self.dt = 1 / 60
        self.up_axis_idx = 2 # index of up axis: X= 0, Y=1, Z=2
        self.i = 0
        # task-specific parameters

        # Self.names contains the names of all the joints we want to apply an action on, currently these are the names
        # of all the dofs we can have an action on as I changed the urdf file. This can be changed by changing again the urdf file 
        self.names = ["joint_LHCoxa_roll", "joint_RHCoxa_roll", "joint_LHFemur", "joint_RHFemur", "joint_LHTibia", "joint_RHTibia",
                    "joint_LMCoxa_roll", "joint_RMCoxa_roll", "joint_LMFemur", "joint_RMFemur", "joint_LMTibia", "joint_RMTibia",
                     "joint_LFCoxa", "joint_RFCoxa", "joint_LFFemur", "joint_RFFemur", "joint_LFTibia", "joint_RFTibia"]
        
        #self.names = ["joint_LHFemur", "joint_RHFemur", "joint_LHTibia", "joint_RHTibia",
         #           "joint_LMFemur", "joint_RMFemur", "joint_LMTibia", "joint_RMTibia",
          #          "joint_LFFemur", "joint_RFFemur", "joint_LFTibia", "joint_RFTibia"]

        self.num_act = len(self.names) #(3 DoFs * 6 legs)
        self.num_obs = 19 + 3*self.num_act  # See compute_fly_observations
        self.starting_height = 2
        self.max_episode_length = 1500  # maximum episode length
        self.render_count = 0
        
        

        self.plane_static_friction = 10.0 #was 1.0
        self.plane_dynamic_friction = 10.0 #was 1.0

        # constants for the reward function
        self.dof_vel_scale = 0.2
        self.heading_weight = 0.5
        self.up_weight = 0.75 #TODO was 1
        self.actions_cost_scale = 0.005
        self.energy_cost_scale = 0.005 # was 0.05 #TODO 
        self.joints_at_limit_cost_scale = 0.1
        self.death_cost = -2.0
        self.termination_height = 1.1
        self.termination_height_up = 6 #

        # allocate buffers
        self.obs_buf, self.reward_buf, self.reset_buf, self.progress_buf = self.init_buffers()

        # acquire gym interface
        self.gym = gymapi.acquire_gym()
        #Create the simulation 
        self.sim = self.create_sim()

        initial_joints_file = 'assets/pose_default.yaml'
        with open(initial_joints_file) as f:
            initial_joints_dict = yaml.safe_load(f)

        self.initial_joints_dict = {k: np.deg2rad(v) for k, v in initial_joints_dict['joints'].items()}
        
        # Useless but was uesful for some time, I let it here, you also have other ways to get this information via the API 
        # careful this doesn't work if we don't take position default: position stretch only has 42 dofs.
        self.joints_92dof = self.initial_joints_dict.keys()
        # num_total_dof is not equal to num_dof because it contains all the dofs even the ones you can't moove
        self.num_total_dof = len(self.joints_92dof) #92


        self.create_ground()
        
        self.fly_asset, self.num_dof, self.num_rigid_body, self.dof_names = self.load_asset()

        print("self.num_dof", self.num_dof)
        # initialise envs 
        self.envs, self.actors, self.pose, self.dof_props = self.create_envs()

        self.dof_limits_lower, self.dof_limits_upper = self.get_limits_dofs()

        self.action_indexes, self.action_indexes_one = self.get_action_indexes()

        self.initial_dofs, self.initial_dofs_one = self.get_initial_dof_pos()

        self.dof_states, self.root_tensor, self.force_tensor = self.get_states_tensor()
        self.dof_pos = self.dof_states.view(self.args.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_states.view(self.args.num_envs, self.num_dof, 2)[..., 1]
        
        self.index_abdomen_sim, self.index_legs_tip = self.get_abdomen_leg_indices()


        # the root position (in our case the torso) is described by 13 floats: 
        # 3 floats for position, 4 floats for quaternion, 3 floats for linear velocity, and 3 floats for angular velocity.
        self.root_positions = self.root_tensor.view(self.args.num_envs, 13)[:, 0:3] 
        self.root_orientations = self.root_tensor.view(self.args.num_envs, 13)[:, 3:7]
        self.root_linvels = self.root_tensor.view(self.args.num_envs, 13)[:, 7:10]
        self.root_angvels = self.root_tensor.view(self.args.num_envs, 13)[:, 10:13]
        
        # some of those are not used anymore but I let them in, they can always be useful
        self.origin_root_tensor = self.create_origin_root_tensor()
        self.origin_position = self.origin_root_tensor.view(self.args.num_envs, 13)[:, 0:3] 
        self.origin_root_tensor_one = self.origin_root_tensor[0]
        self.origin_orientations = self.origin_root_tensor.view(self.args.num_envs, 13)[:, 3:7]

        # generate viewer for visualisation
        self.viewer = self.create_viewer()

        ## Setup the camera for making videos ##
        cameras_initial_pos = gymapi.Vec3(30, 0.0, 10)
        if self.args.record:
            self.record_dir_name = self.args.record_dir_name
            self.time_steps_per_recorded_frame = self.args.time_steps_per_recorded_frame
            self.camera, self.record_root_dir, self.record_command = self.set_up_recording(cameras_initial_pos, self.pose)
        
        ## Initialise other values for the reward and the observation buffer ##

        # potential is the distance from the target scaled by dt
        self.potentials = to_torch([-1000./self.dt], device=self.args.sim_device).repeat(self.args.num_envs)
        # previous potential, needed to calculate the gain or loss of potential between two steps
        self.prev_potentials = self.potentials.clone() 
        # the vector that is normal to the plane
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.args.sim_device).repeat((self.args.num_envs, 1))
        # the vector that gives us the direction the fly should be heading, claculated upon the target position  
        self.heading_vec = to_torch([1, 0, 0], device=self.args.sim_device).repeat((self.args.num_envs, 1))
        # the starting rotation of the flies, similar to self.origin_orientations
        self.inv_start_rot = quat_conjugate(self.origin_orientation).repeat((self.args.num_envs, 1))
        # basis vector of the 3d space
        self.basis_vec0 = self.heading_vec.clone() 
        self.basis_vec1 = self.up_vec.clone() 
        # the target location for each actor
        self.targets = to_torch([1000, 0, 0], device=self.args.sim_device).repeat((self.args.num_envs, 1)) #was [1000, 0, 0]
        self.target_dirs = to_torch([1, 0, 0], device=self.args.sim_device).repeat((self.args.num_envs, 1)) #was [1, 0, 0]

        
        # step simulation to initialise tensor buffers, .clone() can be used on them now
        self.gym.prepare_sim(self.sim)
        
        print("velocity", self.dof_props["velocity"][0])
        print("effort", self.dof_props["effort"][0])
        print("stiffness", self.dof_props["stiffness"][0])
        print("damping", self.dof_props["damping"][0])


    def create_sim(self):
        # configure sim (gravity is pointing down)
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81*1000) #should be *1000
        sim_params.dt = self.dt
        # Number of physic steps in a step 
        sim_params.substeps = 15
        sim_params.use_gpu_pipeline = True

        # set simulation parameters (we use PhysX engine by default, these parameters are from the example file)
        ### I should maybe change these params, see doc here file:///Users/petimo/Desktop/IsaacGym_Preview_4_Package/isaacgym/docs/api/python/struct_py.html#isaacgym.gymapi.PhysXParams
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.rest_offset = 0.001
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.use_gpu = True

        sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)

        return sim
    
    def init_buffers(self):
        # Observation given to the NN [num_environement, num_observations]
        obs_buf = torch.zeros((self.args.num_envs, self.num_obs), device=self.args.sim_device) 
        # Reward given to the NN
        reward_buf = torch.zeros(self.args.num_envs, device=self.args.sim_device)
        # Reset of the environements, if the i value is set to 1 the ith Actor is reset
        reset_buf = torch.ones(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)
        # How many steps have been made since the last reset 
        progress_buf = torch.zeros(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)

        return obs_buf, reward_buf, reset_buf, progress_buf
        
    
    def create_ground(self):
        # Add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = 0.0
        plane_params.normal = gymapi.Vec3(0, 0, 1) # z-Up
        self.gym.add_ground(self.sim, plane_params)

    def load_asset(self):
         # add cartpole asset
        asset_root = 'assets'
        asset_file = 'nmf_no_limits_limited_Dofs.urdf'
        asset_options = gymapi.AssetOptions() #get default options
        asset_options.fix_base_link = False
        fly_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # GIVES ONLY THE DOFS WHICH YOU CAN HAVE AN ACTION ON !!! so 42
        num_dof = self.gym.get_asset_dof_count(fly_asset) #18 
        num_rigid_body = self.gym.get_asset_rigid_body_count(fly_asset) #93
        dof_names = self.gym.get_asset_dof_names(fly_asset)

        return fly_asset, num_dof, num_rigid_body, dof_names


    def create_envs(self):
        # define environment space (for visualisation)
        spacing = 7
        lower = gymapi.Vec3(-spacing, 0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(np.sqrt(self.args.num_envs))

        # Define fly position
        pose = gymapi.Transform()
        pose.p.z = self.starting_height   # Generate the fly starting_height meters from the ground
        self.origin_orientation = torch.tensor([pose.r.x, pose.r.y, pose.r.z, pose.r.w], device=self.args.sim_device)

        # Define fly dof properties
        dof_props = self.gym.get_asset_dof_properties(self.fly_asset)

        # Those cannot be over certain values otherwise the position controler doesn't work anymore !! Those values depend on the gravity and other
        # unknown parameters. Be careful with it a do tests when changing it. Example: see if the limbs are reseted correctly when calling self.Reset()
        dof_props['driveMode'] = gymapi.DOF_MODE_POS
        dof_props['stiffness'].fill(70) 
        dof_props['damping'].fill(0.1)
        dof_props['velocity'].fill(1)
        dof_props["effort"].fill(30)

        # Generate environments
        envs = []
        actors = []
        print(f'Creating {self.args.num_envs} environments.')
        for i in range(self.args.num_envs):
            # create env
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Add fly here in each environment
            fly = self.gym.create_actor(env, self.fly_asset, pose, f'fly{i}', i, 1, 0)
            self.gym.set_actor_dof_properties(env, fly, dof_props)
            
            envs.append(env)
            actors.append(fly)
        
        return envs, actors, pose, dof_props
    

    def get_limits_dofs(self):
        """
            This functions gets the limits of the degrees of freedom. So that we sacale up the actions of the NN correctly.
            Those limits are also used in the reward function
        """

        dof_limits_lower = []
        dof_limits_upper = []
        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.actors[0])
        
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                dof_limits_lower.append(dof_prop['upper'][j])
                dof_limits_upper.append(dof_prop['lower'][j])
            else:
                dof_limits_lower.append(dof_prop['lower'][j])
                dof_limits_upper.append(dof_prop['upper'][j])
        
        dof_limits_lower = to_torch(dof_limits_lower, device=self.args.sim_device)
        dof_limits_upper = to_torch(dof_limits_upper, device=self.args.sim_device)

        return dof_limits_lower, dof_limits_upper


    def get_action_indexes(self):

        """
            This function creates a mask of all the dofs we want to have an effect on
            Find the indexes of the actions we want to modify, these indexes are relative to the simulation
            The indexes range from 0 to (num_dofs * num_envs)
            This function is useless if we use all the dofs contained in the asset ie. if self.names and self.dof_names is equal.
        """

        action_indexes = torch.full((self.num_act * self.args.num_envs, 1), 0, dtype=torch.long, device=self.args.sim_device)
        j = 0
        for i in range(self.args.num_envs):
            for name in self.names:
                action_indexes[j] = self.gym.find_actor_dof_index(self.envs[i], self.actors[i], name, gymapi.DOMAIN_SIM)
                j+=1

        action_indexes, ___ = torch.sort(action_indexes, 0) # It should have a size num_action * num_envs

        action_indexes_one = action_indexes[0:self.num_act] # Same thing but only for one actor
        
        return action_indexes, action_indexes_one

    def get_abdomen_leg_indices(self):
        """
            Getting the abdomen and the leg indicies so it can be use as a mask to find the good values in the force tensor
            Those values of the force tensor are then used to calculate if there was a collision or not
        """
        names_abdomen_parts = ["A1A2", "A3", "A4", "A5", "A6"]
        names_leg_tip = ["RFTarsus5", "LFTarsus5", "RMTarsus5", "LMTarsus5", "RHTarsus5", "LHTarsus5"]
        index_abdomen = []
        index_legs_tip = []
        for i in range(self.args.num_envs):
            for name in names_abdomen_parts:
                index = self.gym.find_actor_rigid_body_index(self.envs[i], self.actors[i], name, gymapi.DOMAIN_SIM)
                index_abdomen.append(index)
            for name in names_leg_tip:
                index = self.gym.find_actor_rigid_body_index(self.envs[i], self.actors[i], name, gymapi.DOMAIN_SIM)
                index_legs_tip.append(index)

        index_abdomen.sort()
        index_legs_tip.sort()
        index_abdomen_sim = to_torch(index_abdomen, device=self.args.sim_device, dtype=torch.long)
        index_legs_tip = to_torch(index_legs_tip, device=self.args.sim_device, dtype=torch.long)

        return index_abdomen_sim, index_legs_tip

    def get_initial_dof_pos(self):
        
        """
            Getting initial joint angles for each dofs of the fly of all the flies, with no velocity set for all dofs, 
            We will use it later to reset the dofs when we reset the fly 
        """
        
        # Initial_dofs should be of size self.num_dof * self.args.num_envs
        initial_dofs = torch.full((self.num_dof*self.args.num_envs, 2), 0, dtype=torch.float32, device=self.args.sim_device) 
        
        for i in range(self.args.num_envs):
            for joint_name in self.dof_names:
                joint_index = self.gym.find_actor_dof_index(self.envs[i], self.actors[i], joint_name, gymapi.DOMAIN_SIM)
                initial_dofs[joint_index, 0] = self.initial_joints_dict.get(joint_name, 0) # defaults to 0 for unspecified joints
        
        #Initial_dof for just one fly, useful in the reset 
        initial_dofs_one = initial_dofs[:self.num_dof]   

        return initial_dofs, initial_dofs_one

    def create_origin_root_tensor(self):
        """
            Create the tensor for the reset of the position and the orientation of the fly 
            self.root_tensor.clone() will only work when the simulation is started since the root_tensor is not initialised yet 
            but this function still is a bit useless and could be replaced by that.
        """
        origin_root_tensor = torch.zeros((self.args.num_envs, 13), device=self.args.sim_device) 
        origin_root_tensor[:,2] = self.starting_height
        origin_root_tensor[:,6] = 1 #the last quaterion should be one

        return origin_root_tensor

    def create_viewer(self):
        # create viewer for debugging (looking at the center of environment)
        
        self.enable_viewer_sync = True
        viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.args.headless == False:
            # subscribe to keyboard shortcuts
            viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

            self.gym.subscribe_viewer_keyboard_event(
                viewer, gymapi.KEY_ESCAPE, "End_simulation")
            self.gym.subscribe_viewer_keyboard_event(
                viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(
                viewer, gymapi.KEY_R, "Reset_everyone")
            self.gym.subscribe_viewer_keyboard_event(
                viewer, gymapi.KEY_E, "End_simulation")
            self.gym.subscribe_viewer_keyboard_event(
                viewer, gymapi.KEY_P, "Print")     
            
        cam_pos = gymapi.Vec3(30, 0.0, 10)
        cam_target = gymapi.Vec3(-1, 0, 0)
        self.gym.viewer_camera_look_at(viewer, self.envs[self.args.num_envs // 2], cam_pos, cam_target)

        return viewer

    def get_states_tensor(self):
        # Get dof state tensor of All the flys
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        
        # Acquire root state tensor descriptor
        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)

        # Acquire force state tensor descritor
        _force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        # Wraps it in a PyTorch Tensor and create convenient views
        root_tensor = gymtorch.wrap_tensor(_root_tensor)
        force_tensor = gymtorch.wrap_tensor(_force_tensor)
        dof_states = gymtorch.wrap_tensor(_dof_states)

        dof_states = dof_states.view(self.args.num_envs, self.num_dof*2)

        return dof_states, root_tensor, force_tensor

    def get_obs(self):
        """
            Refreshes the different tensors and updates the observation buffer and other accordingly 
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim) 
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = compute_fly_observations(
            self.obs_buf, self.root_tensor, self.targets, self.potentials,
            self.inv_start_rot, self.dof_pos, self.dof_vel,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale, 
            self.actions.view(self.args.num_envs, -1), self.dt,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx, self.action_indexes_one,
            self.force_tensor, self.index_legs_tip, self.args.num_envs)

    def get_reward(self):
        # retrieve environment observations from buffer
        nb_of_sim_step = self.progress_buf
        nb_of_sim_step[nb_of_sim_step==0] = 1

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
            self.num_act, 
            self.force_tensor,
            self.index_abdomen_sim,
            self.index_legs_tip
        )


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

        # Reset desired environments torso position and orientation 
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_tensor),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # Reset desired environments degrees of freedom 
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_states),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        to_target = self.targets[env_ids] - self.origin_root_tensor[env_ids, 0:3]
        to_target[:, 2] = 0.0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        # clear up desired buffer states
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
    
        #YOU CANNOT GET_OBS HERE 
        return True 

    def simulate(self):
        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def render(self):
        # Update viewer
        if self.viewer:
            # Check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # Check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif  evt.action == "Reset_everyone" and evt.value > 0:
                    print("We reset everyone !")
                    self.reset_buf = torch.ones(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)
                elif evt.action == "End_simulation" and evt.value > 0:
                    print("We end the simulation")
                    self.end = True
                elif evt.action == "Print" and evt.value > 0:
                    # This can be changed at will 

                    heading_weight_tensor = torch.ones_like(self.obs_buf[:, 11]) * self.heading_weight
                    heading_reward = torch.where(self.obs_buf[:, 11] > 0.8, heading_weight_tensor, self.heading_weight * self.obs_buf[:, 11] / 0.8)
                    print("heading_reward", heading_reward[:10])
                    alive_reward = torch.ones_like(self.potentials) * 0.5
                    print("alive_reward", alive_reward[:10])
                    up_reward = torch.zeros_like(heading_reward)
                    up_reward = torch.where(self.obs_buf[:, 0] > 1.4, up_reward + self.up_weight, up_reward)
                    #print("height", self.obs_buf[:, 0])
                    print("up_reward", up_reward[:10])
                    
                    orient_reward = torch.zeros_like(up_reward)
                    orient_reward = torch.where(torch.square(self.root_orientations[:, 2])  + torch.square(self.root_orientations[:, 3]) > 0.92, orient_reward + self.up_weight, orient_reward)
                    #print("orient_reward", orient_reward)
                    # energy penalty for movement
                    actions_cost = torch.sum(self.actions.view(self.args.num_envs, -1) ** 2, dim=-1)
                    #print("actions_cost", actions_cost)

                    start = (12+2*self.num_act)
                    electricity_cost = torch.sum(torch.abs(self.actions.view(self.args.num_envs, -1) - self.obs_buf[:, start:(start + self.num_act)]), dim=-1)
                    print("electricity_cost", electricity_cost[:10])
                    #Be at the the extremities costs  
                    #print(obs_buf[:, 96:114].size(), upper_limit_of_actions[action_indicies_one].squeeze(-1).repeat((num_env, 1)).size(), upper_limit_of_actions[action_indicies_one].squeeze(-1).repeat((10, 1))) 
                    dof_at_limit_cost = torch.sum(self.obs_buf[:, start:(start + self.num_act)] > self.dof_limits_upper[self.action_indexes_one].squeeze(-1).repeat((self.args.num_envs, 1)) * 0.9, dim=-1)
                    dof_at_limit_cost += torch.sum(self.obs_buf[:, start:(start + self.num_act)] < self.dof_limits_lower[self.action_indexes_one].squeeze(-1).repeat((self.args.num_envs, 1)) * 0.9, dim=-1)
                    print("dof_at_limit_cost", dof_at_limit_cost[:10] * self.joints_at_limit_cost_scale)
                    progress_reward = self.potentials - self.prev_potentials
                    #print("progress_reward", progress_reward)
                    #print("Position", self.root_positions)
                    print("reward up*orient", up_reward[:10] * orient_reward[:10])
                    #print(self.dof_pos[:, self.gym.find_actor_dof_index(self.envs[0], self.actors[0],  "joint_RHCoxa_roll", gymapi.DOMAIN_ENV)])
                    #print(torch.sum(torch.sum(self.force_tensor[self.index_abdomen_sim, :], dim=1).view(self.args.num_envs, -1), dim=1) )
                    
                    #print(scale(torch.ones(18, device=self.args.sim_device), self.dof_limits_lower, self.dof_limits_upper))
                    #print(self.dof_limits_upper)
                    #print(scale(torch.ones(18, device=self.args.sim_device), self.dof_limits_lower, self.dof_limits_upper) <= self.dof_limits_upper+0.01)
                    #print(scale(torch.ones(18, device=self.args.sim_device), self.dof_limits_lower, self.dof_limits_upper) >= self.dof_limits_lower-0.01)
                    leg_reward = torch.sum((torch.sum(self.force_tensor[self.index_legs_tip, :], dim=1).view(self.args.num_envs, -1) > 0).long(), dim=1) * 0.1
                    #print((torch.sum(self.force_tensor[self.index_legs_tip, :], dim=1).view(self.args.num_envs, -1) > 0).long())
                    print(leg_reward[:10])
                    #print(self.actions[self.action_indexes_one].view(len(self.action_indexes_one)))

            # fetch results
            if self.args.sim_device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync or self.args.record:
                self.gym.step_graphics(self.sim)
                if self.enable_viewer_sync:
                    self.gym.draw_viewer(self.viewer, self.sim, True)
                    # Wait for dt to elapse in real time.
                    # This synchronizes the physics simulation with the rendering rate.
                    self.gym.sync_frame_time(self.sim)

            else:
                self.gym.poll_viewer_events(self.viewer)


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
        if not self.args.record:
            return

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
        if not self.args.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        
    
    def step(self, actions):
        # Clone the initial position of the fly
        actions_tensor = torch.clone(self.initial_dofs).detach()
        
        # Sclaes the actions from [-1 1] to [limit_lowe limit_upper] for every action
        actions_scaled = scale(actions, self.dof_limits_lower[self.action_indexes_one].view(1, -1).repeat(self.args.num_envs, 1)
                                        , self.dof_limits_upper[self.action_indexes_one].view(1, -1).repeat(self.args.num_envs, 1))

        # Puts the result back in a convinent form   
        actions_scaled = actions_scaled.view(self.args.num_envs * self.num_act, 1)
        
        # Updates the actions for later 
        self.actions = actions_scaled.clone()
        
        #TODO remouve this check
        if torch.sum(actions_scaled[self.action_indexes_one].view(len(self.action_indexes_one)) > self.dof_limits_upper[self.action_indexes_one].squeeze(-1) + 0.1) and self.print_once:
            self.print_once = False
            print("Action asked to go over !!!")
            #print(actions[0, self.action_indexes_one])
            #print(actions_scaled[self.action_indexes_one].view(len(self.action_indexes_one)))
            #print(self.dof_limits_upper[self.action_indexes_one].squeeze(-1))
            print(actions_scaled[self.action_indexes_one].view(len(self.action_indexes_one)) > self.dof_limits_upper[self.action_indexes_one].squeeze(-1))
        
        # Replaces, using the mask, the values of the actions
        actions_tensor[..., 0][self.action_indexes] = actions_scaled
        
        # We only want position, no velocity. The .contiguous() makes that the tensor is contiguous in memory, required for setting the dof positions
        actions_pos_only = actions_tensor[...,0].contiguous()
        
        # Wraps the torch tensor in an IsaacGym tensor
        positions = gymtorch.unwrap_tensor(actions_pos_only)
        
        # Sets the target position dicted by the actions 
        self.gym.set_dof_position_target_tensor(self.sim, positions)

        # Reset the environements 
        self.reset()

        # Simulate: !! You need a simulate between a 
        self.simulate()
        
        # Renders 
        if not self.args.headless :
            self.render()
        
        # Update camera and Save file
        if self.args.record and (self.render_count % self.time_steps_per_recorded_frame == 0):
            self.record_frame(self.render_count)

        self.render_count+=1

        # Reset the observations
        self.get_obs()        

        self.progress_buf += 1

        # Calculate the reward
        self.get_reward()


# define reward function using JIT
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
    num_actions,
    force_tensor,
    index_abdomen_sim,
    index_legs_tip
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, float, float, float, Tensor, Tensor, Tensor, int, Tensor, int, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    # reward from direction headed
    heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    heading_reward = torch.where(obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8)

    # reward for the torso beeing above a certain height 
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(obs_buf[:, 0] > 1.4, up_reward + up_weight, up_reward)
    up_reward = torch.where(obs_buf[:, 0] < 2.1, up_reward - up_weight, up_reward)

    # reward for beeing parallel to the ground, same as beiing parallel from the base orientation
    # put up_weight there just because I think orientation should be as important as beeing up
    orient_reward = torch.zeros_like(up_reward)

    # works because base orientation is 0i + 0j + 0k + 1 and to be on the same plane reel^2 + k^2 should be = 1
    orient_reward = torch.where(torch.square(orientation[:, 2])  + torch.square(orientation[:, 3]) > 0.98, orient_reward + up_weight, orient_reward)

    start = (12+2*num_actions)
    # energy penalty for movement, the more the movements differ from the previous ones, grater is the cost 
    actions_cost = torch.sum(actions ** 2, dim=-1) #This reward is dumb as I am not passing forces between 0 and 1 but position which sometimes should be high so not penalised 
    electricity_cost = torch.sum(torch.abs(actions - obs_buf[:, start:(start + num_actions)]), dim=-1)
   
    # dofs at the the extremities costs 
    dof_at_limit_cost = torch.sum(obs_buf[:, start:(start + num_actions)] > upper_limit_of_actions[action_indicies_one].squeeze(-1).repeat((num_env, 1)) * 0.9, dim=-1)
    dof_at_limit_cost += torch.sum(obs_buf[:, start:(start + num_actions)] < lower_limit_of_actions[action_indicies_one].squeeze(-1).repeat((num_env, 1)) * 0.9, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * 0.5
    progress_reward = potentials - prev_potentials

    # leg on the ground reward 
    leg_ground_reward = torch.sum((torch.sum(force_tensor[index_legs_tip, :], dim=1).view(num_env, -1) > 0).long(), dim=1) * 0.1 ##TODO 0.1 maybe too high 


    #total_reward = progress_reward * 2 + alive_reward + up_reward * orient_reward + heading_reward - \
     #   actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost * joints_at_limit_cost_scale #+ leg_ground_reward 
    
    total_reward = alive_reward + up_reward * orient_reward - energy_cost_scale * electricity_cost - dof_at_limit_cost * joints_at_limit_cost_scale + leg_ground_reward 

    # adjust reward for fallen agents, see below for more info. This is double. It could me made better maybe. 
    total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward)
    total_reward = torch.where(obs_buf[:, 0] > termination_height_up, torch.ones_like(total_reward) * death_cost, total_reward)
    total_reward = torch.where(torch.square(orientation[:, 2])  + torch.square(orientation[:, 3]) < 0.5, torch.ones_like(total_reward) * death_cost, total_reward)
    total_reward = torch.where(torch.sum(torch.sum(force_tensor[index_abdomen_sim, :], dim=1).view(num_env, -1), dim=1) > 0, torch.ones_like(total_reward) * death_cost, total_reward)

    # Reset if higher or lower than a certain height 
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(obs_buf[:, 0] > termination_height_up, torch.ones_like(reset_buf), reset)
    # Reset if the fly was alive for more steps than max_episode_length
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    # Reset if it is too tilted in any direction 
    reset = torch.where(torch.square(orientation[:, 2])  + torch.square(orientation[:, 3]) < 0.5, torch.ones_like(reset_buf), reset)
    # Reset if the abdomen touches the ground 
    reset = torch.where(torch.sum(torch.sum(force_tensor[index_abdomen_sim, :], dim=1).view(num_env, -1), dim=1) > 0, torch.ones_like(reset_buf), reset)
    
    return total_reward, reset


@torch.jit.script 
def compute_fly_observations(obs_buf, root_states, targets, potentials,
                             inv_start_rot, dof_pos, dof_vel,
                             dof_limits_lower, dof_limits_upper, dof_vel_scale, actions, dt,
                             basis_vec0, basis_vec1, up_axis_idx, action_idx, force_tensor, index_legs_tip, num_envs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, float, Tensor, Tensor, int, Tensor, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

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

    touching = (torch.sum(force_tensor[index_legs_tip, :], dim=1).view(num_envs, -1) > 0).long()

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_action(18), num_action(18), num_actions(18), 1, 6 = 73
    obs = torch.cat((torso_position[:, up_axis_idx].view(-1, 1), vel_loc, angvel_loc,
                     yaw.unsqueeze(-1), roll.unsqueeze(-1), angle_to_target.unsqueeze(-1),
                     up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1), dof_pos_scaled[:, action_idx].squeeze(-1),
                     dof_vel[:, action_idx].squeeze(-1) * dof_vel_scale, actions, pitch.unsqueeze(-1), touching), dim=-1)

    return obs, potentials, prev_potentials_new, up_vec, heading_vec


    
