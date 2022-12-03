from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import sys
import torch
import yaml
import torchgeometry as tgm

class Fly:
    def __init__(self, args):
        self.args = args
        self.end = False

        # configure sim (gravity is pointing down)
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81*1000) #should be *1000
        sim_params.dt = 1 / 1000.
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
        self.num_obs = 42*2  # 42 dofs * (pos, velocity)
        self.num_act = 18 #(3 DoFs * 6 legs)
        #ThC pitch for the front legs (joint_RFCoxa), ThC roll (joint_LMCoxa_roll) for the middle and hind legs, and CTr pitch (joint_RFFemur) and FTi pitch (joint_LFTibia) for all leg
        self.max_episode_length = 500  # maximum episode length
        self.render_count = 0
        self.names = ["joint_LHCoxa_roll", "joint_RHCoxa_roll", "joint_LHFemur", "joint_RHFemur", "joint_LHTibia", "joint_RHTibia",
                    "joint_LMCoxa_roll", "joint_RMCoxa_roll", "joint_LMFemur", "joint_RMFemur", "joint_LMTibia", "joint_RMTibia",
                     "joint_LFCoxa", "joint_RFCoxa", "joint_LFFemur", "joint_RFFemur", "joint_LFTibia", "joint_RFTibia"]
        
        self.plane_static_friction = 1.0
        self.plane_dynamic_friction = 1.0
        self.restitution = 0.0 #not used 


        # allocate buffers
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
        self.envs, self.num_dof, self.dof_indexes, self.initial_dofs, self.initial_dofs_one  = self.create_envs()
        self.dof_states, self.root_tensor = self.get_states_tensor()
        self.dof_pos = self.dof_states.view(self.args.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_states.view(self.args.num_envs, self.num_dof, 2)[..., 1]

        #It is always 13: 3 floats for position, 4 floats for quaternion, 3 floats for linear velocity, and 3 floats for angular velocity.
        self.old_position = torch.zeros((self.args.num_envs, 3), device=self.args.sim_device)
        self.root_positions = self.root_tensor.view(self.args.num_envs, 13)[:, 0:3] 
        self.root_orientations = self.root_tensor.view(self.args.num_envs, 13)[:, 3:7] #THOSE ORIENTATION ARE NOT ADDING UP TO ONE !!
        self.root_linvels = self.root_tensor.view(self.args.num_envs, 13)[:, 7:10]
        self.root_angvels = self.root_tensor.view(self.args.num_envs, 13)[:, 10:13]
        
        self.origin_pos = torch.zeros((self.args.num_envs, 3), device=self.args.sim_device)
        self.origin_root_tensor = torch.zeros((self.args.num_envs, 13), device=self.args.sim_device) #self.root_tensor.clone() ne marchera pas 
        #psk on le fait avant de lancer la simulation
        self.origin_root_tensor[:,2] = 3
        self.origin_root_tensor[:,6] = 1 #the last quaterion should be one... idk why 
        print("ORIGINAL z", self.origin_root_tensor)
        self.origin_root_tensor_one = self.origin_root_tensor[0]
        #print(self.origin_root_tensor.size(), self.origin_root_tensor_one.size())

        # generate viewer for visualisation
        self.viewer = self.create_viewer()

        
        # step simulation to initialise tensor buffers
        self.gym.prepare_sim(self.sim)
        
        #self.reset() #Ne Peux pas être là !
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
        
    def create_envs(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        # TODO: define
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
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
        pose.p.z = 2.5   # generate the fly 3m from the ground
        #pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi / 8) # No rotation needed 


        # define fly dof properties
        dof_props = self.gym.get_asset_dof_properties(fly_asset)
        dof_props['driveMode'] = gymapi.DOF_MODE_POS
        dof_props['stiffness'].fill(10000)
        dof_props['damping'].fill(50)

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
        #try reseting the actor now

        # Find the indexes we want to modify, these indexes are relative to the sim 0 and 42*num_envs
        # It should have a size num_action*num_envs
        dof_indexes = torch.full((self.num_act * self.args.num_envs, 1), 0, dtype=torch.long, device=self.args.sim_device)
        j = 0
        for i in range(self.args.num_envs):
            for name in self.names:
                dof_indexes[j] = self.gym.find_actor_dof_index(envs[i], actors[i], name, gymapi.DOMAIN_SIM)
                j+=1
        #print("Dof_indexes: ", dof_indexes.size(), "Should be: ",  self.num_act*self.args.num_envs)

        dof_indexes, indexosef = torch.sort(dof_indexes, 0)

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
        
        return envs, num_dof, dof_indexes, initial_dofs, initial_dofs_one

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

        cam_pos = gymapi.Vec3(30, 0.0, 10)
        cam_target = gymapi.Vec3(-1, 0, 0)
        self.gym.viewer_camera_look_at(viewer, self.envs[self.args.num_envs // 2], cam_pos, cam_target)

        return viewer

    def get_states_tensor(self):
        # get dof state tensor of All the flys
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        dof_states = dof_states.view(self.args.num_envs, self.num_obs)

        # acquire root state tensor descriptor
        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)

        # wrap it in a PyTorch Tensor and create convenient views
        root_tensor = gymtorch.wrap_tensor(_root_tensor)
        #print("Roooot: ", root_tensor.size())

        return dof_states, root_tensor

    def get_obs(self, env_ids=None):
        # get state observation from each environment id
        if env_ids is None:
            env_ids = torch.arange(self.args.num_envs, device=self.args.sim_device)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim) 
        
        #This is not used 
        self.obs_buf[env_ids] = self.dof_states[env_ids]

        #self.i+=1 
        #print("bababababbababas", self.i ,self.dof_states.size(), self.dof_states)
        #self.i+=1 
        #print("hahahah", self.i ,self.root_tensor.size(), self.root_tensor)

        # Could do smth like that if needed
        # self.obs_buf_root[env_ids] = self.root_tensor[env_ids]

    def get_reward(self):
        # retrieve environment observations from buffer
        nb_of_sim_step = self.progress_buf
        nb_of_sim_step[nb_of_sim_step==0] = 1

        #test_quat = tgm.quaternion_to_angle_axis(torch.tensor([1, 0, 0 ,0],  device=self.args.sim_device))
        #test_quat2 = tgm.quaternion_to_angle_axis(torch.tensor([1, 0, 0 ,0], device=self.args.sim_device))
        #test_gymap = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0),0).to_euler_zyx()
        #print(test_gymap)
        #rot = torch.tensor([np.pi, 0.0, 0.0], device=self.args.sim_device)
        #rot2 = torch.tensor([0.0, np.pi, 0.0], device=self.args.sim_device)
        #rotations_not_quat = tgm.quaternion_to_angle_axis(self.root_orientations)
        #print(self.root_orientations)
        #print(rotations_not_quat)
        #print(rotations_not_quat[1] > np.pi/2, rotations_not_quat[1] < -np.pi/2)
        #print(rotations_not_quat[0] > np.pi/2,  rotations_not_quat[0] < -np.pi/2)

        self.reward_buf[:], self.reset_buf[:], self.timer_down[:] = compute_fly_reward(nb_of_sim_step, self.timer_down, self.origin_pos, 
            self.root_positions, self.reset_buf, self.max_episode_length, 250)
        
        print(self.timer_down)

        self.old_position = self.root_positions
    
    ## Can be made better  
    def reset(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
       
        if len(env_ids) == 0:
            return False

        print("resting, everything for now I guess because..")
        # initial dof [{pos1, vel1},{pos2, vel2},....,{posn,veln}]
        # Can be done differently, we could not pass by these dof_pos, dof_vel, and do smth like self.dof_state[env_ids, :] = smth I'll see later If I have the courage to change that
        self.dof_pos[env_ids, :] = self.initial_dofs_one[..., 0]
        self.dof_vel[env_ids, :] = self.initial_dofs_one[..., 1]
        env_ids_int32 = env_ids.to(dtype=torch.int32)


        self.root_tensor[env_ids, :] = self.origin_root_tensor[env_ids, :] 
        #print("Root tensor before setting", self.origin_root_tensor.size(), self.origin_root_tensor) 

        # Reset desired environments
        #Try destoying the actor and recreate one 
        #Try position controller. <- maybe the better option 
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_tensor),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        setted1 = self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_states),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        #print(setted1, setted)

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

    def exit(self):
        # close the simulator in a graceful way
        if not self.args.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
    
    def step(self, actions):
        # apply action
        # Reçois un tenseur de la taille num_actions * num_envs du coup il faut le mettre correctement pour que ça fasse num_obs * num_envs
        ##actions_tensor = torch.zeros(self.args.num_envs * self.num_dof, device=self.args.sim_device)
        ##actions_tensor[::self.num_dof] = actions.squeeze(-1) * self.max_push_effort
        actions_tensor = torch.clone(self.initial_dofs).detach()
        actions = actions.view(self.args.num_envs * self.num_act, 1)
        #print("actions: ", actions_tensor)
        # print(self.dof_indexes.size())
        # print(actions_tensor.size())
        # print(actions_tensor[..., 0].size())
        # print(actions_tensor[..., 0][self.dof_indexes].size())
        # print(actions.size())
        #Replaces in the mask the values of the actions
        actions_tensor[..., 0][self.dof_indexes] = actions
        
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
            self.render_count+=1
            if self.render_count % 1 == 0:
                self.render()
        
        #You cannot get obs if you reset 
        self.get_obs()        
        self.progress_buf += 1
        if (self.progress_buf[0] > self.max_episode_length):
            print("Should be reseting soon")
        self.get_reward()


# define reward function using JIT
# Aucune idée si ça marche  
@torch.jit.script
def compute_fly_reward(time, timer_down, origin_pos, pos, reset_buf, max_episode_length, max_time_down):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float) -> Tuple[Tensor, Tensor, Tensor]
    

    #We only want progress in the X axis 
    dist_x = pos[...,0] - origin_pos[...,0]
    
    
    reward = dist_x / time 

    timer_down[pos[..., 2]< 1] += 1
    
    # adjust reward for reset agents
    reward = torch.where(time > max_episode_length, torch.ones_like(reward) * -2.0, reward)

    reset = torch.where(time > max_episode_length, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(timer_down > max_time_down, torch.ones_like(reset), reset)
    #reset = torch.where(rotation < rot or rotation < rot2 or rotation < rot3 , torch.ones_like(reset), reset)
    
    return reward, reset, timer_down
