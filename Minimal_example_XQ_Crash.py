from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *

import sys


class Fly:
    def __init__(self):
        self.end = False
        self.dt = 1 / 1000.
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        # configure sim (gravity is pointing down)
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81*1000) #should be *1000
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
        self.num_obs = 114  # See compute_fly_observations
        self.num_act = 18 #(3 DoFs * 6 legs)
        self.starting_height = 2.2
        #ThC pitch for the front legs (joint_RFCoxa), ThC roll (joint_LMCoxa_roll) for the middle and hind legs, and CTr pitch (joint_RFFemur) and FTi pitch (joint_LFTibia) for all leg
        self.max_episode_length = 1000  # maximum episode length
        self.render_count = 0
        
        self.plane_static_friction = 1.0
        self.plane_dynamic_friction = 1.0

        #Constants for the reward function, taken from ant
        self.dof_vel_scale = 0.2
        self.heading_weight = 0.5
        self.up_weight = 0.1
        self.actions_cost_scale = 0.005
        self.energy_cost_scale = 0.05
        self.joints_at_limit_cost_scale = 0.1
        self.death_cost = -2.0
        self.termination_height = 0.8 #PEUT être TROP petit !!
        self.headless = False
        # acquire gym interface
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

        # generate viewer for visualisation
        self.viewer = self.create_viewer()
        
        # step simulation to initialise tensor buffers
        self.gym.prepare_sim(self.sim)

    def create_viewer(self):
        # create viewer for debugging (looking at the center of environment)
        
        self.enable_viewer_sync = True
        viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
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

        return viewer

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

            # step graphics
            print("Pre-fucked")
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                print("Post-fucked")
                self.gym.draw_viewer(self.viewer, self.sim, True)
                print("Post-fucked2")
                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)
                print("Post-fucked3")

            else:
                self.gym.poll_viewer_events(self.viewer)
       

    def exit(self):
        # close the simulator in a graceful way
        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
    
    def step(self):
        # simulate and render
        self.simulate()

        if not self.headless :
            self.render_count+=1
            if self.render_count % 1 == 0:
                self.render()

fly = Fly()
while True:
    fly.step()
