# The Project 

This repository is an implementation of the NeuroMechFly model in an RL environement using NVIDIA's physics simulation environment[Isaac Gym](https://developer.nvidia.com/isaac-gym). This implementation is a fork of Shikun Liu's [minimal-isaac-gym repository](https://github.com/lorenmt/minimal-isaac-gym) adapted and improved upon coupled with the work of [Ramdya Lab](https://www.epfl.ch/labs/ramdya-lab/) at epfl which released a very accurate model of the drosofilia Melanogaster called [NeuroMechFly](https://www.nature.com/articles/s41592-022-01466-7) 


**Disclaimer**:This implementation is still not perfectly accurate regarding the coherance of the units used and so the results found using it a promising but not perfect. I highly recomend to read my project repport, especially the part 4.5 talking about the hyperparameters where I explain a big parts of the problems I encountered and how could one resolve them. 

**Note**: The current implementation supports only one RL algorithms, *PPO continuous*. The interfacing between the environement and the NN  is really simple as I am not using any third-party RL frameworks like the examples of IsaacGymEnvs do. One could easely change it at will and is encourage to do so as I think my NN is not strong enough to learn such a complex environement. But again I recommend to read my project repport to have more info on the subject. 

## Usage

Simply run `python trainer.py`

#### Some arguments can be added:

`--sim_device`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type= str&nbsp;&nbsp;&nbsp;&nbsp;default="cuda:0"    
Decide on which device you want to simulate your environement and the NN, the other possible values are `["cpu", "cuda:1", "cuda:2", ...]`

`--compute_device_id`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type= int&nbsp;&nbsp;&nbsp;&nbsp;default= 0,        
Index of CUDA-enabled GPU to be used for simulation.
      
`--graphics_device_id`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type= int&nbsp;&nbsp;&nbsp;&nbsp;default= 0
<Br>Index of GPU to be used for rendering    

`--num_envs`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type= int&nbsp;&nbsp;&nbsp;&nbsp;default= 1000        
Number of environements in the simulation 

`--headless`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type= bool&nbsp;&nbsp;&nbsp;&nbsp;default= False       
Choose to not display a viewer by setting True.  

`--save_path`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type= str&nbsp;&nbsp;&nbsp;&nbsp;default= None        
If not None, the weights of the NN will be saved to this path. This path should be **Absolute**

`--save_freq`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type= int&nbsp;&nbsp;&nbsp;&nbsp;default= 100         
Frequency at which it will save, normally it should be the number of optimisation steps between each save but it doesn't for some reson I know but don't have the time to change.  

`--load_path`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type= str    default= None   
If not None, the weights of the NN will be loaded from this path at the start of the simulation. This path is **Absolute**

`--record_dir_name`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type= str&nbsp;&nbsp;&nbsp;&nbsp;default= None
<Br>If not None, screeshots of the simulation will be taken and placed in an floder `./recording/record_dir_name`. The base name "recording" can be changed in the code.

`--time_steps_per_recorded_frame`&nbsp;&nbsp;&nbsp;&nbsp;type= int&nbsp;&nbsp;&nbsp;&nbsp;default= 2
<Br>How many time steps between each screenshot 

`--testing`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type= bool&nbsp;&nbsp;&nbsp;&nbsp;default= False
<Br>If set to True, the variance of the action will be set to a small value and no training will take place. Use this to test your NN by loading the weights in with the command `--load_path` described before.


