####################################
#      YOU MAY EDIT THIS FILE      #
# MOST OF YOUR CODE SHOULD GO HERE #
####################################

# Imports from external libraries
import numpy as np

# Imports from this project
import constants
import config


# The Robot class (which could be called "Agent") is the "brain" of the robot, and is used to decide what action to execute in the environment
class Robot:

    # Initialise a new robot
    def __init__(self):
        # The environment
        self.environment = None
        # The number of steps in the episode so far
        self.num_steps = 0
        # A list of visualisations which will be displayed on the right-side of the window
        self.visualisation_lines = []
        self.visualisation_circles = []

    # Reset the robot at the start of an episode
    def reset(self):
        self.num_steps = 0
        self.visualisation_lines = []
        self.visualisation_circles = []

    # Give the robot access to the environment
    def set_environment(self, environment):
        self.environment = environment

    # Function to get the next action in the plan
    def select_action(self, state):
        # Initially, we just return a random action, but you should implement a planning algorithm
        if self.num_steps == 0:
            self.cem_planning(state)
        # Check if the episode has finished
        action = self.planned_actions[self.num_steps]
        self.num_steps += 1
        episode_done = self.num_steps == len(self.planned_actions)

        return action, episode_done

    def cem_planning(self, state):
        sampled_actions = np.zeros([config.CEM_NUM_ITER, config.CEM_NUM_PATHS, config.CEM_EPISODE_LENGTH, 2], dtype=np.float32)
        sampled_paths = np.zeros([config.CEM_NUM_ITER, config.CEM_NUM_PATHS, config.CEM_EPISODE_LENGTH+1, 2], dtype=np.float32)
        path_distances = np.zeros([config.CEM_NUM_ITER, config.CEM_NUM_PATHS], dtype=np.float32) # algo developed such that the robot regularly reaches within a distance of 0.01 of goal state
        action_mean = np.zeros([config.CEM_NUM_ITER, config.CEM_EPISODE_LENGTH, 2], dtype=np.float32)
        action_std = np.zeros([config.CEM_NUM_ITER, config.CEM_EPISODE_LENGTH, 2], dtype=np.float32)

        # Loop over the CEM iterations
        for iter_num in range(config.CEM_NUM_ITER):
            # Loop over the number of paths to sample
            for path_num in range(config.CEM_NUM_PATHS):
                # Assign initial state to sampled_paths and pointer
                sampled_paths[iter_num, path_num, 0] = state
                curr_state = state
                collision = False
                # Sample actions for each step of the episode
                for step in range(config.CEM_EPISODE_LENGTH):
                    # If first iteration, sample from uniform distribution
                    if iter_num == 0:
                        action = np.random.uniform(-constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE, 2)
                    # If not, sample an action from the distribution caleculated from previous iteration
                    else:
                        action = np.random.normal(action_mean[iter_num-1, step], action_std[iter_num-1, step], 2)
                        action = np.clip(action, -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE)
                    # Calculate the next state using the environment dynamics
                    next_state = self.environment.dynamics(curr_state, action)
                    joints = self.environment.get_joint_pos_from_state(next_state)
                    if self.collides(joints):
                        collision = True
                    sampled_actions[iter_num, path_num, step] = action
                    sampled_paths[iter_num, path_num, step+1] = next_state
                    curr_state = next_state
                
                # Calculate the distance between the final state and the goal state for this path
                final_state = sampled_paths[iter_num, path_num, -1]
                hand_pos = self.environment.get_joint_pos_from_state(final_state)[-1]
                distance = np.linalg.norm(hand_pos - self.environment.goal_state)
                if collision:
                    distance += 10.0
                path_distances[iter_num, path_num] = distance

            # Select the elite paths based on the distances
            elites = np.argsort(path_distances[iter_num])[:config.CEM_NUM_ELITES]
            elite_actions = sampled_actions[iter_num, elites]
            action_mean[iter_num] = np.mean(elite_actions, axis=0)
            action_std[iter_num] = np.std(elite_actions, axis=0)
            
        # OPTION 1: Planned action is the mean action sequence from the final iteration
        # self.planned_actions = action_mean[-1]

        # OPTION 2: Planned action is the best action sequence from the final iteration
        best_path_index = np.argmin(path_distances[-1])
        self.planned_actions = sampled_actions[-1, best_path_index]


        # Compute paths for mean action sequence for visualisation
        mean_paths = np.zeros([config.CEM_NUM_ITER, config.CEM_EPISODE_LENGTH+1, 2], dtype=np.float32)
        for iter_num in range(config.CEM_NUM_ITER):
            curr_state = state
            mean_paths[iter_num, 0] = state
            for step in range(config.CEM_EPISODE_LENGTH):
                next_state = self.environment.dynamics(curr_state, action_mean[iter_num, step])
                mean_paths[iter_num, step+1] = next_state
                curr_state = next_state
        
        # Create visualisation for the mean path in each iteration
        self.visualisation_lines = []
        for iter_num in range(config.CEM_NUM_ITER):
            intensity = (iter_num + 1) / config.CEM_NUM_ITER
            brightness = 50 + 205 * intensity
            colour = (brightness, brightness, brightness)
            width = 0.002 + 0.003 * intensity
            for step in range(config.CEM_EPISODE_LENGTH):
                a1_1, a2_1 = mean_paths[iter_num, step]
                a1_2, a2_2 = mean_paths[iter_num, step + 1]
                x1, y1 = self.environment.get_joint_pos_from_state([a1_1, a2_1])[-1]
                x2, y2 = self.environment.get_joint_pos_from_state([a1_2, a2_2])[-1]
                line = VisualisationLine(x1, y1, x2, y2, colour, width)
                self.visualisation_lines.append(line)
        
        final_states = sampled_paths[-1, :, -1]
        hand_positions = np.array([self.environment.get_joint_pos_from_state(s)[-1] for s in final_states])
        final_distances = np.linalg.norm(hand_positions - self.environment.goal_state, axis=1)
        
        print("Best/ Mean/ Worst final distances to goal in last iteration: ",
              np.min(final_distances),
              np.mean(final_distances),
              np.max(final_distances))
        
    
    def collides(self, joint_positions):
        obs_pos = self.environment.obstacle_pos
        obs_rad = self.environment.obstacle_radius

        for i in range(len(joint_positions) - 1):
            if self.environment.line_circle_intersection(
                joint_positions[i],
                joint_positions[i-1],
                obs_pos,
                obs_rad
            ):
                return True
        return False


             

                

# The VisualisationLine class enables us to store a line segment which will be drawn to the screen
class VisualisationLine:
    # Initialise a new visualisation (a new line)
    def __init__(self, x1, y1, x2, y2, colour=(255, 255, 255), width=0.01):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.colour = colour
        self.width = width


# The VisualisationCircle class enables us to a circle which will be drawn to the screen
class VisualisationCircle:
    # Initialise a new visualisation (a new circle)
    def __init__(self, x, y, radius, colour=(255, 255, 255)):
        self.x = x
        self.y = y
        self.radius = radius
        self.colour = colour