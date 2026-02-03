####################################
#      YOU MAY EDIT THIS FILE      #
# MOST OF YOUR CODE SHOULD GO HERE #
####################################

# Imports from external libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import seaborn as sns

# Imports from this project
import constants
import config

# Configure matplotlib for interactive mode
plt.ion()

# Set the seaborn style
sns.set_theme(style="darkgrid", context="talk")


# The Robot class (which could be called "Agent") is the "brain" of the robot, and is used to decide what action to execute in the environment
class Robot:

    # Initialise a new robot
    def __init__(self, forward_kinematics):
        # Give the robot the forward kinematics function, to calculate the hand position from the state
        self.forward_kinematics = forward_kinematics
        # A list of visualisations which will be displayed on the right-side of the window
        self.planning_visualisation_lines = []
        self.model_visualisation_lines = []
        # The position of the robot's base
        self.robot_base_pos = np.array(constants.ROBOT_BASE_POS)
        # The goal state
        self.goal_state = 0
        # The number of steps in the episode
        self.num_steps = 0
        # The number of current episodes
        self.num_episodes = 0
        # Check for model being trained (redo this comment)
        self.model_trained = False


        self.replay_buffer = ReplayBuffer()
        self.dynamics_model = DynamicsModel()

    # Reset the robot at the start of an episode
    def reset(self):
        self.num_steps = 0

    # Give the robot access to the goal state
    def set_goal_state(self, goal_state):
        self.goal_state = goal_state

    # Function to get the next action in the plan
    def select_action(self, state):
        # For now, a random action, biased towards 'moving right'
        action = np.random.uniform(low=-constants.MAX_ACTION_MAGNITUDE, high=0.5*constants.MAX_ACTION_MAGNITUDE, size=2)
        self.num_steps += 1

        if self.num_steps == 0:
            # Reset visualisation 
            self.model_visualisation_lines = []
            self.planning_visualisation_lines = []
            self.cem_planning(state)

        if self.num_steps == config.EPISODE_LENGTH:
            self.reset()
            self.num_episodes += 1 
            episode_done = True
        else:
            episode_done = False
        # Train the model on all data collected every 20 peisodes
        if self.num_episodes == 20 and not self.model_trained:
            self.dynamics_model.train(self.replay_buffer)
            self.create_model_visualisations()
            self.model_trained = True
        return action, episode_done

    # Function to add a transition to the buffer
    def add_transition(self, state, action, next_state):
        self.replay_buffer.add_transition(state, action, next_state)

    def create_model_visualisations(self):
        # TO FILL IN
        pass


# The VisualisationLine class enables us to store a line segment which will be drawn to the screen
class VisualisationLine:
    # Initialise a new visualisation (a new line)
    def __init__(self, x1, y1, x2, y2, colour, width):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.colour = colour
        self.width = width



# This is the network that is trained on the transition data 
class Network(nn.Module):

    # Initialise 
    def __init__(self, input_size=4, hidden_size=20, output_size=2):
        super(Network, self).__init__()
        # Define the first hidden layer
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.activation1 = nn.ReLU()
        # Define the second hidden layer
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.activation2 = nn.ReLU()
        # Define the third hidden layer
        self.hidden3 = nn.Linear(hidden_size, hidden_size)
        self.activation3 = nn.ReLU()
        # Define the output layer
        self.output = nn.Linear(hidden_size, output_size)


    # Forward pass
    def forward(self, input):
        # Pass data through the layers
        x = self.hidden1(input)
        x = self.activation1(x)
        x = self.hidden2(x)
        x = self.activation2(x)
        x = self.hidden3(x)
        x = self.activation3(x)
        # Pass data through output layer
        output = self.output(x)
        return output
    
# DynamicsModel is used to learn the environment dynamics 
class DynamicsModel:

    def __init__(self):
        self.network = Network()
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.01)
        self.loss_function = nn.MSELoss()
        self.losses = []

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Num Minibatches')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Loss Curve')
        self.ax.set_yscale('log')
        self.line, = self.ax.plot([], [], linestyle='-', marker=None, color='blue')
        plt.show()
    
    # Start training the dynamics model on the data in the buffer
    def train(self, buffer, num_minibatches=1000, minibatch_size=64):
        if buffer.size < minibatch_size:
            return
        loss_sum = 0
        for _ in range(num_minibatches):
            inputs, targets = buffer.sample_minibatch(minibatch_size)

            predictions = self.network(inputs)
            loss = self.loss_function(predictions, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log loss per minibatch
            self.losses.append(loss.item())
        self.line.set_xdata(range(1, len(self.losses)+1))
        self.line.set_ydata(self.losses)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
            

    # Use the dynamics model to predict the next state, which requires a forward pass of the network
    def predict_next_state(self, state, action):
        # Prepare the input tensor
        state_action = np.concatenate([state, action])
        input = torch.from_numpy(state_action).float().unsqueeze(0) # Add batch dimension
        # Forward pass
        self.network.eval()
        with torch.no_grad():
            prediction_tensor = self.network(input)
        
        next_state = prediction_tensor.squeeze(0).numpy()
        return next_state

class ReplayBuffer:
    # Initialise
    def __init__(self):
        self.data = []
        self.size = 0
    
    # Add a transition (state, action, next_state) to the buffer, each time the robot takes a step
    def add_transition(self, state, action, next_state):
        self.data.append((state, action, next_state))
        self.size += 1
    
    # Retrieve data collected
    def retrieve_data(self):
        states, actions, next_states = zip(*self.data)
        inputs = np.concatenate([states, actions], axis=1)
        targets = np.array(next_states)

        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()

        return inputs, targets

    def sample_minibatch(self, minibatch_size):
        if minibatch_size > self.size:
            raise ValueError("Minibatch size larger than buffer size.")
        
        indices = np.random.choice(self.size, minibatch_size, replace=False)
        minibatch = [self.data[i] for i in indices]

        states, actions, next_states = zip(*minibatch)
        inputs = np.concatenate([states, actions], axis=1)
        targets = np.array(next_states)

        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()

        return inputs, targets
