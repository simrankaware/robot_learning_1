####################################
#      YOU MAY EDIT THIS FILE      #
# MOST OF YOUR CODE SHOULD GO HERE #
####################################

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import seaborn as sns

import constants
import config

plt.ion()
sns.set_theme(style="darkgrid", context="talk")


# ============================================================
# Robot
# ============================================================

class Robot:

    def __init__(self, forward_kinematics):
        self.forward_kinematics = forward_kinematics
        self.planning_visualisation_lines = []
        self.model_visualisation_lines = []
        self.robot_base_pos = np.array(constants.ROBOT_BASE_POS)

        self.goal_state = None
        self.num_steps = 0
        self.num_episodes = 0

        self.replay_buffer = ReplayBuffer()
        self.dynamics_model = DynamicsModel()

        # Planning
        self.planned_actions = None
        self.replan_steps = None

    def reset(self):
        self.num_steps = 0
        self.planned_actions = None

        T = config.EPISODE_LENGTH
        self.replan_steps = {0, T // 3, 2 * T // 3}

    def set_goal_state(self, goal_state):
        self.goal_state = goal_state

    def add_transition(self, state, action, next_state):
        self.replay_buffer.add_transition(state, action, next_state)

    # ------------------------------------------------------------
    # CLOSED-LOOP (SPARSE) PLANNING
    # ------------------------------------------------------------
    def select_action(self, state):

        if self.replan_steps is None:
            T = config.EPISODE_LENGTH
            self.replan_steps = {0, T // 3, 2 * T // 3}

        if self.planned_actions is None or self.num_steps in self.replan_steps:
            self.cem_planning(state)


        # Execute first action
        action = self.planned_actions[0].astype(np.float32)

        # Shift plan (warm-start)
        self.planned_actions = np.vstack([
            self.planned_actions[1:],
            self.planned_actions[-1:]
        ])

        self.num_steps += 1

        if self.num_steps == config.EPISODE_LENGTH:
            self.reset()
            self.num_episodes += 1
            self.dynamics_model.train(self.replay_buffer, config.TRAIN_NUM_MINIBATCH)
            self.create_model_visualisations()
            episode_done = True
        else:
            episode_done = False

        return action, episode_done

    # ------------------------------------------------------------
    # CEM PLANNING
    # ------------------------------------------------------------
    def cem_planning(self, state):

        H = config.CEM_EPISODE_LENGTH
        state = np.array(state, dtype=np.float32)

        sampled_actions = np.zeros(
            [config.CEM_NUM_ITER, config.CEM_NUM_PATHS, H, 2],
            dtype=np.float32
        )
        costs = np.zeros(
            [config.CEM_NUM_ITER, config.CEM_NUM_PATHS],
            dtype=np.float32
        )

        action_mean = np.zeros([config.CEM_NUM_ITER, H, 2], dtype=np.float32)
        action_std = np.zeros([config.CEM_NUM_ITER, H, 2], dtype=np.float32)

        init_std = 0.3 * constants.MAX_ACTION_MAGNITUDE

        for iter_num in range(config.CEM_NUM_ITER):
            for path_num in range(config.CEM_NUM_PATHS):

                curr_state = state.copy()

                for step in range(H):
                    if iter_num == 0:
                        action = np.random.uniform(
                            -constants.MAX_ACTION_MAGNITUDE,
                            constants.MAX_ACTION_MAGNITUDE,
                            size=(2,)
                        )
                    else:
                        action = np.random.normal(
                            action_mean[iter_num - 1, step],
                            action_std[iter_num - 1, step] + 1e-6,
                            size=(2,)
                        )

                    action = np.clip(
                        action,
                        -constants.MAX_ACTION_MAGNITUDE,
                        constants.MAX_ACTION_MAGNITUDE
                    )

                    sampled_actions[iter_num, path_num, step] = action
                    curr_state = self.dynamics_model.predict_next_state(curr_state, action)

                hand_pos = self.forward_kinematics(curr_state)[-1]
                goal_dist = np.linalg.norm(hand_pos - self.goal_state)
                motion_penalty = np.linalg.norm(curr_state - state)

                costs[iter_num, path_num] = goal_dist + 0.1 * motion_penalty

            elite_idx = np.argsort(costs[iter_num])[:config.CEM_NUM_ELITES]
            elites = sampled_actions[iter_num, elite_idx]

            action_mean[iter_num] = np.mean(elites, axis=0)
            action_std[iter_num] = np.std(elites, axis=0)

        best_path = np.argmin(costs[-1])
        self.planned_actions = sampled_actions[-1, best_path]

        self._create_planning_visualisation(state, action_mean)

    # ------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------
    def _create_planning_visualisation(self, state, action_mean):

        self.planning_visualisation_lines = []
        H = config.CEM_EPISODE_LENGTH

        for iter_num in range(config.CEM_NUM_ITER):
            curr_state = state.copy()
            intensity = (iter_num + 1) / config.CEM_NUM_ITER
            colour = (50 + 205 * intensity,) * 3
            width = 0.002 + 0.003 * intensity

            for step in range(H):
                next_state = self.dynamics_model.predict_next_state(
                    curr_state, action_mean[iter_num, step]
                )

                x1, y1 = self.forward_kinematics(curr_state)[-1]
                x2, y2 = self.forward_kinematics(next_state)[-1]

                self.planning_visualisation_lines.append(
                    VisualisationLine(x1, y1, x2, y2, colour, width)
                )

                curr_state = next_state

    def create_model_visualisations(self):
        self.model_visualisation_lines = []


# ============================================================
# VisualisationLine
# ============================================================

class VisualisationLine:
    def __init__(self, x1, y1, x2, y2, colour, width):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.colour = colour
        self.width = width


# ============================================================
# Dynamics Model
# ============================================================

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )

    def forward(self, x):
        return self.net(x)


class DynamicsModel:
    def __init__(self):
        self.network = Network()
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.losses = []

        self.losses = []
        self.fig, self.ax = plt.subplots()
        self.ax.set_yscale("log")
        self.line, = self.ax.plot([])

    def train(self, buffer, num_minibatches):
        if buffer.size < config.TRAIN_MINIBATCH_SIZE:
            return

        for _ in range(num_minibatches):
            x, y = buffer.sample_minibatch(config.TRAIN_MINIBATCH_SIZE)
            pred = self.network(x)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.losses.append(loss.item())
        self.line.set_ydata(self.losses)
        self.line.set_xdata(range(len(self.losses)))
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def predict_next_state(self, state, action):
        inp = np.concatenate([state, action]).astype(np.float32)
        inp = torch.from_numpy(inp).unsqueeze(0)

        with torch.no_grad():
            out = self.network(inp)

        return out.squeeze(0).numpy()


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self):
        self.data = []
        self.size = 0

    def add_transition(self, state, action, next_state):
        self.data.append((
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
        ))
        self.size += 1

    def sample_minibatch(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        batch = [self.data[i] for i in idx]

        states, actions, next_states = zip(*batch)
        x = torch.from_numpy(np.hstack([states, actions])).float()
        y = torch.from_numpy(np.vstack(next_states)).float()

        return x, y
