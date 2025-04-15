from __future__ import absolute_import
from __future__ import print_function

import optparse
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa


def get_vehicle_numbers(lanes):
    vehicle_per_lane = [0] * len(lanes)
    for i, lane in enumerate(lanes):
        for v in traci.lane.getLastStepVehicleIDs(lane):
            if traci.vehicle.getLanePosition(v) > 10:
                vehicle_per_lane[i] += 1
    return vehicle_per_lane


def get_waiting_time(lanes):
    return sum([traci.lane.getWaitingTime(lane) for lane in lanes])


def phaseDuration(junction, phase_time, phase_state):
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.policy = nn.Linear(fc2_dims, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        probs = F.softmax(self.policy(x), dim=-1)
        return probs


class PGAgent:
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, alpha=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(input_dims, fc1_dims, fc2_dims, n_actions)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=alpha)
        self.log_probs = []
        self.rewards = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.device)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def learn(self):
        G = 0
        returns = []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, Gt in zip(self.log_probs, returns):
            loss -= log_prob * Gt

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

        self.optimizer.step()

        self.log_probs = []
        self.rewards = []


def run(train=True, model_name="pg_model", epochs=50, steps=500):
    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg"])

    all_junctions = traci.trafficlight.getIDList()
    traci.close()

    input_dims = 4
    n_actions = 4

    agent = PGAgent(input_dims=input_dims, fc1_dims=256, fc2_dims=256, n_actions=n_actions)

    select_lane = [
        ["yyyrrrrrrrrr", "GGGrrrrrrrrr"],
        ["rrryyyrrrrrr", "rrrGGGrrrrrr"],
        ["rrrrrryyyrrr", "rrrrrrGGGrrr"],
        ["rrrrrrrrryyy", "rrrrrrrrrGGG"],
    ]

    best_time = float("inf")
    total_time_list = []

    for e in range(epochs):
        if train:
            traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg"])
        else:
            traci.start([checkBinary("sumo-gui"), "-c", "configuration.sumocfg",  "--tripinfo-output", "tripinfo.xml"])
        print(f"Epoch {e}")

        step = 0
        total_time = 0
        traffic_lights_time = {j: 0 for j in all_junctions}
        prev_action = {j: 0 for j in all_junctions}

        while step <= steps:
            traci.simulationStep()
            for junction in all_junctions:
                if traffic_lights_time[junction] == 0:
                    lanes = traci.trafficlight.getControlledLanes(junction)[:4]  # Limit to 4 lanes
                    state = get_vehicle_numbers(lanes)
                    waiting_time = get_waiting_time(lanes)
                    total_time += waiting_time

                    action = agent.choose_action(state)
                    phaseDuration(junction, 15, select_lane[action][1])
                    traffic_lights_time[junction] = 15

                    reward = -waiting_time
                    agent.store_reward(reward)

                else:
                    traffic_lights_time[junction] -= 1

            step += 1

        traci.close()
        agent.learn()
        total_time_list.append(total_time)

        if total_time < best_time:
            best_time = total_time
            if train:
                torch.save(agent.policy.state_dict(), f"models/{model_name}.pt")

        if not train:
            break

    if train:
        plt.plot(range(len(total_time_list)), total_time_list)
        plt.xlabel("Epoch")
        plt.ylabel("Total Waiting Time")
        plt.savefig(f"plots/time_vs_epoch_{model_name}.png")
        plt.show()
    else:
        print(total_time)


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("-m", dest='model_name', type='string', default="pg_model", help="Model name")
    optParser.add_option("--train", action='store_true', default=False, help="Train the model")
    optParser.add_option("-e", dest='epochs', type='int', default=50, help="Number of epochs")
    optParser.add_option("-s", dest='steps', type='int', default=500, help="Simulation steps per epoch")
    options, _ = optParser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    run(train=options.train, model_name=options.model_name, epochs=options.epochs, steps=options.steps)
