import gymnasium as gym
import numpy as np
import os
from datetime import datetime
from MyModule.Agent import DQNAgent
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from copy import deepcopy
import torch
import torch.nn.functional as F

class DDQNAgent(DQNAgent):
    TAU = 1 # How much of the policy network should be copied into the target network ( between 0 and 1 )
    N = 1_000 # How ofter that copying process must happen ( in time steps )

    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        # creates targets network
        self._target = deepcopy(self._network)

    # for copying weights into the new network
    def _copy_weights(self):
        for target_param, param in zip(self._target.parameters(), self._network.parameters()):
            target_param.data.copy_(self.TAU * param + (1-self.TAU) * target_param)

    # overwritting the parent method
    def _compute_loss(self, states, new_states, actions, rewards, terminals):

        all_q_preds = self._network(states)

        # gets only the q-value of the actions that were actually taken. The vector's shape is also corrected in this line
        q_preds = all_q_preds.gather(1, actions.unsqueeze(1)).flatten()

        # we do not want these following calculations to interfere in the network optimization
        with torch.no_grad():
            # gets the maximum predicted q-value for the actions on each of the next states
            q_nexts = self._target(new_states).max(1)[0]

            # Bellman's equation
            target = (rewards + self.gamma*q_nexts*(1-terminals)).to(self._device)

        # mean squared error using the update law
        loss = F.mse_loss(q_preds, target)
        return loss
    
    # overwritting the parent method
    def train(self, batch_size):
        loss = super().train(batch_size)
        if self.steps%self.N == 0:
            self._copy_weights()
        return loss


if __name__ == "__main__":
    # creating the environment
    env_name = "CartPole-v1"
    env = gym.make(env_name, render_mode="rgb_array_list")

    # creating a new directory for outputting the results
    dirname, filename = os.path.dirname(__file__), os.path.basename(__file__)
    dir = "training sessions/" + datetime.now().strftime("%m-%d-%Y %Hh%Mmin%Ss") + " - " + env_name + " " + filename + "/"
    path = os.path.join(dirname, dir)
    os.makedirs(path)

    # creating the agent
    agent = DDQNAgent(env.observation_space, env.action_space)

    # maximum number of episodes
    max_episodes = 350

    # to store every epsiode data for later plotting
    all_returns = np.zeros(max_episodes)
    all_losses = np.zeros(max_episodes)

    # main loop
    for i in range(0, max_episodes):
        steps = 0
        episode_return = 0
        episode_loss = 0

        # environment variables before the episode starts
        done = False
        state, _ = env.reset()

        # while there's not a terminal state
        while not done:
            
            # agent chooses an action
            action = agent.get_action(state)

            # state transition
            next_state, reward, done, _, _ = env.step(action)

            # save experience on memory and traning
            agent.memory.include(state, next_state, action, reward, done)
            loss = agent.train(256)

            #
            steps +=1
            episode_return += reward
            episode_loss += loss

            # next state becomes the current state
            state = next_state
        
        # printing values
        print(f"\rEPISODE [{i + 1}] | RETURN [{episode_return}] | FINAL LOSS [{(episode_loss/steps):.2f}] | EPSILON [{agent.e:.2f}]            ", end="")
        all_returns[i] = episode_return
        all_losses[i] = episode_loss/steps
    
    # plotting and saving training results
    agent.save(path + "weights.pth")
    np.savetxt(path + "training_data.csv", np.vstack((all_returns, all_losses)), delimiter=",")
    
    sns.set_theme()

    # plotting the returns
    plt.figure(figsize=(12,8))
    plt.title("Episode Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.plot(np.arange(max_episodes), all_returns)
    plt.savefig(path + "returns.png")

    # plotting the losses
    plt.clf()
    plt.title("Episode Losses")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.plot(np.arange(max_episodes), all_losses)
    plt.savefig(path + "losses.png")

    # save the last episode as a gif
    frames = env.render()
    imageio.mimsave(path + "last_episode.gif", frames, fps=30)