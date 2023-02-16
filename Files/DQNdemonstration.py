import gymnasium as gym
import numpy as np
import os
from datetime import datetime
from MyModule.Agent import DQNAgent
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import json

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
    agent = DQNAgent(env.observation_space, env.action_space)

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

    # saving the agent variables
    temp_dict = {}
    temp_dict.update(agent.__dict__)
    for cl in agent.__class__.mro()[:-1]:
        temp_dict.update(cl.__dict__)
    temp_dict = {k:v for k,v in temp_dict.items() if "__" not in k} # removes all the special attributes
    with open(path + 'attr.json','w') as fp:
        json.dump(temp_dict, fp, skipkeys=True, default=str, indent=4) # dumps into a file

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