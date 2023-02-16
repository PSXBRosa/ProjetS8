import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functools import reduce
from .MemoryBuffer import MemoryBuffer

class DQNAgent:
    """
    Creates a very simple DQN agent
    """
    gamma = 0.99        # discount to be applied to get the discounted return
    decay = 0.9994      # decay for the probability of exploration (epsilon-greedy)
    min_epsilon = 0.01  # minimum probability of exploration
    lr = 1e-4           # learning rate
    
    def __init__(
        self,
        observation_space,
        action_space
    ):

        """
        Instantiates the agent with the provided parameters

        Parameters:
            observation_space : gym.spaces
                observation space of the environment
            action_space : gym.spaces
                action space of the environment
        Returns:
            None
        """

        self._os = observation_space
        self._as = action_space
        self.steps = 0
        self._epsilon = 1
        self.greedy = False

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # the algorithm works using experience replay, therefore we must setup a way of acessing
        # those past experiences, this all is handled by the custom class MemoryBuffer instantiated bellow
        self.memory = MemoryBuffer(100_000, observation_space.shape, action_space.n)

        # flattening of the action and observation spaces
        in_dim = reduce(lambda a,b: a*b, observation_space.shape)
        out_dim = action_space.n
        
        # creation of the neural network for calculating the Q values
        self._network = nn.Sequential(        # neural network
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

        # creation of the optimizer
        self._optimizer = optim.Adam(self._network.parameters(), lr=self.lr)

    def get_action(self, state):
        """
        implements the epsilon-greedy algorithm for choosing the action to be taken

        Parameters:
            state: array_like
                current state of the agent
        Returns:
            out: int
                action to be taken
        """

        # applies decay to the epsilon value if it's still above the lower limit
        self._epsilon *= self.decay if self._epsilon > self.min_epsilon else self.min_epsilon

        # exploration criteria
        if np.random.random() < self._epsilon and not self.greedy:
            return self._as.sample()

        with torch.no_grad():
            state = torch.tensor(state).to(self._device, dtype=torch.float)
            action = self._network(state).argmax(dim=-1)
            action = action.cpu().numpy()
            return action

    def _compute_loss(self, states, new_states, actions, rewards, terminals):
        """
        private method for computing the batch loss

        Parameters:
            states: torch.tensor
                batch of states
            new_states: torch.tensor
                batch of the following states
            actions: torch.tensor
                batch of actions taken in the state transitions
            rewards: torch.tensor
                batch of the rewards gathered
            terminals: torch.tensor
                batch of flags indicating if the new state is terminal or not
        Returns:
            loss: torch.Tensor
                computed loss
        """

        all_q_preds = self._network(states)

        # gets only the q-value of the actions that were actually taken. The vector's shape is also corrected in this line
        q_preds = all_q_preds.gather(1, actions.unsqueeze(1)).flatten()

        # we do not want these following calculations to interfere in the network optimization
        with torch.no_grad():
            # gets the maximum predicted q-value for the actions on each of the next states
            q_nexts = self._network(new_states).max(1)[0]

            # Bellman's equation
            target = (rewards + self.gamma*q_nexts*(1-terminals)).to(self._device)

        # mean squared error using the update law
        loss = F.mse_loss(q_preds, target)
        return loss


    def train(
        self,
        batch_size
    ):
        """
        step on the agent training

        Parameters:
            batch_size: int
                size of the batch of experiences to be replayed
        Returns:
            loss: float
                total loss for this training step
        """
        self.steps += 1

        # if the memory doesn't have enough experiences to sample
        if len(self.memory) < batch_size:
            return float("inf")

        # sample experiences
        batch = (torch.as_tensor(arr).to(self._device) for arr in self.memory.sample(batch_size))

        # gets the predicted q-value for the actions on each of the states
        self._network.train()
        
        loss = self._compute_loss(*batch)

        # backwards propagation
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._network.eval()
        return loss.item()

    def save(self, path):
        """
        saves the current network weights

        Parameters:
            path: str
                where to save the weights file
        Returns:
            None
        """

        torch.save(self._network, path)  

    @property
    def e(self):
        """
        getter method for the current epsilon value
        """
        return self._epsilon