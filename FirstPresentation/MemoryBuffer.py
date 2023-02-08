import numpy as np

class MemoryBuffer:
    """
    Custom class for storing and handling the experiences used for training the network using
    experience replay
    """
    
    def __init__(self, maxlen, obs_dims, n_actions):
        """
        Initializes the memory buffer with the given parameters

        Parameters:
            maxlen : int
                maximum lenght for the memeory buffer
            obs_dims: tuple
                observation samples dimentions
            n_actions : int
                number of possible actions
        Returns:
            None
        """
        
        self._states = np.zeros((maxlen, *obs_dims), dtype=np.float32)
        self._new_states = np.zeros((maxlen, *obs_dims), dtype=np.float32)
        self._actions = np.zeros(maxlen, dtype=np.int64)
        self._rewards = np.zeros(maxlen, dtype=np.float32)
        self._terminal = np.zeros(maxlen, dtype=np.float32)

        self._index = 0
        self._maxlen = maxlen
        self._full = False
    
    def __len__(self):
        return self._index if not self._full else self._maxlen

    def include(self, state, new_state, action, reward, terminal):
        """
        includes a set of samples to the memory buffer

        Parameters:
            state : array_like
                initial state observation
            new_state : array_like
                new state observation
            action : array_like
                action taken
            reward : float
                reward gotten from the transition
            terminal : bool
                whether the new_state is or is not terminal
        Returns:
            None
        """

        self._states[self._index] = state
        self._new_states[self._index] = new_state
        self._actions[self._index] = action
        self._rewards[self._index] = reward
        self._terminal[self._index] = terminal

        # increases the current index position, it's important to note 
        # that if there's an overflow, the index will return to the start of the buffer
        # and will start rewriting the data
        self._index = (self._index + 1) % self._maxlen 

        # index is at the last position
        if self._index == self._maxlen - 1:
            # updated full flag
            self._full = True

    def sample(self, n):
        """
        samples n experiences from the buffer

        Parameters:
            n : int
                number of experiences to sample
        Returns:
            states : np.array
                state batch
            new states : np.array
                new state batch
            actions : np.array
                action batch
            rewards : np.array
                reward batch
            terminals : np.array
                terminal info batch
        """

        indexes = np.random.randint(0, len(self), n)
        return (self._states[indexes],
                self._new_states[indexes],
                self._actions[indexes],
                self._rewards[indexes],
                self._terminal[indexes])

              
