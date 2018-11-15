import numpy as np
from gym import spaces
from gym import Env


class ObservedPointEnv(Env):
    """
    point mass on a 2-D plane
    four tasks: move to (-10, -10), (-10, 10), (10, -10), (10, 10)

    Problem 1: augment the observation with a one-hot vector encoding the task ID
     - change the dimension of the observation space
     - augment the observation with a one-hot vector that encodes the task ID
    """
    #====================================================================================#
    #                           ----------PROBLEM 1----------
    #====================================================================================#
    # YOUR CODE SOMEWHERE HERE
    def __init__(self, num_tasks=1):
        self.tasks = [0, 1, 2, 3][:num_tasks]
        self.task_idx = -1
        self.task_one_hot = np.zeros(len(self.tasks))
        self.reset_task()
        self.reset()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def concat_task_vec(self, coords):
        if len(coords.shape) == 2:
            task_one_hot = np.tile(self.task_one_hot, (coords.shape[0], 1))
        else:
            task_one_hot = self.task_one_hot
        #     new_obs_vec = np.zeros((coords.shape[0], (coords.shape[1]+len(self.tasks))))
        #     for i, c in enumerate(coords):
        #         new_obs_vec[i] = np.hstack([c, self.task_one_hot])
        # else:
        new_obs_vec = np.hstack([coords, task_one_hot])
        return new_obs_vec

    def reset_task(self, is_evaluation=False):
        # for evaluation, cycle deterministically through all tasks
        if is_evaluation:
            self.task_idx = (self.task_idx + 1) % len(self.tasks)
        # during training, sample tasks randomly
        else:
            self.task_idx = np.random.randint(len(self.tasks))
        self.task_one_hot[self.task_idx] = 1
        self._task = self.tasks[self.task_idx]
        goals = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        self._goal = np.array(goals[self.task_idx])*10

    def reset(self):
        self._state = self.concat_task_vec(np.array([0, 0], dtype=np.float32))[None, :]
        return self._get_obs()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        # print(self._state[:2].shape)
        x, y = np.squeeze(self._state[:, :2])
        # compute reward, add penalty for large actions instead of clipping them
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        # check if task is complete
        done = abs(x) < 0.01 and abs(y) < 0.01
        # move to next state
        self._state = self.concat_task_vec(np.squeeze(self._state[:, :2]) + action)
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

    def seed(self, seed):
        np.random.seed = seed


# import numpy as np
# from gym import spaces
# from gym import Env
#
#
# class ObservedPointEnv(Env):
#     """
#     point mass on a 2-D plane
#     four tasks: move to (-10, -10), (-10, 10), (10, -10), (10, 10)
#     Problem 1: augment the observation with a one-hot vector encoding the task ID
#      - change the dimension of the observation space
#      - augment the observation with a one-hot vector that encodes the task ID
#     """
#     #====================================================================================#
#     #                           ----------PROBLEM 1----------
#     #====================================================================================#
#     # YOUR CODE SOMEWHERE HERE
#     def __init__(self, num_tasks=1):
#         self.tasks = [0, 1, 2, 3][:num_tasks]
#         self.task_idx = -1
#         self.task_one_hot = np.zeros(len(self.tasks))
#         self.reset_task()
#         self.reset()
#
#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
#         self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))
#
#     def reset_task(self, is_evaluation=False):
#         # for evaluation, cycle deterministically through all tasks
#         if is_evaluation:
#             self.task_idx = (self.task_idx + 1) % len(self.tasks)
#         # during training, sample tasks randomly
#         else:
#             self.task_idx = np.random.randint(len(self.tasks))
#         self.task_one_hot[self.task_idx] = 1
#         self._task = self.tasks[self.task_idx]
#         goals = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
#         self._goal = np.array(goals[self.task_idx])*10
#
#     def reset(self):
#         self._state = np.zeros(2, dtype=np.float32)
#         # np.array([0, 0], dtype=np.float32)[None, :]
#         return self._get_obs()
#
#     def _get_obs(self):
#         state = np.hstack([np.copy(self._state)[], np.tile(self.task_one_hot, (self._state.shape[0], 1))])
#         return state
#
#     def step(self, action):
#         x, y = self._state
#         # compute reward, add penalty for large actions instead of clipping them
#         x -= self._goal[0]
#         y -= self._goal[1]
#         reward = - (x ** 2 + y ** 2) ** 0.5
#         # check if task is complete
#         done = abs(x) < 0.01 and abs(y) < 0.01
#         # move to next state
#         self._state = self._state + action
#         ob = self._get_obs()
#         return ob, reward, done, dict()
#
#     def viewer_setup(self):
#         print('no viewer')
#         pass
#
#     def render(self):
#         print('current state:', self._state)
#
#     def seed(self, seed):
#         np.random.seed = seed
