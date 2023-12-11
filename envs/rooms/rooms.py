import numpy as np
from gym import core, spaces
import random
import cv2
from tqdm import tqdm
import time


class RoomsEnv(core.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, rows=16, cols=16, empty=False, random_walls=False, discrete=True, sink_goal=True,
                 n_redundancies=1, max_repeats=1, goal=None, state=None,
                 goal_in_state=True, max_steps=None,
                 goal_only_visible_in_room=False, seed=None, vis=False,
                 fixed_reset=False, vert_wind=(0, 0), horz_wind=(0, 0)):
        '''
        vert_wind = (up, down)
        horz_wind = (right, left)
        '''
        self.rows, self.cols = rows, cols
        if max_steps is None:
            repeats = 3 if empty else 7
            self.max_steps = (rows + cols) * repeats
        else:
            self.max_steps = max_steps

        self.sink_goal = sink_goal
        self.goal_in_state = goal_in_state
        self.goal_only_visible_in_room = goal_only_visible_in_room

        self.vert_wind = np.array(vert_wind)
        self.horz_wind = np.array(horz_wind)

        self.n_redundancies = n_redundancies
        self.max_repeats = max_repeats
        self.discrete = discrete
        self.scale = np.maximum(rows, cols)
        self.im_size = 42
        self.goal_th = 1
        if self.discrete:
            self.action_space = spaces.Discrete(3 + n_redundancies)
            n_channels = 2 + goal_in_state
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.im_size, self.im_size, n_channels), dtype=np.uint8)
            self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1))] + [
                np.array((0, 1))] * n_redundancies
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
            n_channels = 2 + goal_in_state * 2
            self.observation_space = spaces.Box(low=0, high=1, shape=(n_channels,), dtype=np.float6464)
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        self.map, self._seed = self._randomize_walls(random=random_walls, empty=empty)
        self.goal_cell, self.goal = self._random_from_map(goal)
        self.state_cell, self.state = self._random_from_map(state)
        self.pos = self.state_cell.astype(np.float64)

        self.fixed_reset = fixed_reset
        if fixed_reset:
            self.reset_state_cell, self.reset_state = self.state_cell.copy(), self.state.copy()
        else:
            self.reset_state_cell, self.reset_state = None, None

        self.tot_reward = 0
        self.viewer = None
        self.vis = vis
        self.occupancy_image = None


    def reset(self):
        if self.fixed_reset:
            self.state_cell, self.state = self.reset_state_cell, self.reset_state
            self.pos = self.state_cell.astype(np.float64)
        else:
            self.state_cell, self.state = self._random_from_map(None)

        self.nsteps = 0
        self.tot_reward = 0

        obs = self._obs_from_state(self.discrete)
        return obs

    def step(self, action: int):
        # actions: 0 = up, 1 = down, 2 = left, 3:end = right
        for _ in range(random.randint(1, self.max_repeats)):
            self._move(action)
            if self.discrete:
                r = np.all(self.state_cell == self.goal_cell).astype(np.float64)
            else:
                r = (np.abs((self.pos - self.goal_cell)).sum() < self.goal_th).astype(np.float64)
            obs = self._obs_from_state(self.discrete)

            if self.nsteps >= self.max_steps or (r and self.sink_goal):
                done = True
            else:
                done = False

            self.tot_reward += r
            self.nsteps += 1
            info = dict()

            if done:
                info['episode'] = {'r': self.tot_reward, 'l': self.nsteps}
                break
            if self.vis:
                self.update(obs)
                self.render(img=self.occupancy_image.astype(np.uint8))

        return obs, r, done, info

    def _move(self, action, discrete=None):
        if self.discrete or discrete:
            return self._move_discrete(action)
        else:
            return self._move_continuous(action)

    def update(self, image_t):
        if self.occupancy_image is None:
            self.occupancy_image = image_t
        else:
            self.occupancy_image = np.clip(self.occupancy_image.astype(np.int) + image_t, 0, 255)

    def _move_discrete(self, action):
        wind_u = np.random.binomial(n=1, p=self.vert_wind[0])
        wind_d = np.random.binomial(n=1, p=self.vert_wind[1])
        wind_r = np.random.binomial(n=1, p=self.horz_wind[0])
        wind_l = np.random.binomial(n=1, p=self.horz_wind[1])
        next_cell = self.state_cell + self.directions[action] + [wind_d - wind_u, wind_r - wind_l]
        try:
            if self.map[next_cell[0], next_cell[1]] == 0:
                self.state_cell = next_cell
                self.state = np.zeros_like(self.map)
                self.state[(self.state_cell[0]):(self.state_cell[0] + 1),
                           (self.state_cell[1]):(self.state_cell[1] + 1)] = 1
        except IndexError:
            return

    def _move_continuous(self, action, ex=1, ey=1):
        xy = action
        x, y = xy
        # redundancy is made by making the 'right' movement harder
        if y < 0:
            ey = self.n_redundancies
        # redundancy is made by making the 'up' movement harder
        if x < 0:
            ex = self.n_redundancies
        new_pos = [np.sign(x) * abs(x**ex), np.sign(y) * abs(y**ey)] + self.pos

        wind_u = np.random.normal(self.vert_wind[0], 0.1)
        wind_d = np.random.normal(self.vert_wind[1], 0.1)
        wind_r = np.random.normal(self.horz_wind[0], 0.1)
        wind_l = np.random.normal(self.horz_wind[1], 0.1)
        new_pos += [wind_d - wind_u, wind_r - wind_l]

        next_cell = np.round(new_pos).astype(np.int)
        try:
            if self.map[next_cell[0], next_cell[1]] == 0 and \
                    np.all(next_cell) >= 0 and \
                    next_cell[0] < self.rows - 1 and next_cell[1] < self.cols - 1:
                self.state_cell = next_cell
                self.pos = new_pos
                self.state = np.zeros_like(self.map)
                self.state[(self.state_cell[0]):(self.state_cell[0] + 1),
                           (self.state_cell[1]):(self.state_cell[1] + 1)] = 1
        except IndexError:
            return

    def _random_from_map(self, goal):
        if goal is None:
            cell = self.rng.choice(self.rows), self.rng.choice(self.cols)
        else:
            cell = tuple(goal)
        while self.map[cell[0], cell[1]] == 1:
            cell = self.rng.choice(self.rows), self.rng.choice(self.cols)
        map = np.zeros_like(self.map)
        for i in [0]:#[-1, 0, 1]:
            for j in [0]:#[-1, 0, 1]:
                map[(cell[0] + i):
                    (cell[0] + i + 1),
                    (cell[1] + j):
                    (cell[1] + j + 1)] = 1
        return np.array(cell), map

    def _obs_from_state(self, discrete):
        if discrete:
            im_list = [self.state, self.map]
            if self.goal_in_state:
                if self.goal_only_visible_in_room:
                    if self._which_room(self.state_cell) == self._which_room(self.goal_cell):
                        im_list.append(self.goal)
                    else:
                        im_list.append(np.zeros_like(self.map))
                else:
                    im_list.append(self.goal)
            im_stack = 70 * np.stack(im_list, axis=-1).astype(np.uint8)
            return cv2.resize(im_stack, dsize=(self.im_size, self.im_size), interpolation=cv2.INTER_AREA)
        else:
            obs = list(self.pos)
            if self.goal_in_state:
                if self.goal_only_visible_in_room:
                    if self._which_room(self.state_cell) == self._which_room(self.goal_cell):
                        obs += list(self.goal_cell)
                else:
                    obs += list(self.goal_cell)
            return np.array(obs) / self.scale

    def _which_room(self, cell):
        if cell[0] <= self._seed[0] and cell[1] <= self._seed[1]:
            return 0
        elif cell[0] <= self._seed[0] and cell[1] > self._seed[1]:
            return 1
        elif cell[0] > self._seed[0] and cell[1] <= self._seed[1]:
            return 2
        else:
            return 3

    def _randomize_walls(self, random=False, empty=False):
        map = np.zeros((self.rows, self.cols))

        map[0, :] = 1
        map[:, 0] = 1
        map[-1:, :] = 1
        map[:, -1:] = 1

        if empty:
            return map, 0

        if random:
            seed = (self.rng.randint(2, self.rows - 2), self.rng.randint(2, self.cols - 2))
            doors = (self.rng.randint(1, seed[0]),
                     self.rng.randint(seed[0] + 1, self.rows - 1),
                     self.rng.randint(1, seed[1]),
                     self.rng.randint(seed[1] + 1, self.cols - 1))
        else:
            seed = (self.rows // 2, self.cols // 2)
            doors = (self.rows // 4, 3 * self.rows // 4, self.cols // 4, 3 * self.cols // 4)

        map[seed[0]:seed[0] + 1, :] = 1
        map[:, seed[1]:(seed[1] + 1)] = 1
        map[doors[0]:(doors[0]+1), seed[1]:(seed[1] + 1)] = 0
        map[doors[1]:(doors[1]+1), seed[1]:(seed[1] + 1)] = 0
        map[seed[0]:(seed[0] + 1), doors[2]:(doors[2]+1)] = 0
        map[seed[0]:(seed[0] + 1), doors[3]:(doors[3]+1)] = 0

        return map, seed

    def render(self, mode='human', img=None):
        if img is None:
            img = self._obs_from_state(True)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen


if __name__ == '__main__':

    n_redundancies = 2
    discrete = False
    max_repeats = 1
    room_size = 10
    up_wind = 0.
    down_wind = 0.
    right_wind = 0.
    left_wind = 0.

    env = RoomsEnv(rows=room_size, cols=room_size, discrete=discrete,
                   goal=[1, 1], state=[room_size - 2, room_size - 2],
                   fixed_reset=True, n_redundancies=n_redundancies, max_repeats=max_repeats,
                   horz_wind=(right_wind, left_wind), vert_wind=(up_wind, down_wind), empty=False, seed=0)

    obs = env.reset()
    for _ in tqdm(range(10000), desc='', leave=True):
        obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
        time.sleep(0.03)
        if done:
            obs = env.reset()
