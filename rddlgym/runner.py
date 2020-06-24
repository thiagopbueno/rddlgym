# This file is part of rddlgym.

# rddlgym is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# rddlgym is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with rddlgym. If not, see <http://www.gnu.org/licenses/>.

# pylint: disable=missing-docstring

import time


from rddlgym import Trajectory


class Runner:
    """Runner class implements the planner-environment loop.

    Args:
        env (rddlgym.RDDLEnv): The RDDLEnv gym environment.
        planner (tfplan.planners.Planner): The planner.
        debug (bool): The debug flag.
    """

    def __init__(self, env, planner, **kwargs):
        self.env = env
        self.planner = planner

        self.debug = kwargs.get("debug", False)

        self._on_episode_start_hook = kwargs.get("on_episode_start")
        self._on_episode_end_hook = kwargs.get("on_episode_end")
        self._on_step_hook = kwargs.get("on_step")

    def build(self):
        """Builds the runner's underlying components."""
        if hasattr(self.planner, "build"):
            self.planner.build()

    def _on_episode_start(self):
        if self._on_episode_start_hook:
            self._on_episode_start_hook()

    def _on_episode_end(self, trajectory, uptime):
        if self._on_episode_end_hook:
            self._on_episode_end_hook(trajectory, uptime)

    def _on_step(self, timestep, state, action, reward, next_state, done, info):
        # pylint: disable=too-many-arguments
        if self._on_step_hook:
            self._on_step_hook(timestep, state, action, reward, next_state, done, info)

    def run(self, mode=None):
        """Runs the planner-environment loop until termination.

        Args:
            mode (str): The environment render mode.

        Returns:
            total_reward (float): The total reward for the run.
            trajectory (List[Transition]): The state-action-reward trajectory.
        """
        state, timestep = self.env.reset()
        done = False

        trajectory = Trajectory(self.env)

        start_time = time.perf_counter()
        self._on_episode_start()

        total_reward = 0.0

        while not done:
            action = self.planner(state, timestep)
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward

            info["total_reward"] = total_reward

            self._on_step(timestep, state, action, reward, next_state, done, info)

            trajectory.add_transition(
                timestep, state, action, reward, next_state, info, done
            )

            if mode is not None:
                self.env.render(mode)

            state = next_state
            timestep = self.env.timestep

        uptime = time.perf_counter() - start_time
        self._on_episode_end(trajectory, uptime)

        return trajectory, uptime

    def close(self):
        """Closes the environment."""
        if hasattr(self.planner, "close"):
            self.planner.close()

        self.env.close()

    def __enter__(self):
        self.build()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
