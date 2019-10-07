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


from collections import namedtuple


Transition = namedtuple("Transition", "step state action reward next_state info done")


class Runner:
    """Runner class implements the planner-environment loop.

    Args:
        env (rddlgym.RDDLEnv): The RDDLEnv gym environment.
        planner (tfplan.planners.Planner): The planner.
        debug (bool): The debug flag.
    """

    def __init__(self, env, planner, debug=False):
        self.env = env
        self.planner = planner
        self.debug = debug

    def build(self):
        """Builds the runner's underlying components."""
        self.planner.build()

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

        total_reward = 0.0
        trajectory = []

        while not done:
            action = self.planner(state, timestep)
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward

            trajectory.append(
                Transition(timestep, state, action, reward, next_state, info, done)
            )

            if mode is not None:
                self.env.render(mode)

            if self.debug:
                self._print_debug_info(
                    timestep, state, action, reward, next_state, done, info
                )

            state = next_state
            timestep = self.env.timestep

        return total_reward, trajectory

    def close(self):
        """Closes the environment."""
        self.env.close()

    def __enter__(self):
        self.build()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    @staticmethod
    def _print_debug_info(timestep, state, action, reward, next_state, done, info):
        # pylint: disable=too-many-arguments
        print(f"::: Timestep = {timestep} :::")
        print()
        print(">> State:")
        for name, fluent in state.items():
            print(f"{name} = {fluent}")
        print()
        print(">> Action:")
        for name, fluent in action.items():
            print(f"{name} = {fluent}")
        print()
        print(">> Next State:")
        for name, fluent in next_state.items():
            print(f"{name} = {fluent}")
        print()
        print(">> Info:")
        for name, fluent in info.items():
            print(f"{name} = {fluent}")
        print()
        print(f">> Reward = {reward}")
        print(f">> Terminal = {done}")
        print()
