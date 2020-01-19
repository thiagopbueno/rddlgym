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


from rddlgym import Trajectory


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
        if hasattr(self.planner, "build"):
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

        trajectory = Trajectory(self.env)

        while not done:
            action = self.planner(state, timestep)
            next_state, reward, done, info = self.env.step(action)

            trajectory.add_transition(
                timestep, state, action, reward, next_state, info, done
            )

            if mode is not None:
                self.env.render(mode)

            state = next_state
            timestep = self.env.timestep

        return trajectory

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
