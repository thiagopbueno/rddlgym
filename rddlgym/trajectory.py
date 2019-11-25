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


# pylint: disable=missing-docstring,too-many-locals,protected-access


from collections import OrderedDict, namedtuple
import numpy as np
import pandas as pd


Transition = namedtuple("Transition", "step state action reward next_state info done")


class Trajectory:
    """Trajectory class handles state-action-interm-reward sequences."""

    def __init__(self, env):
        self.env = env

        self._trajectory = []

    def add_transition(self, step, state, action, reward, next_state, info, done):
        """Adds transition to the trajectory."""
        # pylint: disable=too-many-arguments
        transition = Transition(step, state, action, reward, next_state, info, done)
        self._trajectory.append(transition)

    def as_dataframe(self):
        """Returns the trajectory as a dataframe with columns as fluent variables."""
        # pylint: disable=too-many-branches
        state_fluent_variables = self.env._compiler.rddl.state_fluent_variables
        states_dict = OrderedDict({})
        for _, state_vars in state_fluent_variables:
            for name in state_vars:
                states_dict[name] = np.empty(shape=(len(self),))

        action_fluent_variables = self.env._compiler.rddl.action_fluent_variables
        actions_dict = OrderedDict({})
        for _, action_vars in action_fluent_variables:
            for name in action_vars:
                actions_dict[name] = np.empty(shape=(len(self),))

        interm_fluent_variables = self.env._compiler.rddl.interm_fluent_variables
        interms_dict = OrderedDict({})
        for _, interm_vars in interm_fluent_variables:
            for name in interm_vars:
                interms_dict[name] = np.empty(shape=(len(self),))

        reward_dict = {"reward": np.empty(shape=len(self))}

        done_dict = {"done": np.empty(shape=len(self))}

        for transition in self._trajectory:
            step = transition.step

            state = transition.state
            for fluent_name, state_vars in state_fluent_variables:
                for i, fluent_variable in enumerate(state_vars):
                    fluent_value = np.reshape(state[fluent_name], -1)
                    states_dict[fluent_variable][step] = fluent_value[i]

            action = transition.action
            for fluent_name, action_vars in action_fluent_variables:
                for i, fluent_variable in enumerate(action_vars):
                    fluent_value = np.reshape(action[fluent_name], -1)
                    actions_dict[fluent_variable][step] = fluent_value[i]

            interm = transition.info
            for fluent_name, interm_vars in interm_fluent_variables:
                for i, fluent_variable in enumerate(interm_vars):
                    fluent_value = np.reshape(interm[fluent_name], -1)
                    interms_dict[fluent_variable][step] = fluent_value[i]

            reward = transition.reward
            reward_dict["reward"][step] = reward

            done = transition.done
            done_dict["done"][step] = done

        trajectory_dict = OrderedDict(
            {**states_dict, **actions_dict, **interms_dict, **reward_dict, **done_dict}
        )

        return pd.DataFrame(data=trajectory_dict)

    @property
    def states(self):
        """Returns a dict mapping state fluent name to sequence of values."""
        if not self._trajectory:
            return {}

        fluents = self._trajectory[0].state.keys()
        states_dict = OrderedDict({name: [] for name in fluents})

        for transition in self._trajectory:
            for name, value in transition.state.items():
                states_dict[name].append(value)

        return states_dict

    @property
    def actions(self):
        """Returns a dict mapping action fluent name to sequence of values."""
        if not self._trajectory:
            return {}

        fluents = self._trajectory[0].action.keys()
        actions_dict = OrderedDict({name: [] for name in fluents})

        for transition in self._trajectory:
            for name, value in transition.action.items():
                actions_dict[name].append(value)

        return actions_dict

    @property
    def infos(self):
        """Returns a dict mapping action fluent name to sequence of values."""
        if not self._trajectory:
            return {}

        fluents = self._trajectory[0].info.keys()
        actions_dict = OrderedDict({name: [] for name in fluents})

        for transition in self._trajectory:
            for name, value in transition.info.items():
                actions_dict[name].append(value)

        return actions_dict

    @property
    def rewards(self):
        """Returns list of rewards."""
        if not self._trajectory:
            return []

        rewards_lst = []
        for transition in self._trajectory:
            rewards_lst.append(transition.reward)

        return rewards_lst

    @property
    def initial_state(self):
        """Returns the trajectory's initial state."""
        if not self._trajectory:
            return None

        return self._trajectory[0].state

    @property
    def final_state(self):
        """Returns the trajectory's final state."""
        if not self._trajectory:
            return None

        return self._trajectory[-1].next_state

    @property
    def total_reward(self):
        """Returns the total sum of the trajectory's rewards."""
        return sum(self.rewards)

    def __len__(self):
        return len(self._trajectory)

    def __iter__(self):
        return iter(self._trajectory)

    def __getitem__(self, i):
        return self._trajectory[i]
