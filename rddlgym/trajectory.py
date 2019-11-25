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


# pylint: disable=missing-docstring,too-many-arguments


from collections import OrderedDict, namedtuple


Transition = namedtuple("Transition", "step state action reward next_state info done")


class Trajectory:
    """Trajectory class manages the sequence of state-action-reward
    transitions."""

    def __init__(self):
        self._trajectory = []

    def add_transition(self, step, state, action, reward, next_state, info, done):
        """Adds transition to the trajectory."""
        transition = Transition(step, state, action, reward, next_state, info, done)
        self._trajectory.append(transition)

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
