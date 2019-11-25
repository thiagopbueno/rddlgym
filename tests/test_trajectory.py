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

# pylint: disable=missing-docstring,redefined-outer-name,invalid-name,protected-access


import numpy as np
import pytest

from rddlgym import make, GYM, Runner


@pytest.fixture(scope="module", params=["Navigation-v2"])
def trajectory(request):
    rddl = request.param
    env = make(rddl, mode=GYM)

    def planner(state, timestep):
        # pylint: disable=unused-argument
        return env.action_space.sample()

    trajectory = Runner(env, planner).run()
    return trajectory


def test_states(trajectory):
    states = trajectory.states
    assert all(len(fluent) == len(trajectory) for fluent in states.values())
    for fluent, value_lst in states.items():
        for t, value in enumerate(value_lst):
            assert fluent in trajectory[t].state
            assert np.allclose(value, trajectory[t].state[fluent])


def test_actions(trajectory):
    actions = trajectory.actions
    assert all(len(fluent) == len(trajectory) for fluent in actions.values())
    for fluent, value_lst in actions.items():
        for t, value in enumerate(value_lst):
            assert fluent in trajectory[t].action
            assert np.allclose(value, trajectory[t].action[fluent])


def test_interms(trajectory):
    interms = trajectory.infos
    assert all(len(fluent) == len(trajectory) for fluent in interms.values())
    for fluent, value_lst in interms.items():
        for t, value in enumerate(value_lst):
            assert fluent in trajectory[t].info
            assert np.allclose(value, trajectory[t].info[fluent])


def test_rewards(trajectory):
    rewards = trajectory.rewards
    assert len(rewards) == len(trajectory)


def test_total_reward(trajectory):
    total_reward = trajectory.total_reward
    assert np.allclose(total_reward, sum(trajectory.rewards))


def test_as_dataframe(trajectory):
    rddl = trajectory.env._compiler.rddl
    state_vars = rddl.state_fluent_variables
    action_vars = rddl.action_fluent_variables
    interm_vars = rddl.interm_fluent_variables

    state_len = sum(len(fluent_vars) for _, fluent_vars in state_vars)
    action_len = sum(len(fluent_vars) for _, fluent_vars in action_vars)
    interm_len = sum(len(fluent_vars) for _, fluent_vars in interm_vars)

    df = trajectory.as_dataframe()
    assert len(df) == len(trajectory)
    assert len(df.columns) == state_len + action_len + interm_len + 2
