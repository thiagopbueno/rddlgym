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

# pylint: disable=missing-docstring,protected-access,redefined-outer-name


from collections import OrderedDict
import numpy as np
import pytest

from rddlgym import make, GYM
from rddlgym import Runner


HORIZON = 20


@pytest.fixture(scope="module", params=["Navigation-v1"])
def runner(request):
    rddl = request.param
    env = make(rddl, mode=GYM)

    def planner(state, timestep):
        # pylint: disable=unused-argument
        return env.action_space.sample()

    return Runner(env, planner)


def test_run(runner):
    trajectory = runner.run()
    assert len(trajectory) == runner.env.horizon

    for idx, transition in enumerate(trajectory):
        assert transition.step == idx
        assert isinstance(transition.state, OrderedDict)
        assert isinstance(transition.action, OrderedDict)
        assert isinstance(transition.reward, np.float32)
        assert isinstance(transition.next_state, OrderedDict)
        assert isinstance(transition.next_state, OrderedDict)
        assert isinstance(transition.info, OrderedDict)
        assert isinstance(transition.done, bool)

    assert all(not transition.done for transition in trajectory[:-1])
    assert trajectory[-1].done

    assert np.isclose(
        trajectory.total_reward,
        sum(map(lambda transition: transition.reward, trajectory)),
    )
