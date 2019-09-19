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

# pylint: disable=protected-access,missing-docstring,redefined-outer-name


import gym
import numpy as np
import pytest
import tensorflow as tf

import rddl2tf

import rddlgym
from rddlgym.env import RDDLEnv


@pytest.fixture
def rddl():
    return "Navigation-v1"


@pytest.fixture(scope="function")
def env(rddl):
    env_ = rddlgym.make(rddl, mode=rddlgym.GYM)
    yield env_
    env_.close()


def test_init(env):
    assert isinstance(env, RDDLEnv)
    assert env.__class__.__bases__[0] == gym.Env
    assert isinstance(env._compiler, rddl2tf.DefaultCompiler)
    assert env.timestep is None
    assert env.state is None
    assert isinstance(env._sess, tf.Session)
    assert env._sess.graph is env._compiler.graph


def test_observation_space(env):
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert len(env.observation_space.spaces) == len(env._compiler.initial_state_fluents)
    fluents = dict(env._compiler.initial_state_fluents)
    for name, space in env.observation_space.spaces.items():
        assert name in fluents
        assert isinstance(space, gym.spaces.Box)
        assert space.shape == fluents[name].shape.fluent_shape
        assert all(not dim for dim in space.bounded_below)
        assert all(not dim for dim in space.bounded_above)


def test_action_space(env):
    assert isinstance(env.action_space, gym.spaces.Dict)
    assert len(env.action_space.spaces) == len(env._compiler.default_action_fluents)
    fluents = dict(env._compiler.default_action_fluents)
    for name, space in env.action_space.spaces.items():
        assert name in fluents
        assert isinstance(space, gym.spaces.Box)
        assert space.shape == fluents[name].shape.fluent_shape
        assert all(not dim for dim in space.bounded_below)
        assert all(not dim for dim in space.bounded_above)


def test_observation_sample(env):
    obs = env.observation_space.sample()
    assert obs in env.observation_space


def test_action_sample(env):
    action = env.action_space.sample()
    assert action in env.action_space


def test_state_inputs(env):
    _check_tensors(env._state_inputs, dict(env._compiler.initial_state_fluents))


def test_action_inputs(env):
    _check_tensors(env._action_inputs, dict(env._compiler.default_action_fluents))


def test_build_transition_ops(env):
    fluents = dict(env._compiler.initial_state_fluents)
    assert len(env._next_state) == len(fluents)
    for tensor_fluent in env._next_state:
        tensor_scope, fluent_name = tensor_fluent.name.split("/")[:2]
        fluent_name = fluent_name.replace("-", "/")
        assert tensor_scope == "state_cpfs"
        assert fluent_name in fluents
        assert not tensor_fluent.batch


def test_build_reward_ops(env):
    assert isinstance(env._reward, tf.Tensor)
    assert env._reward.dtype == tf.float32
    assert env._reward.shape == (1,)


def test_reset(env):
    state, timestep = env.reset()
    assert timestep == 0
    assert state in env.observation_space


def test_step(env):
    _ = env.reset()
    timestep = env.timestep
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    assert next_state in env.observation_space
    assert env.state is next_state
    assert env.timestep == timestep + 1
    assert isinstance(reward, np.float32)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert len(info) == len(env._compiler.rddl.domain.intermediate_cpfs)


def test_trajectory(env):
    _ = env.reset()
    done = False
    count = 0
    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        count += 1
    assert count == env.horizon


def test_close(env):
    env.close()
    assert env._sess._closed


def _check_tensors(tensors, fluents):
    assert len(tensors) == len(fluents)
    for name, tensor in tensors.items():
        assert name in fluents
        assert isinstance(tensor, tf.Tensor)
        assert tensor.dtype == fluents[name].dtype
        assert tensor.shape == fluents[name].shape.fluent_shape
