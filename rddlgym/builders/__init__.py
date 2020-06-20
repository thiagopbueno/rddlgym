import functools
import json
import os

import numpy as np


def generate_objs_list(prefix, n):
    return ", ".join(f"{prefix}{i+1}" for i in range(n))


def generate_topology(predicate, obj_prefix, matrix):
    return "\n\t".join(
        f"{predicate}({obj_prefix}{i+1},{obj_prefix}{j+1});"
        for i in range(len(matrix))
        for j in range(len(matrix[0]))
        if matrix[i][j] == 1
    )


def generate_linear_topology(predicate, obj_prefix, n):
    return "\n\t".join(
        f"{predicate}({obj_prefix}{i},{obj_prefix}{i+1});"
        for i in range(n-1))


def generate_predicate_list(predicate, obj_prefix, values):
    return "\n\t".join(
        f"{predicate}({obj_prefix}{i+1}) = {round(value, 2)};"
        for i, value in enumerate(values))


def config(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        self_ = args[0]
        values = f(*args, **kwargs)
        self_._config[f.__name__] = values
        return values

    return wrapper


class RDDLBuilder:

    def __init__(self):
        self._config = {}

    def build(self):
        rddl = f"""{self._domain_section}
{self._non_fluents_section}
{self._instance_section}"""
        return rddl

    def save(self, filepath):
        dirname = os.path.dirname(filepath)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(filepath, "w") as file:
            file.write(self.build())

    def dump_config(self, filepath):
        for key, value in self._config.items():
            value = np.array(value)
            if len(value.shape) == 1:
                value = value[:, np.newaxis]
            self._config[key] = value.tolist()

        config = {
            "module": "reservoir",
            "cls_name": "Reservoir",
            "config": self._config,
            "initial_state": self.initial_state
        }

        with open(filepath, "w") as file:
            file.write(json.dumps(config, indent=None))

    @property
    def _domain_section(self):
        return f"""domain {self.domain_id} {{
    {self.REQUIREMENTS}
    {self.TYPES}
    {self._pvariables}
    {self._cpfs}
    {self._reward}
    {self._action_preconditions}
    {self._state_invariants}
}}
"""

    @property
    def _non_fluents_section(self):
        return f"""non-fluents {self.non_fluents_id} {{
    domain = {self.domain_id};
    objects {{
        {self._objects}
    }};
    non-fluents {{
        {self._non_fluents}
    }};
}}
"""

    @property
    def _instance_section(self):
        return f"""instance {self.instance_id} {{
    domain = {self.domain_id};
    non-fluents = {self.non_fluents_id};
    init-state {{
        {self._init_state}
    }};
    max-nondef-actions = pos-inf;
    horizon = {self.horizon};
    discount = {self.discount}
}}
"""

    @property
    def _pvariables(self):
        return f"""
    pvariables {{
        {self.NONFLUENTS}
        {self.STATEFLUENTS}
        {self.INTERMFLUENTS}
        {self.ACTIONFLUENTS}
    }};"""

    @property
    def _cpfs(self):
        return f"""
    cpfs {{
        {self.INTERMCPFS}
        {self.STATECPFS}
    }};"""

    @property
    def _reward(self):
        return f"""
    reward = {self.REWARD}"""

    @property
    def _action_preconditions(self):
        return f"""
    action-preconditions {{
        {self.ACTIONPRECONDITIONS}
    }};"""

    @property
    def _state_invariants(self):
        return f"""
    state-invariants {{
        {self.STATEINVARIANTS}
    }};"""
