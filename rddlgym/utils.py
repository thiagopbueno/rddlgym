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

import pyrddl
from pyrddl.parser import RDDLParser

import rddlgym

import os
from enum import Enum, auto


class Mode(Enum):
    RAW = auto()
    AST = auto()


def read_model(filename):
    with open(filename, 'r') as file:
        rddl = file.read()
        return rddl


def parse_model(filename):
    rddl = read_model(filename)
    parser = RDDLParser()
    parser.build()
    model = parser.parse(rddl)
    return model


def make(id, mode=Mode.AST):
    dirname = os.path.join(os.path.dirname(rddlgym.__file__), 'files')
    filename = os.path.join(dirname, '{}.rddl'.format(id))
    if not os.path.isfile(filename):
        raise ValueError('Invalid RDDL id: {}'.format(id))

    if mode == Mode.RAW:
        return read_model(filename)
    elif mode == Mode.AST:
        return parse_model(filename)
    else:
        raise ValueError('Invalid mode: {}'.format(mode))
