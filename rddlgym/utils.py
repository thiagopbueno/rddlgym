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


from pyrddl.parser import RDDLParser
from rddl2tf.compiler import Compiler

import rddlgym

import os
import json
from enum import Enum, auto


class Mode(Enum):
    RAW = auto()
    AST = auto()
    SCG = auto()


def read_db():
    dirname = os.path.join(os.path.dirname(rddlgym.__file__), 'files')
    domains = os.path.join(dirname, 'all.json')
    with open(domains, 'r') as file:
        domains = json.loads(file.read())
        return domains


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


def compile_model(filename):
    model = parse_model(filename)
    compiler = Compiler(model)
    return compiler


def load(filename, mode=Mode.AST):
    if mode == Mode.RAW:
        return read_model(filename)
    elif mode == Mode.AST:
        return parse_model(filename)
    elif mode == Mode.SCG:
        return compile_model(filename)
    else:
        raise ValueError('Invalid rddlgym mode: {}'.format(mode))


def make(rddl, mode=Mode.AST):
    if os.path.isfile(rddl):
        return load(rddl, mode)
    else:
        dirname = os.path.join(os.path.dirname(rddlgym.__file__), 'files')
        filename = os.path.join(dirname, '{}.rddl'.format(rddl))
        if not os.path.isfile(filename):
            raise ValueError("Couldn't find RDDL domain: {}".format(rddl))
        return load(filename, mode)


def list_all():
    db = read_db()
    for domain in db:
        print(domain)


def info(rddl):
    db = read_db()
    if rddl not in db:
        raise ValueError("Couldn't find RDDL domain: {}".format(rddl))
    metadata = db[rddl]
    print(rddl)
    print('>> Authors:      {}'.format(', '.join(metadata['authors'])))
    print('>> Date:         {}'.format(metadata['date']))
    print('>> Requirements: {}'.format(', '.join(metadata['requirements'])))
    print('>> Description:')
    print(metadata['description'])


def show(rddl):
    model = make(rddl, mode=Mode.RAW)
    print(model)
