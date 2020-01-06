# rddlgym [![Build Status](https://travis-ci.org/thiagopbueno/rddlgym.svg?branch=master)](https://travis-ci.org/thiagopbueno/rddlgym) [![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/thiagopbueno/rddlgym/blob/master/LICENSE)


A toolkit for working with RDDL domains in Python3.

# Quickstart

```text
$ pip3 install rddlgym
```

# Usage

```text
$ rddlgym --help
Usage: rddlgym [OPTIONS] COMMAND [ARGS]...

  rddlgym: A toolkit for working with RDDL domains in Python3.

Options:
  --help  Show this message and exit.

Commands:
  info   Print metadata for a `rddl` domain/instance.
  ls     List all RDDL domains and instances available.
  parse  Check RDDL file parsing.
  run    Run random policy in `rddl` domain/instance.
  show   Print `rddl` file.
  viz    Visualize simulated trajectories.
```

# License

Copyright (c) 2018-2020 Thiago Pereira Bueno All Rights Reserved.

rddlgym is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

rddlgym is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with rddlgym. If not, see http://www.gnu.org/licenses/.
