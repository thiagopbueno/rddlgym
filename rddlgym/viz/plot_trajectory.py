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

# pylint: disable=invalid-name,missing-docstring


from collections import defaultdict, OrderedDict
import os

import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import numpy as np
import pandas as pd


plt.style.use("seaborn")


def _get_pvariables_dict(df):
    fluents = defaultdict(lambda: [])
    objects = defaultdict(lambda: [])

    for col in df.columns:
        fluent = col[: col.index("(")]
        fluents[fluent].append(col)

        objs = col[col.index("(") + 1 : -1].split(",")
        for obj in objs:
            objects[obj].append(col)

    return fluents, objects


def _get_colors(df):
    xkcd_colors = mcd.XKCD_COLORS
    color_names = np.random.choice(list(xkcd_colors), len(df.columns), replace=False)
    colors = {
        fluent_name: xkcd_colors[color_name]
        for fluent_name, color_name in zip(df.columns, color_names)
    }
    return colors


def _plot_trace(df, variables, colors, group_by_fluent):

    figs = OrderedDict()

    for i, (pvariable, fluent_vars) in enumerate(variables.items()):

        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
        figs[pvariable] = fig

        for col in fluent_vars:
            values = df[col]

            if group_by_fluent:
                label = col[col.index("(") + 1 : -1]
            else:
                label = col[: col.index("(")]

            color = colors[col]

            ax.plot(values, marker=".", label=label, color=color)

        ax.set_title(pvariable, fontweight="bold")
        ax.legend(loc="lower right", frameon=True)
        ax.set_xticks(range(len(df)))

        if i == len(variables) - 1:
            ax.set_xlabel("Timesteps")

    return figs


def plot_trajectory(df):
    colors = _get_colors(df)
    fluents, objects = _get_pvariables_dict(df)

    fig_by_fluents = _plot_trace(df, fluents, colors, group_by_fluent=True)
    fig_by_objects = _plot_trace(df, objects, colors, group_by_fluent=False)

    return fig_by_fluents, fig_by_objects


def _plot_avg_traces(mean, std, variables, colors, group_by_fluent):

    lower, upper = mean - std, mean + std

    figs = OrderedDict()


    for i, (pvariable, fluent_vars) in enumerate(variables.items()):

        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
        figs[pvariable] = fig

        for col in fluent_vars:

            if group_by_fluent:
                label = col[col.index("(") + 1 : -1]
            else:
                label = col[: col.index("(")]

            color = colors[col]

            ax.plot(mean[col], marker=".", label=label, color=color)
            ax.fill_between(
                mean.index,
                lower[col],
                upper[col],
                alpha=0.2,
                edgecolor=color,
                facecolor=color,
            )

        ax.set_title(pvariable, fontweight="bold")
        ax.legend(loc="lower right", frameon=True)
        ax.set_xticks(range(len(mean)))

        if i == len(variables) - 1:
            ax.set_xlabel("Timesteps")

    return figs


def plot_all_trajectories(dataframes):
    df = pd.concat(dataframes)

    mean = df.groupby(df.index, sort=False).mean()
    std = df.groupby(df.index, sort=False).std()

    colors = _get_colors(df)
    fluents, objects = _get_pvariables_dict(df)

    figs_by_fluents = _plot_avg_traces(mean, std, fluents, colors, group_by_fluent=True)
    figs_by_objects = _plot_avg_traces(mean, std, objects, colors, group_by_fluent=False)

    return figs_by_fluents, figs_by_objects
