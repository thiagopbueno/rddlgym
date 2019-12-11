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


from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import numpy as np


plt.style.use("seaborn")


def plot_trajectory(rddl, df, group_by_fluent=True, group_by_obj=False):
    print(df.describe())
    print(df.mean())
    print(df.min())
    print(df.max())

    reward = df.pop("reward")
    done = df.pop("done")

    objects = defaultdict(lambda: [])
    fluents = defaultdict(lambda: [])
    for col in df.columns:
        fluent = col[: col.index("(")]
        fluents[fluent].append(col)

        objs = col[col.index("(") + 1 : -1].split(",")
        for obj in objs:
            objects[obj].append(col)

    steps = len(df)

    xkcd_colors = mcd.XKCD_COLORS

    color_names = np.random.choice(list(xkcd_colors), len(df.columns), replace=False)
    colors = {
        fluent_name: xkcd_colors[color_name]
        for fluent_name, color_name in zip(df.columns, color_names)
    }

    if group_by_fluent:
        fig, axes = plt.subplots(
            nrows=len(fluents), ncols=1, sharex=True, constrained_layout=True
        )
        fig.suptitle(f"{rddl} (fluent view)", fontweight="bold", fontsize=16)

        for i, (fluent, fluent_vars) in enumerate(fluents.items()):

            for col in fluent_vars:
                values = df[col]
                label = col[col.index("(") + 1 : -1]
                color = colors[col]
                axes[i].plot(values, marker=".", label=label, color=color)

            axes[i].set_title(fluent, fontweight="bold")
            axes[i].legend(loc="lower right")
            axes[i].set_xticks(range(steps))

            if i == len(fluents) - 1:
                axes[i].set_xlabel("Timesteps")

    if group_by_obj:
        fig, axes = plt.subplots(
            nrows=len(objects), ncols=1, sharex=True, constrained_layout=True
        )
        fig.suptitle(f"{rddl} (object view)", fontweight="bold", fontsize=16)

        for i, (obj, fluent_vars) in enumerate(objects.items()):

            for col in fluent_vars:
                values = df[col]
                label = col[: col.index("(")]
                color = colors[col]
                axes[i].plot(values, marker=".", label=label, color=color)

            axes[i].set_title(obj, fontweight="bold")
            axes[i].set_xticks(range(steps))
            axes[i].legend(loc="lower right")

            if i == len(fluents) - 1:
                axes[i].set_xlabel("Timesteps")

    plt.show()


def plot_reward(df):
    pass
