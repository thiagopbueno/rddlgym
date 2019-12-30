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


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use("seaborn")


def plot_rewards(filepath):
    df = pd.read_csv(filepath)
    reward = df.pop("reward")
    stats = reward.describe()

    print(dict(stats))

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)

    ax.plot(reward)
    ax.axhline(stats["mean"], linestyle="--", linewidth=1)
    ax.axhline(stats["max"], linestyle="--", linewidth=1)
    ax.axhline(stats["min"], linestyle="--", linewidth=1)

    bbox = dict(boxstyle="round", fc="0.8")
    for key, y in dict(stats).items():
        if key in ["mean", "max", "min"]:
            ax.annotate(
                s=f"{key}={y:.3f}", xy=(0, y - 0.02), fontweight="bold", bbox=bbox
            )

    ax.set_title("Rewards", fontweight="bold")
    ax.set_xticks(range(len(df)))
    ax.set_xlabel("Timesteps")

    return fig


def plot_all_rewards(dirpath):

    csv_files = []
    for path in os.listdir(dirpath):
        fullpath = os.path.join(dirpath, path)
        if os.path.isdir(fullpath) and path.startswith("run"):
            data = pd.read_csv(os.path.join(fullpath, "data.csv"))
            csv_files.append(data)

    rewards = [df.pop("reward") for df in csv_files]

    df = pd.concat(rewards)
    mean = df.groupby(df.index, sort=False).mean()
    std = df.groupby(df.index, sort=False).std()

    lower, upper = mean - std, mean + std

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)

    ax1.plot(mean, linestyle="-")
    ax1.fill_between(
        mean.index, lower, upper, alpha=0.2,
    )

    ax1.set_title("Avg Rewards Per Episode", fontweight="bold")
    ax1.set_xticks(range(len(mean)))
    ax1.set_xlabel("Timesteps")

    total_rewards = [df.sum() for df in rewards]
    mean, std = np.mean(total_rewards), np.std(total_rewards)

    num_bins = 20
    n, bins, patches = ax2.hist(total_rewards, num_bins, density=True)
    ax2.set_title("Histogram", fontweight="bold")
    ax2.axvline(mean, linestyle="--", linewidth=1)
    ax2.axvline(mean - std, linestyle="--", linewidth=1)
    ax2.axvline(mean + std, linestyle="--", linewidth=1)
    ax2.set_xlabel("Total Reward Per Episode")

    return fig
