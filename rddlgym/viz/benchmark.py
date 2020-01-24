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

# pylint: disable=missing-docstring


import os
import re
import importlib.util

from bokeh.colors import RGB
from bokeh.models import FactorRange, Span
from bokeh.plotting import figure

import numpy as np
import pandas as pd
import streamlit as st


@st.cache
def get_experiments(logdir):
    experiments = set()

    for dirpath, dirnames, filenames in os.walk(logdir):

        if "config.json" not in filenames:
            continue

        if not dirnames:
            continue

        for dirname in dirnames:

            match = re.search(r"(run\d+)$", dirname)
            if not match:
                continue

            experiment_id = dirpath.replace(logdir, "")
            if experiment_id.startswith("/"):
                experiment_id = experiment_id[1:]

            if experiment_id.endswith("/"):
                experiment_id = experiment_id[:-1]

            experiments.add(experiment_id)

    return sorted(experiments)


@st.cache
def get_csv_filenames(experiment_id):
    csv_files = []

    dirpath = os.path.join(LOGDIR, experiment_id)

    for path in os.listdir(dirpath):
        if not path.startswith("run"):
            continue

        filepath = os.path.join(dirpath, path, "data.csv")
        if os.path.isfile(filepath):
            csv_files.append(filepath)

    return csv_files


@st.cache
def get_reward_data(filepath):
    df = pd.read_csv(filepath)
    return df.pop("reward")


@st.cache
def get_total_rewards_data(experiments):
    total_rewards_data = {
        "experiment_id": [],
        "mean": [],
        "std": [],
    }

    for experiment_id in experiments:
        csv_files = get_csv_filenames(experiment_id)
        dataframes = [get_reward_data(csv) for csv in csv_files]

        total_rewards = [df.sum() for df in dataframes]
        total_rewards_data["experiment_id"].append(experiment_id)
        total_rewards_data["mean"].append(np.mean(total_rewards))
        total_rewards_data["std"].append(np.std(total_rewards))

    index = total_rewards_data["experiment_id"]
    df_total_rewards = pd.DataFrame(total_rewards_data, index=index)

    return df_total_rewards


def plot_total_reward(
    data, colors, group_by=None, filter_regex=None, width=800, height=600
):
    if filter_regex:
        data = data[data["experiment_id"].str.contains(filter_regex, regex=True)]

    if data.empty:
        return None

    factors = []

    for path in data["experiment_id"]:
        folders = path.split("/")

        category_idx, group_by_idx = None, None
        for i, folder in enumerate(folders):
            if category_idx is None and "=" in folder:
                category_idx = i

            if group_by and group_by_idx is None and re.search(f"{group_by}=", folder):
                group_by_idx = i

        category = "/".join(folders[:category_idx])

        if group_by_idx is not None:
            subcategory = folders[group_by_idx]
            del folders[group_by_idx]
            name = "/".join(folders[category_idx:])
            factors.append((category, subcategory, name))
        else:
            name = "/".join(folders[category_idx:])
            factors.append((category, name))

    p = figure(
        title="Total Rewards",
        toolbar_location="above",
        y_range=FactorRange(*factors),
        plot_height=height,
        plot_width=width,
    )

    p.title.text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "9pt"
    p.yaxis.group_text_font_size = "10pt"

    p.hbar(y=factors, right=data["mean"], height=0.6)

    mean_value = data["mean"].mean()
    max_value = data["mean"].max()
    vline1 = Span(
        location=mean_value, dimension="height", line_color="red", line_width=2
    )
    vline2 = Span(
        location=max_value, dimension="height", line_color="green", line_width=2
    )
    p.renderers.extend([vline1, vline2])

    return p


@st.cache
def get_cumulative_rewards_data(experiments):
    cumulative_rewards = {}

    for experiment_id in experiments:
        csv_files = get_csv_filenames(experiment_id)
        dataframes = [get_reward_data(csv) for csv in csv_files]

        s = pd.concat([df.cumsum() for df in dataframes])
        s = s.groupby(s.index, sort=False)
        mean = s.mean().rename("mean")
        std = s.std().rename("std")

        cumulative_rewards[experiment_id] = pd.concat([mean, std], axis=1)

    return cumulative_rewards


def plot_cumulative_reward(data, colors, filter_regex=None, width=800, height=600):
    p = figure(
        title="Cumulative Rewards",
        x_axis_label="Timesteps",
        toolbar_location="above",
        plot_height=height,
        plot_width=width,
        background_fill_color="#fafafa",
    )

    p.title.text_font_size = "12pt"

    for experiment_id, df in data.items():
        if filter is not None and not re.search(filter_regex, experiment_id):
            continue

        mean = df["mean"]
        std = df["std"]
        lower = mean - std
        upper = mean + std

        color = colors[experiment_id]

        index = mean.index
        label = experiment_id
        p.line(x=index, y=mean, line_width=2, color=color, legend_label=label)
        p.varea(x=index, y1=lower, y2=upper, color=color, alpha=0.2)

    return p


@st.cache
def _get_colors(experiments):
    n = len(experiments)

    color_range_rgb = np.arange(1, 256)
    red = np.random.choice(color_range_rgb, size=n, replace=False)
    green = np.random.choice(color_range_rgb, size=n, replace=False)
    blue = np.random.choice(color_range_rgb, size=n, replace=False)

    colors = {}
    for i, (r, g, b) in enumerate(zip(red, green, blue)):
        colors[experiments[i]] = RGB(r, g, b)
    return colors


st.sidebar.subheader("Logging Directory")
LOGDIR = st.sidebar.text_input("Enter a log directory (e.g. /path/to/data/dir/)")

st.sidebar.subheader("Views")
VIEW = st.sidebar.radio("Report data by", ("Total Rewards", "Cumulative Rewards"))

st.sidebar.subheader("Group By")
GROUP_BY = st.sidebar.text_input("Enter a valid hyperparameter:")

st.sidebar.subheader("Filter")
FILTER_REGEX = st.sidebar.text_input("Enter a regex expression:")


# st.sidebar.subheader("Plotting Configuration")
# WIDTH = st.sidebar.slider("width", min_value=300, max_value=1200, value=700)
# HEIGHT = st.sidebar.slider("height", min_value=300, max_value=1200, value=700)


def main():
    st.title("RDDL VisKit - Benchmark")

    if not os.path.isdir(LOGDIR):
        return

    experiments = get_experiments(LOGDIR)
    colors = _get_colors(experiments)

    if VIEW == "Total Rewards":
        data = get_total_rewards_data(experiments)
        p = plot_total_reward(data, colors, GROUP_BY, FILTER_REGEX)
        if p is not None:
            st.bokeh_chart(p)
    else:
        data = get_cumulative_rewards_data(experiments)
        p = plot_cumulative_reward(data, colors, FILTER_REGEX)
        st.bokeh_chart(p)


main()
