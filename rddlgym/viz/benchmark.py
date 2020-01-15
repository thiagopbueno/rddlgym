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


import os
import re
from pprint import pprint
import importlib.util

from bokeh.plotting import figure
from bokeh.models import FactorRange
from bokeh.models import Span
import numpy as np
import pandas as pd
import streamlit as st


def get_tune_config(config_file):
    spec = importlib.util.spec_from_file_location("module.name", config_file)
    experiments = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiments)
    return experiments.CONFIG_TEMPLATE


@st.cache
def get_experiments(logdir, config_file):
    experiments = []

    config = get_tune_config(config_file)

    for dirpath, dirnames, filenames in os.walk(logdir):
        if "config.json" not in filenames:
            continue

        match = re.search(r"(\d+)$", dirpath)
        if not match:
            continue

        trial = int(match.group(1))
        name = config[trial]["name"]

        dirpath = dirpath[: -len(match.group(1))]
        dirpath = dirpath.replace(logdir, "")
        if dirpath.startswith("/"):
            dirpath = dirpath[1:]

        if dirpath.endswith("/"):
            dirpath = dirpath[:-1]

        experiments.append((dirpath, trial, name))

    return sorted(experiments)


@st.cache
def get_csv_filenames(dirpath, trial):
    csv_files = []

    experiment_path = os.path.join(logdir, dirpath, str(trial))

    for path in os.listdir(experiment_path):
        if not path.startswith("run"):
            continue

        filepath = os.path.join(experiment_path, path, "data.csv")
        if os.path.isfile(filepath):
            csv_files.append(filepath)

    return csv_files


@st.cache
def get_reward_data(filepath):
    df = pd.read_csv(filepath)
    return df.pop("reward")


@st.cache
def get_experiments_data(experiments):
    data = {
        "experiment_id": [],
        "dirpath": [],
        "trial": [],
        "name": [],
        "mean": [],
        "std": [],
    }

    for dirpath, trial, name in experiments:
        csv_files = get_csv_filenames(dirpath, trial)
        dataframes = [get_reward_data(csv) for csv in csv_files]
        total_rewards = [df.sum() for df in dataframes]

        data["experiment_id"].append(os.path.join(dirpath, str(trial)))
        data["dirpath"].append(dirpath)
        data["trial"].append(trial)
        data["name"].append(name)
        data["mean"].append(np.mean(total_rewards))
        data["std"].append(np.std(total_rewards))

    return pd.DataFrame(data, index=data["experiment_id"])


def plot_total_reward(data, filter=None, width=800, height=600):
    if filter:
        data = data[data["name"].str.contains(filter, regex=True)]

    factors = [(dirpath, name) for dirpath, name in zip(data["dirpath"], data["name"])]

    p = figure(
        title="Total Rewards",
        y_range=FactorRange(*factors),
        plot_height=height,
        plot_width=width,
    )

    p.title.text_font_size = "18pt"
    p.yaxis.axis_label_text_font_size = "16pt"
    p.yaxis.group_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "11pt"

    p.hbar(y=factors, right=data["mean"], height=0.6)

    mean = data["mean"].mean()
    vline = Span(location=mean, dimension="height", line_color="red", line_width=2)
    p.renderers.append(vline)

    return p


"""
# RDDL VisKit - Benchmark
"""

st.sidebar.subheader("Experiment Config")
config_file = st.sidebar.text_input("Enter a config file (e.g., /path/to/config.py)")

st.sidebar.subheader("Logging Directory")
logdir = st.sidebar.text_input("Enter a logdir (e.g. /path/to/data/dir/)")

st.sidebar.subheader("Filter")
filter_regex = st.sidebar.text_input("Enter a regex expression:")

st.sidebar.subheader("Plotting Configuration")
width = st.sidebar.slider("width", min_value=300, max_value=1200, value=700)
height = st.sidebar.slider("height", min_value=300, max_value=1200, value=700)


def main():
    if not os.path.isdir(logdir) or not os.path.isfile(config_file):
        return

    experiments = get_experiments(logdir, config_file)
    data = get_experiments_data(experiments)
    st.bokeh_chart(plot_total_reward(data, filter_regex, width, height))


main()
