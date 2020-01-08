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

import pandas as pd
import streamlit as st

from rddlgym.viz.plot_trajectory import plot_trajectory, plot_all_trajectories
from rddlgym.viz.plot_rewards import plot_rewards, plot_all_rewards, plot_total_rewards


@st.cache
def get_trace_dataframe(filepath):
    df = pd.read_csv(filepath)
    reward = df.pop("reward")
    df.pop("done")
    return df, reward


@st.cache
def get_csv_filenames(dirpath):
    csv_files = []
    for path in os.listdir(dirpath):
        if not path.startswith("run"):
            continue

        filepath = os.path.join(dirpath, path, "data.csv")
        if os.path.isfile(filepath):
            csv_files.append(filepath)

    return csv_files


def get_experiment_folders(logdir):
    folders = {"-"}

    if logdir and os.path.isdir(logdir):
        for root, _, _ in os.walk(logdir):
            match = re.search(r"(run\d+$)", root)
            if match:
                folder = root.replace(logdir, "").replace(match.group(1), "")
                if folder.startswith("/"):
                    folder = folder[1:]
                folders.add(folder)

    if len(folders) > 1:
        folders.remove("-")

    return list(sorted(folders))


def plot_trace_run(dirpath, run):
    filepath = os.path.join(dirpath, run, "data.csv")
    df1, df2 = get_trace_dataframe(filepath)

    trace_by_fluents, trace_by_objects = plot_trajectory(df1)
    rewards = plot_rewards(df2)

    for fig in [trace_by_fluents, trace_by_objects]:
        fig.set_size_inches(width, height, forward=True)

    figs = {
        "fluents": trace_by_fluents,
        "objects": trace_by_objects,
        "rewards": rewards,
    }

    return figs


def plot_traces(dirpath):
    dataframes = []
    rewards_dict = {}

    for filepath in get_csv_filenames(dirpath):
        df1, df2 = get_trace_dataframe(filepath)
        dataframes.append(df1)
        rewards_dict[filepath] = df2

    traces_by_fluents, traces_by_objects = plot_all_trajectories(dataframes)
    rewards_fig = plot_all_rewards(rewards_dict.values())

    total_rewards_fig = plot_total_rewards(rewards_dict)

    for fig in [traces_by_fluents, traces_by_objects]:
        fig.set_size_inches(width, height, forward=True)

    figs = {
        "fluents": traces_by_fluents,
        "objects": traces_by_objects,
        "rewards": rewards_fig,
        "total_rewards": total_rewards_fig,
    }

    return figs


"""
# RDDL VisKit
"""

st.sidebar.subheader("Logging Directory")
logdir = st.sidebar.text_input("Enter a logdir (e.g. /path/to/dir/)")

experiment = st.sidebar.selectbox("Select an experiment", get_experiment_folders(logdir))

st.sidebar.subheader("Views")
traces_view = st.sidebar.checkbox("Average", value=True)
run_view = st.sidebar.checkbox("Run")
group_by = st.sidebar.radio("Group by", ("fluents", "objects"))

st.sidebar.subheader("Plot Config")
width = st.sidebar.slider("Width", min_value=5, max_value=30, value=10)
height = st.sidebar.slider("Height", min_value=5, max_value=30, value=10)


def main():
    if not logdir:
        return

    if not os.path.isdir(logdir):
        st.error(f"Invalid file directory: {logdir}")
        return

    dirpath = os.path.join(logdir, experiment)

    traces_figs = plot_traces(dirpath)

    """
    ## Rewards
    """
    st.pyplot(traces_figs["rewards"])
    st.pyplot(traces_figs["total_rewards"])

    if traces_view:
        """
        ## Traces
        """
        st.pyplot(traces_figs[group_by])

    if run_view:
        if os.path.isdir(dirpath):
            runs = []
            for path in sorted(os.listdir(dirpath)):
                fullpath = os.path.join(dirpath, path)
                if os.path.isdir(fullpath) and path.startswith("run"):
                    runs.append(path)

            if runs:
                """
                ## Run
                """
                run = st.selectbox("Select a run to report", options=runs)
                run_figs = plot_trace_run(dirpath, run)

                st.pyplot(run_figs["rewards"])
                st.pyplot(run_figs[group_by])


main()
