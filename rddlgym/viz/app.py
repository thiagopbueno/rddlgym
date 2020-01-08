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
from rddlgym.viz.plot_rewards import plot_rewards_per_run, plot_total_reward_histogram, plot_total_reward_per_run, plot_rewards


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


@st.cache
def get_runs(dirpath):
    runs = []

    if os.path.isdir(dirpath):
        for path in sorted(os.listdir(dirpath)):
            fullpath = os.path.join(dirpath, path)
            if os.path.isdir(fullpath) and path.startswith("run"):
                runs.append(path)

    return runs

@st.cache
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

    rewards_per_run = plot_rewards_per_run(rewards_dict.values())
    total_reward_hist = plot_total_reward_histogram(rewards_dict.values())
    total_rewards_fig = plot_total_reward_per_run(rewards_dict)

    figs = {
        "fluents": traces_by_fluents,
        "objects": traces_by_objects,
        "total_rewards_per_run": total_rewards_fig,
        "total_reward_hist": total_reward_hist,
        "rewards_per_run": rewards_per_run,
    }

    return figs


"""
# RDDL VisKit - Trace Visualizer
"""

st.sidebar.subheader("Logging Directory")
logdir = st.sidebar.text_input("Enter a logdir (e.g. /path/to/data/dir/)")

experiment = st.sidebar.selectbox("Select an experiment", get_experiment_folders(logdir))

st.sidebar.subheader("Views")
view = st.sidebar.radio("Report data", ("average", "per run"))
group_by = st.sidebar.radio("Group by", ("fluents", "objects"))


def main():
    if not logdir:
        return

    if not os.path.isdir(logdir):
        st.error(f"Invalid file directory: {logdir}")
        return

    dirpath = os.path.join(logdir, experiment)

    traces_figs = plot_traces(dirpath)

    """
    ## Performance
    """
    if st.checkbox("Total Rewards Per Run", value=True):
        st.pyplot(traces_figs["total_rewards_per_run"])

    if st.checkbox("Histogram of Total Rewards"):
        st.pyplot(traces_figs["total_reward_hist"])

    if st.checkbox("Rewards Per Run"):
        st.pyplot(traces_figs["rewards_per_run"])

    """
    ## Traces
    """
    if view == "average":
        traces_figs = traces_figs[group_by]

        variables = list(traces_figs.keys())
        pvariables_selection = st.multiselect("Select variables to plot", variables, default=variables)

        for pvariable, fig in traces_figs.items():
            if pvariable in pvariables_selection:
                st.pyplot(fig)
    else:
        runs = get_runs(dirpath)

        if runs:
            run = st.selectbox("Select a run to report", options=runs)
            run_figs = plot_trace_run(dirpath, run)

            if st.checkbox("Rewards"):
                st.pyplot(run_figs["rewards"])

            run_trace_figs = run_figs[group_by]

            variables = list(run_trace_figs.keys())
            pvariables_selection = st.multiselect("Select variables to plot", variables, default=variables)

            for pvariable, fig in run_trace_figs.items():
                if pvariable in pvariables_selection:
                    st.pyplot(fig)


main()
