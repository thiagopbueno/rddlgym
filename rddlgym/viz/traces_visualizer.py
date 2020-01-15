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


from collections import defaultdict, OrderedDict
import os
import re

from bokeh.plotting import figure
from bokeh.models import Span
from bokeh.colors import RGB

import numpy as np
import pandas as pd
import streamlit as st


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
    if not os.path.isdir(logdir):
        return ["-"]

    folders = set()

    for root, _, _ in os.walk(logdir):
        match = re.search(r"(run\d+$)", root)
        if match:
            folder = root.replace(logdir, "").replace(match.group(1), "")
            if folder.startswith("/"):
                folder = folder[1:]
            folders.add(folder)

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


def plot_total_reward_per_run(dataframes_dict):
    x = [str(run) for run in np.arange(0, len(dataframes_dict))]
    total_rewards = np.empty((len(dataframes_dict),))

    for filepath, df in dataframes_dict.items():
        run_regex = re.search(r".*/run(.*)/.*", filepath)
        run = int(run_regex.group(1))
        total_rewards[run] = df.sum()

    p = figure(
        title="Total Reward per Episode",
        toolbar_location="above",
        x_axis_label="Runs",
        x_range=x,
        plot_height=400,
        plot_width=700,
        background_fill_color="#fafafa",
    )

    p.title.text_font_size = "14pt"
    p.xaxis.axis_label_text_font_size = "12pt"

    p.vbar(x=x, top=total_rewards, width=0.6)

    mean = np.mean(total_rewards)
    hline = Span(location=mean, dimension="width", line_color="red", line_width=2)
    p.renderers.append(hline)

    return p


def plot_total_reward_histogram(dataframes):
    total_rewards = [df.sum() for df in dataframes]
    mean, std = np.mean(total_rewards), np.std(total_rewards)
    lower = mean - std
    upper = mean + std

    hist, edges = np.histogram(total_rewards, density=True, bins=30)

    p = figure(
        title="Histogram",
        toolbar_location="above",
        x_axis_label="Total Reward per Episode",
        plot_height=400,
        plot_width=700,
        background_fill_color="#fafafa",
    )

    p.title.text_font_size = "14pt"
    p.y_range.start = 0
    p.xaxis.axis_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font_size = "11pt"
    p.yaxis.major_label_text_font_size = "11pt"

    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])
    vline1 = Span(location=mean, dimension="height", line_color="red", line_width=2)
    vline2 = Span(location=lower, dimension="height", line_color="green", line_width=2)
    vline3 = Span(location=upper, dimension="height", line_color="green", line_width=2)
    p.renderers.extend([vline1, vline2, vline3])

    return p


def plot_rewards_per_run(dataframes):
    df = pd.concat(dataframes)
    mean = df.groupby(df.index, sort=False).mean()
    std = df.groupby(df.index, sort=False).std()

    lower, upper = mean - std, mean + std

    p = figure(
        title="Rewards per Run",
        toolbar_location="above",
        x_axis_label="Timesteps",
        plot_height=400,
        plot_width=700,
        background_fill_color="#fafafa",
    )

    p.xaxis.ticker = mean.index

    p.title.text_font_size = "14pt"
    p.xaxis.axis_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font_size = "11pt"
    p.yaxis.major_label_text_font_size = "11pt"

    p.line(x=mean.index, y=mean, line_width=2)
    p.varea(x=mean.index, y1=lower, y2=upper, alpha=0.2)

    return p


def plot_rewards(df):
    stats = df.describe()

    p = figure(
        title="Rewards",
        toolbar_location="above",
        x_axis_label="Timesteps",
        plot_height=400,
        plot_width=700,
        background_fill_color="#fafafa",
    )

    p.xaxis.ticker = df.index

    p.title.text_font_size = "14pt"
    p.xaxis.axis_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font_size = "11pt"
    p.yaxis.major_label_text_font_size = "11pt"

    p.line(x=df.index, y=df, line_width=2)

    return p


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


@st.cache
def _get_colors(columns):
    n = len(columns)

    color_range_rgb = np.arange(1, 256)
    red = np.random.choice(color_range_rgb, size=n, replace=False)
    green = np.random.choice(color_range_rgb, size=n, replace=False)
    blue = np.random.choice(color_range_rgb, size=n, replace=False)
    opacity = np.random.uniform(0.7, 0.9)

    colors = {}
    for i, (r, g, b) in enumerate(zip(red, green, blue)):
        colors[columns[i]] = RGB(r, g, b, opacity)
    return colors


def _plot_trace(df, variables, colors, group_by_fluent):

    figs = OrderedDict()

    for i, (pvariable, fluent_vars) in enumerate(variables.items()):

        p = figure(
            title=pvariable,
            toolbar_location="above",
            plot_height=300,
            plot_width=700,
            background_fill_color="#fafafa",
        )

        figs[pvariable] = p

        p.xaxis.ticker = df.index

        p.title.text_font_size = "14pt"

        for col in fluent_vars:
            values = df[col]

            if group_by_fluent:
                label = col[col.index("(") + 1 : -1]
            else:
                label = col[: col.index("(")]

            color = colors[col]

            p.line(
                x=values.index, y=values, color=color, legend_label=label, line_width=2
            )
            p.circle(x=values.index, y=values, color=color, legend_label=label, size=3)

        if i == len(variables) - 1:
            p.xaxis.axis_label = "Timesteps"
            p.xaxis.axis_label_text_font_size = "12pt"

    return figs


def plot_trajectory(df):
    colors = _get_colors(df.columns)
    fluents, objects = _get_pvariables_dict(df)

    fig_by_fluents = _plot_trace(df, fluents, colors, group_by_fluent=True)
    fig_by_objects = _plot_trace(df, objects, colors, group_by_fluent=False)

    return fig_by_fluents, fig_by_objects


def _plot_avg_traces(mean, std, variables, colors, group_by_fluent):

    lower, upper = mean - std, mean + std

    figs = OrderedDict()

    for i, (pvariable, fluent_vars) in enumerate(variables.items()):

        p = figure(
            title=pvariable,
            toolbar_location="above",
            plot_height=300,
            plot_width=700,
            background_fill_color="#fafafa",
        )

        figs[pvariable] = p

        p.xaxis.ticker = mean.index

        p.title.text_font_size = "14pt"

        for col in fluent_vars:

            if group_by_fluent:
                label = col[col.index("(") + 1 : -1]
            else:
                label = col[: col.index("(")]

            color = colors[col]

            x = mean.index
            y = mean[col]
            y1 = lower[col]
            y2 = upper[col]

            p.line(x=x, y=y, color=color, legend_label=label, line_width=2)
            p.circle(x=x, y=y, color=color, legend_label=label, size=3)
            p.varea(x=x, y1=y1, y2=y2, color=color, alpha=0.2)

        if i == len(variables) - 1:
            p.xaxis.axis_label = "Timesteps"
            p.xaxis.axis_label_text_font_size = "12pt"

    return figs


def plot_all_trajectories(dataframes):
    df = pd.concat(dataframes)

    mean = df.groupby(df.index, sort=False).mean()
    std = df.groupby(df.index, sort=False).std()

    colors = _get_colors(df.columns)
    fluents, objects = _get_pvariables_dict(df)

    figs_by_fluents = _plot_avg_traces(mean, std, fluents, colors, group_by_fluent=True)
    figs_by_objects = _plot_avg_traces(
        mean, std, objects, colors, group_by_fluent=False
    )

    return figs_by_fluents, figs_by_objects


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

experiment = st.sidebar.selectbox(
    "Select an experiment", get_experiment_folders(logdir)
)

st.sidebar.subheader("Performance")
total_reward_checkbox = st.sidebar.checkbox("Total Rewards per Run", value=True)
hist_checkbox = st.sidebar.checkbox("Histogram of Total Rewards")
rewards_per_run_checkbox = st.sidebar.checkbox("Rewards per Run")

st.sidebar.subheader("Views")
view = st.sidebar.radio("Report data by", ("average", "per run"))
group_by = st.sidebar.radio("Group by", ("fluents", "objects"))


def main():
    if not logdir:
        return

    if not os.path.isdir(logdir):
        st.error(f"Invalid file directory: {logdir}")
        return

    dirpath = os.path.join(logdir, experiment)
    traces_figs = plot_traces(dirpath)

    if any([total_reward_checkbox, hist_checkbox, rewards_per_run_checkbox]):
        st.header("Performance")

    if total_reward_checkbox:
        st.bokeh_chart(traces_figs["total_rewards_per_run"])

    if hist_checkbox:
        st.bokeh_chart(traces_figs["total_reward_hist"])

    if rewards_per_run_checkbox:
        st.bokeh_chart(traces_figs["rewards_per_run"])

    """
    ## Traces
    """
    if view == "average":
        traces_figs = traces_figs[group_by]

        variables = list(traces_figs.keys())
        pvariables_selection = st.multiselect(
            "Select variables to plot", variables, default=variables
        )

        for pvariable, fig in traces_figs.items():
            if pvariable in pvariables_selection:
                st.bokeh_chart(fig)
    else:
        runs = get_runs(dirpath)

        if runs:
            run = st.selectbox("Select a run to report", options=runs)
            run_figs = plot_trace_run(dirpath, run)

            if st.checkbox("Rewards"):
                st.bokeh_chart(run_figs["rewards"])

            run_trace_figs = run_figs[group_by]

            variables = list(run_trace_figs.keys())
            pvariables_selection = st.multiselect(
                "Select variables to plot", variables, default=variables
            )

            for pvariable, fig in run_trace_figs.items():
                if pvariable in pvariables_selection:
                    st.bokeh_chart(fig)


main()
