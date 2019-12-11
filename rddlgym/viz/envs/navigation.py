"""Render function for Navigation problem."""

import matplotlib.pyplot as plt
import numpy as np


def render(env, trajectory):

    start = trajectory.initial_state["location/1"]
    goal = env.non_fluents["GOAL/1"]

    centers = env.non_fluents["DECELERATION_ZONE_CENTER/2"]
    decays = env.non_fluents["DECELERATION_ZONE_DECAY/1"]
    correlations = env.non_fluents["DECELERATION_ZONE_CORRELATION/3"]
    zones = [(x, y, d, c) for (x, y), d, c in zip(centers, decays, correlations)]

    path = trajectory.states["location/1"]
    deltas = trajectory.actions["move/1"]

    _, ax = _create_fig()
    _render_start_and_goal_positions(ax, start, goal)
    _render_deceleration_zones(ax, start, goal, zones)
    _render_path(ax, start, path, deltas)

    plt.legend()
    plt.show()


def _create_fig():
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.grid()
    return fig, ax


def _render_start_and_goal_positions(ax, start, goal):
    ax.plot(
        [start[0]],
        [start[1]],
        marker="X",
        markersize=15,
        color="limegreen",
        label="initial",
    )
    ax.plot(
        [goal[0]], [goal[1]], marker="X", markersize=15, color="crimson", label="goal"
    )


def _render_deceleration_zones(ax, start, goal, zones, npoints=1000):
    lower = (start[0] - 2.0, start[1] - 2.0)
    upper = (goal[0] + 2.0, goal[1] + 2.0)
    X, Y = np.meshgrid(
        np.linspace(lower[0], upper[0], npoints),
        np.linspace(lower[1], upper[1], npoints),
    )

    Lambda = 1.0
    for xcenter, ycenter, decay, correlation in zones:
        a = correlation[0][0]
        b = correlation[1][1]
        ab = correlation[1][0] + correlation[0][1]

        x_diff = np.abs(X - xcenter)
        y_diff = np.abs(Y - ycenter)
        D = np.sqrt(a * x_diff ** 2 + b * y_diff ** 2 + ab * x_diff * y_diff)
        Lambda *= 2 / (1 + np.exp(-decay * D)) - 1.00

    ticks = np.arange(0.0, 1.01, 0.10)

    cp = ax.contourf(X, Y, Lambda, ticks, cmap=plt.cm.bone)
    plt.colorbar(cp, ticks=ticks)
    cp = ax.contour(X, Y, Lambda, ticks, colors="black", linestyles="dashed")


def _render_path(ax, start, path, deltas):
    xpath = [p[0] for p in path]
    ypath = [p[1] for p in path]
    ax.plot(xpath, ypath, "b.", label="states")

    # x0, y0 = start
    xdeltas = [d[0] for d in deltas]
    ydeltas = [d[1] for d in deltas]
    ax.quiver(
        xpath,
        ypath,
        xdeltas,
        ydeltas,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="dodgerblue",
        width=0.005,
        label="actions",
    )
