"""wandb transition logger."""

import wandb


def log(timestep, state, action, reward, next_state, done, info):
    # pylint: disable=too-many-arguments
    """Log transition to wandb current project."""

    del next_state
    del done

    total_reward = info.pop("total_reward")

    fluent_dict = {}
    for fluent_type, fluent in [("state", state), ("action", action), ("info", info)]:
        fluent_dict[fluent_type] = {}

        for name, values in fluent.items():
            name = name[:name.find("/")]

            for i, value in enumerate(values):
                key = f"{name}[{i}]"
                fluent_dict[fluent_type][key] = value

    wandb.log(
        {
            **(fluent_dict["state"]),
            **(fluent_dict["action"]),
            **(fluent_dict["info"]),
            "reward": reward,
            "cost": -reward,
            "cum_total_reward": total_reward,
            "cum_total_cost": -total_reward
        },
        step=timestep
    )


def summary(trajectory, uptime):
    """Log trajectory summary stats."""
    wandb.run.summary["total_reward"] = trajectory.total_reward
    wandb.run.summary["total_cost"] = -trajectory.total_reward
    wandb.run.summary["uptime"] = uptime
