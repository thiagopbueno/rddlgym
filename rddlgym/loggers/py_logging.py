"""Basic transition logger."""

import logging


def set_logger(filename, filemode="w"):
    """Set basic config."""
    logging.basicConfig(filename=filename, filemode=filemode, level=logging.INFO)


def log(timestep, state, action, reward, next_state, done, info):
    # pylint: disable=too-many-arguments
    """Log transition to file."""

    del next_state
    del done

    logging.info(f">> timestep={timestep}")

    for fluent in [state, action, info]:
        for name, values in fluent.items():
            values = ", ".join(f"{val:.3f}" for val in values)
            logging.info(f"  {name}=[{values}]")

    logging.info(f"  reward={reward}")


def summary(trajectory, uptime):
    """Log trajectory summary stats."""
    logging.info(f">> total_reward={trajectory.total_reward}")
    logging.info(f">> uptime={uptime}")
