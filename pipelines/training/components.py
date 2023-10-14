import os
from typing import Optional

import click
from omegaconf import DictConfig, OmegaConf

from utils.ml_logging import get_logger

# Set up logging
logger = get_logger()


class Context:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg


@click.group()
@click.pass_context
def cli(ctx) -> None:
    """Execute before every command."""
    logger.info("Executing the pipeline component...")


@cli.command()
@click.pass_context
def run_feature_engineering(
    ctx: click.Context,
    input_path: Optional[str],
    output_directory: Optional[str],
    date: Optional[str],
) -> None:
    """
    Run feature engineering process, save resulting datasets to specified directory.
    """


@click.option("--config", type=str, default=None, help="Path to the configuration file")
def main(config: Optional[str] = None) -> None:
    """Entry point of the script."""
    if not config:
        config = "pipelines/configs/feature_engineering/fe.yaml"
    if not os.path.exists(config):
        raise ValueError(f"Configuration file not found at {config}")

    cfg = OmegaConf.load(config)
    cli(obj=Context(cfg))


if __name__ == "__main__":
    main()
