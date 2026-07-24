"""Entrypoint for the entire MLflow project run with remote tracking server
"""
import os
import logging

import mlflow

from fire import Fire

from src import console_handler

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)


def run_project(
        config_file: str,
        experiment_name: str,
        tracking_uri: str = None
):
    """Runs the project specified by the config_file as an experiment under the name experiment_name,
    optionally against a remote tracking server.

    Lazy (skip-unchanged-stages) execution and the determinism guard are configured in the pipeline YAML
    itself (the ``lazy`` / ``ensure_determinism`` keys, see config/hello_world.yaml), backed by the
    MLflow tracking store rather than any external database.

    config_file (str): The path to the configuration file for the experiment
    experiment_name (str): A name for the type of experiment
    tracking_uri (str = None): The connection string to a remote MLflow tracking server
    """
    # input parameters validation
    if tracking_uri is not None:
        # set the remote URI of the tracking server
        msg = f'Using remote MLflow tracking server at {"*" * (len(tracking_uri) - 3) + tracking_uri[-3:]} (redacted)'
        mlflow.set_tracking_uri(tracking_uri)
    else:
        msg = f'Using local MLflow tracking server (localhost)'
    logger.info(msg)

    mlflow.projects.run(
        uri=os.path.dirname(os.path.abspath(__file__)),
        entry_point='src/stages/run.py',
        parameters={'config-file': config_file},
        experiment_name=experiment_name,
        env_manager='local'
    )


if __name__ == '__main__':
    Fire(run_project)
