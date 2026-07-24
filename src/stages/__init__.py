"""Orchestrator for the analysis pipeline. All the workflow is included in a single Python program that looks at the
results of each step and decides what to submit next using custom code.
"""
import os
import sys
import logging

sys.path.append(os.getcwd())

from src import root_path, console_handler
# === imports go below this line ===

# Baseline deterministic seeding: the orchestrator (run.py) derives a per-stage seed from the stage's code
# state + parameters and exports it as PIPELINE_SEED before dispatching each stage subprocess. We seed here,
# on package import, so every stage gets seeded with zero per-stage code. The orchestrator's own import sees
# no PIPELINE_SEED (it is set only just before each stage) and is therefore unaffected.
_pipeline_seed = os.environ.get('PIPELINE_SEED')
if _pipeline_seed is not None:
    try:
        from src.utils.seeding import seed_everything

        _seeded = seed_everything(int(_pipeline_seed))
        _seed_logger = logging.getLogger(__name__)
        _seed_logger.addHandler(console_handler)
        _seed_logger.info('seeding: PIPELINE_SEED=%s applied to %s', _pipeline_seed, ', '.join(_seeded))
    except (ValueError, ImportError) as _seed_error:
        logging.getLogger(__name__).warning('seeding: could not apply PIPELINE_SEED=%r (%s)',
                                             _pipeline_seed, _seed_error)
