"""Deterministic seeding of the common RNG libraries for pipeline stages.

``seed_everything`` seeds only the libraries that are actually importable in the current environment, so a
stage that does not depend on (say) ``torch`` pays no extra import cost. The orchestrator
(``src/stages/run.py``) derives the seed deterministically from each stage's code state + parameters and
hands it to the stage subprocess via the ``PIPELINE_SEED`` environment variable; the auto-hook in
``src/stages/__init__.py`` reads that variable and calls this function before the stage body runs.

This is a *baseline* seed: a stage that manages its own seeding (e.g. ``tune_distr_regression`` re-seeds per
trial with its own ``seed`` parameter) simply runs after it and overrides it where needed.
"""
import os
import random
import logging
import importlib.util

logger = logging.getLogger(__name__)

# numpy's legacy seed and lightning both require a seed in [0, 2**32 - 1]
_SEED_MODULUS = 2 ** 32


def _available(module_name: str) -> bool:
    """True if ``module_name`` can be imported without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def seed_everything(seed: int) -> list:
    """Seed every supported RNG library that is present in the environment.

    Args:
        seed: The base seed (any non-negative int; reduced modulo 2**32 where a library requires it).

    Returns:
        The list of library names that were actually seeded (for logging). Always includes ``random``.
    """
    seed = int(seed)
    bounded = seed % _SEED_MODULUS
    seeded = []

    # python hash randomization (only takes effect in child processes, but set for completeness) + stdlib RNG
    os.environ['PYTHONHASHSEED'] = str(bounded)
    random.seed(seed)
    seeded.append('random')

    if _available('numpy'):
        import numpy as np
        np.random.seed(bounded)
        seeded.append('numpy')

    # lightning.seed_everything also seeds python/numpy/torch and (with workers=True) the dataloader workers
    if _available('lightning'):
        try:
            import lightning as L
            L.seed_everything(bounded, workers=True)
            seeded.append('lightning')
        except Exception as error:  # pragma: no cover - defensive: never let seeding crash a stage
            logger.debug('seeding: lightning.seed_everything failed (%s)', error)

    if _available('torch'):
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seeded.append('torch')

    return seeded
