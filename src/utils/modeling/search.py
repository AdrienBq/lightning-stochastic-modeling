"""Hyperparameter search-space handling for the tuning stage, shared by all three families.

The search space YAML (config/<family>/search_space.yaml) mixes plain values with parameter nodes:
- ``{type: categorical, choices: [...]}``
- ``{type: int, low: ..., high: ...}``
- ``{type: float, low: ..., high: ..., log: true|false}``

:func:`sample_trial` walks the space recursively and replaces every parameter node with a sampled value (random
sampler); :func:`suggest_trial_optuna` does the same through an optuna trial (TPE sampler). After sampling,
:func:`apply_constraints` repairs combinations that are individually valid but jointly meaningless.
"""
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

PARAM_TYPES = ('categorical', 'int', 'float')


def is_parameter_node(node) -> bool:
    """Whether a YAML node is a tunable-parameter specification."""
    return isinstance(node, dict) and node.get('type') in PARAM_TYPES


def _sample_value(node: dict, rng: np.random.Generator):
    kind = node['type']
    if kind == 'categorical':
        choices = node['choices']
        return choices[int(rng.integers(len(choices)))]
    if kind == 'int':
        return int(rng.integers(node['low'], node['high'] + 1))
    low, high = float(node['low']), float(node['high'])
    if node.get('log', False):
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    return float(rng.uniform(low, high))


def sample_trial(space, rng: np.random.Generator):
    """Recursively replace parameter nodes with randomly sampled values (plain values pass through)."""
    if is_parameter_node(space):
        return _sample_value(space, rng)
    if isinstance(space, dict):
        return {key: sample_trial(value, rng) for key, value in space.items()}
    return space


def suggest_trial_optuna(space, optuna_trial, prefix: str = ''):
    """Recursively replace parameter nodes with optuna suggestions (names derived from the YAML path)."""
    if is_parameter_node(space):
        kind = space['type']
        if kind == 'categorical':
            return optuna_trial.suggest_categorical(prefix, space['choices'])
        if kind == 'int':
            return optuna_trial.suggest_int(prefix, space['low'], space['high'])
        return optuna_trial.suggest_float(prefix, space['low'], space['high'], log=space.get('log', False))
    if isinstance(space, dict):
        return {
            key: suggest_trial_optuna(value, optuna_trial, prefix=f'{prefix}.{key}' if prefix else key)
            for key, value in space.items()
        }
    return space


def apply_constraints(trial: dict, *, upstream_model_path: Optional[str] = None) -> dict:
    """Repair jointly-meaningless hyperparameter combinations in a sampled trial (in place, also returned).

    Both rules exist because the search space samples each subtree independently and cannot express a dependency
    between them. Neither is a validity check — every value involved is legal on its own.

    1. ``occurrence_head.loss == focal_bce`` forces ``loss.intensity_weight_gamma = 0``. Focal BCE already carries
       ``positive_class_weight``, and on a BINARY target ``(1 + y)^gamma`` takes only two values ``{1, 2^gamma}``
       — i.e. it IS a positive-class weight. Sampling both makes the effective weight the product of two
       independently-tuned numbers, so the sweep explores one quantity along two axes and the trials table shows
       neither.
    2. An ``upstream-model-path`` forces ``finetuning.enabled = true`` and makes the ``unet:`` block inert. A warm
       start loads the upstream checkpoint's architecture along with its weights, so a sampled architecture would
       be silently discarded; and with phase 1 skipped, ``finetuning.enabled = false`` would leave nothing to run.

    Args:
        trial: A sampled trial configuration; mutated in place.
        upstream_model_path: The stage's ``upstream-model-path``, when the run warm-starts from a checkpoint.

    Returns:
        The same ``trial`` object, repaired.
    """
    occurrence_head = trial.get('occurrence_head', {})
    if occurrence_head.get('enabled') and occurrence_head.get('loss') == 'focal_bce':
        loss = trial.setdefault('loss', {})
        if float(loss.get('intensity_weight_gamma', 0.0)) != 0.0:
            logger.info(
                f'occurrence_head.loss is focal_bce, which already carries positive_class_weight; forcing '
                f'loss.intensity_weight_gamma from {loss["intensity_weight_gamma"]} to 0.0 so the two mechanisms '
                f'do not duplicate.'
            )
            loss['intensity_weight_gamma'] = 0.0

    if upstream_model_path:
        finetuning = trial.setdefault('finetuning', {})
        if not finetuning.get('enabled'):
            logger.info(
                'Warm-starting from an upstream checkpoint, so there is no phase 1 to fall back on; forcing '
                'finetuning.enabled to true.'
            )
            finetuning['enabled'] = True
        if 'unet' in trial:
            logger.info(
                'Warm-starting from an upstream checkpoint: the architecture comes from the checkpoint, so the '
                'sampled `unet` block is ignored for this trial.'
            )
    return trial


def flatten_trial(trial: dict, prefix: str = '') -> dict:
    """Flatten a (nested) trial configuration into dot-separated keys, for tabular trial logs."""
    flat = {}
    for key, value in trial.items():
        path = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_trial(value, prefix=path))
        else:
            flat[path] = value
    return flat
