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

    ⚠️ This function repairs the sampled DICT and never loads a checkpoint. ``upstream_model_path`` is read only for
    its truthiness; the weights are loaded by the stage's ``module_factory`` (see ``tuning.py``'s module docstring),
    which is also where the architecture override below is owed.

    Both rules exist because the search space samples each subtree independently and cannot express a dependency
    between them. Neither is a validity check — every value involved is legal on its own.

    **1. ``loss.name == 'focal_bce'`` forces ``loss.intensity_weight_gamma = 0``.** Focal BCE already carries
    ``positive_class_weight``, and on a BINARY target ``intensity_weights(y, gamma) = (1 + y)^gamma`` takes only two
    values ``{1, 2^gamma}`` — i.e. it IS a positive-class weight. Sampling both makes the effective weight the
    product of two independently-tuned numbers, so the sweep explores one quantity along two axes and the trials
    table shows neither.

    ⚠️ The rule is scoped to ``focal_bce`` deliberately, and must NOT be widened to every binary task. A DISTANCE
    loss on the hourly probability field is legitimate — ``rmse`` on a probability is exactly ``sqrt(brier_score)``,
    hence proper — and none of the distance losses carries a positive-class weight of its own. There
    ``intensity_weight_gamma`` is the only reweighting knob available, and collapsing to ``{1, 2^gamma}`` on a binary
    target is precisely what makes it useful rather than redundant.

    This rule is inert in the three daily search spaces, whose ``loss.name`` choices are all distance losses; it
    fires only where ``focal_bce`` is reachable, i.e. an hourly pipeline.

    **2. An ``upstream-model-path`` forces ``finetuning.enabled = true``.** With phase 1 replaced by the upstream
    checkpoint there is nothing to fall back on, so a sampled ``false`` would make the trial a no-op — and worse,
    it would raise: the MC-dropout module builds its ensemble loss only when ``finetuning.enabled`` is set, and its
    ``set_phase('finetune')`` rejects the phase outright otherwise. This rule is what makes a warm-started trial's
    single-phase schedule LEGAL; it does not itself skip phase 1, which is the module's ``training_phases()``.

    In practice the rule fires for ``mc_dropout`` ONLY — it is the only family whose ``tune`` stage takes an
    ``upstream-model-path``, because MC-dropout warm-starts from the upstream's WEIGHTS. Diffusion's upstream sits on
    ``prepare_regression`` instead, where the upstream's PREDICTION is materialised once as the last conditioning
    channel, so its sweep never sees this argument.

    The second thing the warm-start branch does is **log** that the sampled ``unet:`` block will be ignored. That is
    a statement about an obligation, not an enforcement: making it true is the job of the module factory, which must
    build the network from the CHECKPOINT's architecture rather than from ``trial['unet']``. Nothing is removed from
    the trial here — the block is left byte-identical, so the log and the dict deliberately disagree.

    Args:
        trial: A sampled trial configuration; mutated in place.
        upstream_model_path: The stage's ``upstream-model-path``, when the run warm-starts from a checkpoint.

    Returns:
        The same ``trial`` object, repaired.
    """
    loss = trial.get('loss', {})
    if loss.get('name') == 'focal_bce' and float(loss.get('intensity_weight_gamma', 0.0)) != 0.0:
        logger.info(
            f'loss.name is focal_bce, which already carries positive_class_weight; forcing '
            f'loss.intensity_weight_gamma from {loss["intensity_weight_gamma"]} to 0.0 so the two mechanisms do '
            f'not duplicate.'
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
