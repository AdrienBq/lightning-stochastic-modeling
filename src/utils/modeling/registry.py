"""Checkpoint-driven dispatch between the three model families.

Every module writes a ``module_class`` marker into its checkpoints (``on_save_checkpoint``).
:func:`load_model_module` reads that marker and loads the matching class, so THE shared evaluation stage — and the
upstream-prediction pass of ``prepare_regression`` — can consume a checkpoint of any family without knowing which.

Three resolution paths, in priority order:
  1. an explicit ``model_family`` argument (authoritative; the evaluation stage exposes it as ``--model-family``);
  2. the ``module_class`` checkpoint marker;
  3. a best-effort sniff of the saved ``hyper_parameters`` trial sections, for marker-less checkpoints.

The MC-dropout family is returned wrapped in :class:`~src.utils.modeling.mc_dropout_eval.MCDropoutEnsembleModule`,
which adapts it to the ensemble ``predict_step`` contract so the evaluation stage drives all three identically.
This is the mechanism behind the "one evaluation for all families" invariant: only the per-batch ensemble generator
differs (MC-dropout forward passes vs ODE sampling vs none).
"""
import logging
import os

import torch

from src.utils.modeling.deterministic_module import DeterministicUnetModule
from src.utils.modeling.diffusion_module import DiffusionModule
from src.utils.modeling.mc_dropout_eval import MCDropoutEnsembleModule
from src.utils.modeling.mc_dropout_module import MCDropoutModule

logger = logging.getLogger(__name__)

# marker value (written by each module's on_save_checkpoint, or chosen explicitly / by sniff) -> module class
MODULE_REGISTRY = {
    'deterministic_unet': DeterministicUnetModule,
    'mc_dropout': MCDropoutModule,
    'diffusion': DiffusionModule,
    # legacy marker: checkpoints written before the family rename. Same class, so they still load.
    'distr_regression': DeterministicUnetModule,
}
FAMILY_NAMES = ('deterministic_unet', 'mc_dropout', 'diffusion')
DEFAULT_MODULE_CLASS = 'deterministic_unet'      # marker-less, unsniffable checkpoints are the plain U-net

# families returned wrapped in an evaluation adapter rather than raw: marker -> wrapper factory
EVAL_WRAPPERS = {
    'mc_dropout': MCDropoutEnsembleModule,
}


def _load_checkpoint(checkpoint_path: str) -> dict:
    try:                                          # torch >= 2.6 flips weights_only to True; our own checkpoints
        return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:                             # torch < 1.13 lacks the kwarg
        return torch.load(checkpoint_path, map_location='cpu')


def _sniff_family(checkpoint: dict) -> str:
    """Infer the family of a MARKER-LESS checkpoint from its saved ``hyper_parameters`` trial sections.

    The families carry disjoint trial keys: diffusion has ``flow``, MC-dropout has ``dropout_p``, and the plain
    U-net has neither. ``transformer`` and ``mc_inference`` are the pre-rename names of the first two, kept so a
    checkpoint from the source branches still resolves.

    Best-effort by nature — pass ``model_family`` explicitly when it matters."""
    hyper_parameters = checkpoint.get('hyper_parameters', {}) if isinstance(checkpoint, dict) else {}
    trial = hyper_parameters.get('trial', {}) if isinstance(hyper_parameters, dict) else {}
    if not isinstance(trial, dict):
        return DEFAULT_MODULE_CLASS
    if 'flow' in trial or 'transformer' in trial:
        return 'diffusion'
    if 'dropout_p' in trial or 'mc_inference' in trial:
        return 'mc_dropout'
    return DEFAULT_MODULE_CLASS


def read_module_class_name(checkpoint_path: str, model_family: str = None) -> str:
    """Resolve the family for a checkpoint: explicit ``model_family`` first, then the ``module_class`` marker, then
    a sniff of the saved hyperparameters (defaulting to the U-net module)."""
    if model_family is not None:
        if model_family not in MODULE_REGISTRY:
            raise ValueError(
                f'Unknown model_family "{model_family}" (expected one of {sorted(MODULE_REGISTRY)}).'
            )
        return model_family

    checkpoint = _load_checkpoint(checkpoint_path)
    if isinstance(checkpoint, dict) and checkpoint.get('module_class') is not None:
        name = checkpoint['module_class']
        if name not in MODULE_REGISTRY:
            raise ValueError(
                f'Checkpoint "{checkpoint_path}" declares unknown module_class "{name}" '
                f'(expected one of {sorted(MODULE_REGISTRY)}).'
            )
        return name

    name = _sniff_family(checkpoint)
    logger.info(
        f'Checkpoint "{checkpoint_path}" has no module_class marker; sniffed family "{name}" from its '
        f'hyperparameters. Pass model_family to override if this is wrong.'
    )
    return name


def load_model_module(checkpoint_path: str, map_location='cpu', model_family: str = None):
    """Load the module of the resolved family from a checkpoint.

    Args:
        checkpoint_path: Path to a ``best_model.ckpt`` (or any Lightning checkpoint of a supported family).
        map_location: ``torch.load`` map location for the weights (default CPU).
        model_family: Optional explicit family override (one of :data:`FAMILY_NAMES`); authoritative when given.
            Needed for marker-less checkpoints whose family cannot be sniffed.

    Returns:
        The loaded module, wrapped in its evaluation adapter when the family registers one — MC-dropout comes back
        as an :class:`MCDropoutEnsembleModule` exposing the ensemble ``predict_step`` contract.
    """
    checkpoint_path = os.fspath(checkpoint_path)
    name = read_module_class_name(checkpoint_path, model_family=model_family)
    module_class = MODULE_REGISTRY[name]
    logger.info(f'Loading "{name}" module from checkpoint "{checkpoint_path}".')
    module = module_class.load_from_checkpoint(checkpoint_path, map_location=map_location)

    wrapper = EVAL_WRAPPERS.get(name)
    if wrapper is not None:
        logger.info(f'Wrapping the "{name}" module in {wrapper.__name__} for the shared evaluation contract.')
        return wrapper(module)
    return module
