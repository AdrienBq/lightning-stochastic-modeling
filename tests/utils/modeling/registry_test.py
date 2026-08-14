"""Tests for src/utils/modeling/registry.py — checkpoint-driven dispatch between the three families.

Ported from branch ``aru-probabilistic-eval``'s ``tests/test_probabilistic_eval_compat.py``, with
``load_regression_module`` renamed to ``load_model_module`` and the sniff key ``mc_inference`` replaced by
``dropout_p``.

**A's ``test_mc_dropout_loads_for_eval_despite_unimplemented_finetune_loss`` is deliberately NOT ported.** It asserted
that ``registry._mc_dropout_eval_overrides`` neutralises a fine-tune loss the vendored code could not build — machinery
Step 3 deleted as verified-safe, because the merged ``losses.py`` implements every config-reachable name so the raise it
worked around cannot occur. The replacement is the positive test at the bottom of this file: the loss that motivated
the workaround now BUILDS, so the workaround's premise is gone.

**The cross-family ``predict_step`` parity test lives here** rather than in a module's own file: making the families
interchangeable is registry.py's job, and the parity is the "one evaluation for all families" invariant itself.
"""
import numpy as np
import pytest
import torch

from src.utils.modeling.deterministic_module import DeterministicUnetModule
from src.utils.modeling.diffusion_module import DiffusionModule
from src.utils.modeling.mc_dropout_eval import MCDropoutEnsembleModule
from src.utils.modeling.mc_dropout_module import MCDropoutModule
from src.utils.modeling.registry import (
    DEFAULT_MODULE_CLASS, FAMILY_NAMES, MODULE_REGISTRY, load_model_module, read_module_class_name,
)

MEMBERS = 4
OCCURRENCE_EVENT = (0.0, True)
ENSEMBLE_KEYS = {'prediction', 'ensemble_members', 'probability', 'observation', 'ensemble_partials'}


@pytest.fixture
def checkpoints(unet_trial, mc_trial, diffusion_trial, normalization, target_stats, save_checkpoint):
    """One saved checkpoint per family, written the way Lightning would."""
    def build():
        return {
            'deterministic_unet': save_checkpoint(
                DeterministicUnetModule(unet_trial(), 5, target_stats(), normalization), 'det.ckpt'),
            'mc_dropout': save_checkpoint(
                MCDropoutModule(mc_trial(), 5, target_stats(), normalization), 'mc.ckpt'),
            'diffusion': save_checkpoint(
                DiffusionModule(diffusion_trial(), 5, target_stats(), normalization), 'diff.ckpt'),
        }
    return build


# =====================================================================================================================
# The registry itself
# =====================================================================================================================
def test_every_family_is_registered():
    assert set(FAMILY_NAMES) <= set(MODULE_REGISTRY)
    assert set(FAMILY_NAMES) == {'deterministic_unet', 'mc_dropout', 'diffusion'}


def test_the_legacy_marker_alias_still_resolves():
    """Checkpoints written before the family rename must keep loading — same class, different recorded name."""
    assert MODULE_REGISTRY['distr_regression'] is MODULE_REGISTRY['deterministic_unet']


def test_the_default_family_is_the_plain_unet():
    assert DEFAULT_MODULE_CLASS == 'deterministic_unet'


def test_an_unknown_explicit_family_raises_listing_the_valid_ones(checkpoints):
    paths = checkpoints()
    with pytest.raises(ValueError, match='mc_dropout'):
        read_module_class_name(paths['deterministic_unet'], model_family='transformer_v2')


# =====================================================================================================================
# Dispatch: by explicit family, by marker, by sniff  (ported)
# =====================================================================================================================
@pytest.mark.parametrize('family', ['deterministic_unet', 'mc_dropout', 'diffusion'])
def test_each_family_round_trips_through_its_own_marker(family, checkpoints):
    """Every module writes ``module_class`` in ``on_save_checkpoint``, and the registry reads it back. This is what
    lets the shared evaluation stage consume a checkpoint of any family without being told which."""
    paths = checkpoints()
    assert read_module_class_name(paths[family]) == family


@pytest.mark.parametrize('family', ['deterministic_unet', 'mc_dropout', 'diffusion'])
def test_an_explicit_family_overrides_the_marker(family, checkpoints):
    """The evaluation stage exposes this as ``--model-family``, and it is authoritative — needed for marker-less
    checkpoints whose family cannot be sniffed."""
    paths = checkpoints()
    assert read_module_class_name(paths['deterministic_unet'], model_family=family) == family


def test_mc_dropout_comes_back_wrapped_in_its_ensemble_adapter(checkpoints):
    """The wrapping is the mechanism behind "one evaluation for all families": only the per-batch ensemble generator
    differs between the three, so MC-dropout is adapted rather than special-cased downstream."""
    loaded = load_model_module(checkpoints()['mc_dropout'])
    assert isinstance(loaded, MCDropoutEnsembleModule)
    assert loaded.expected_in_channels == 5


@pytest.mark.parametrize('family,expected', [('deterministic_unet', DeterministicUnetModule),
                                            ('diffusion', DiffusionModule)])
def test_the_unwrapped_families_come_back_raw(family, expected, checkpoints):
    loaded = load_model_module(checkpoints()[family])
    assert isinstance(loaded, expected)
    assert not isinstance(loaded, MCDropoutEnsembleModule)


@pytest.mark.parametrize('trial_key,expected', [
    ('flow', 'diffusion'), ('transformer', 'diffusion'),
    ('dropout_p', 'mc_dropout'), ('mc_inference', 'mc_dropout'),
])
def test_a_marker_less_checkpoint_is_sniffed_from_its_trial_keys(trial_key, expected, tmp_path):
    """Best-effort fallback for checkpoints written before the marker existed. The families carry disjoint trial keys;
    ``transformer`` and ``mc_inference`` are the pre-rename names, kept so a source-branch checkpoint still resolves."""
    import os

    path = os.path.join(str(tmp_path), f'{trial_key}.ckpt')
    torch.save({'state_dict': {}, 'hyper_parameters': {'trial': {trial_key: {}}}}, path)
    assert read_module_class_name(path) == expected


def test_a_marker_less_checkpoint_with_no_signal_falls_back_to_the_unet(tmp_path):
    import os

    path = os.path.join(str(tmp_path), 'bare.ckpt')
    torch.save({'state_dict': {}, 'hyper_parameters': {'trial': {'unet': {}}}}, path)
    assert read_module_class_name(path) == DEFAULT_MODULE_CLASS


def test_a_checkpoint_declaring_an_unknown_marker_raises(tmp_path):
    """Better than falling back silently: an unrecognised marker means the checkpoint came from code this repo does not
    have, and scoring it as a U-net would produce plausible nonsense."""
    import os

    path = os.path.join(str(tmp_path), 'weird.ckpt')
    torch.save({'state_dict': {}, 'module_class': 'quantile_gan'}, path)
    with pytest.raises(ValueError, match='quantile_gan'):
        read_module_class_name(path)


# =====================================================================================================================
# THE cross-family contract: both stochastic families return the same predict_step dict
# =====================================================================================================================
def test_both_stochastic_families_return_an_identical_contract(
        mc_trial, diffusion_trial, normalization, target_stats, batch):
    """Identical KEYS and identical shapes, so everything downstream — baselines, the metric suite, the report — is
    genuinely shared code rather than two paths that happen to agree today."""
    x, y = batch()

    mc = MCDropoutEnsembleModule(MCDropoutModule(mc_trial(), 5, target_stats(), normalization).eval())
    mc.eval_ensemble_size = MEMBERS
    mc.eval_occurrence_event = OCCURRENCE_EVENT

    diffusion = DiffusionModule(diffusion_trial(), 5, target_stats(), normalization).eval()
    diffusion.eval_ensemble_size = MEMBERS
    diffusion.eval_occurrence_event = OCCURRENCE_EVENT
    diffusion.eval_sampling_steps = 2

    with torch.no_grad():
        mc_output = mc.predict_step((x, y), 0)
        diffusion_output = diffusion.predict_step((x, y), 0)

    assert set(mc_output) == ENSEMBLE_KEYS
    assert set(diffusion_output) == ENSEMBLE_KEYS
    for output in (mc_output, diffusion_output):
        assert output['prediction'].shape == (x.shape[0], x.shape[2], x.shape[3])
        assert output['ensemble_members'].shape == (x.shape[0], MEMBERS, x.shape[2], x.shape[3])
        assert output['observation'].shape == (x.shape[0], x.shape[2], x.shape[3])
    # same partials structure -> finalize_ensemble_metrics treats them identically
    assert set(mc_output['ensemble_partials']) == set(diffusion_output['ensemble_partials'])


def test_the_deterministic_family_returns_no_ensemble_keys(unet_trial, normalization, target_stats, batch):
    """A deterministic run must NOT claim an ensemble: ``ensemble_members`` present with one member would make
    ``spread_skill_sums`` return NaN through ``ddof=1`` and the report draw a 6-panel layout from a point forecast."""
    module = DeterministicUnetModule(unet_trial(), 5, target_stats(), normalization).eval()
    x, y = batch()
    with torch.no_grad():
        output = module.predict_step((x, y), 0)
    assert set(output) == {'prediction', 'probability', 'observation'}
    assert 'ensemble_members' not in output


# =====================================================================================================================
# The replacement for A's deleted _mc_dropout_eval_overrides test
# =====================================================================================================================
def test_the_loss_that_motivated_the_eval_override_now_builds():
    """A carried ``registry._mc_dropout_eval_overrides`` solely because its VENDORED ``build_finetune_loss`` raised on
    ``afcrps_psd``, so an MC-dropout checkpoint trained with that loss could not be loaded for evaluation at all. The
    merged ``losses.py`` implements every config-reachable name, so the override was deleted as unnecessary rather than
    kept as a safety net — and this is the assertion that makes that deletion safe rather than hopeful."""
    from src.utils.modeling.losses import ENSEMBLE_LOSSES, build_ensemble_loss

    assert 'afcrps_psd' in ENSEMBLE_LOSSES
    loss = build_ensemble_loss({'enabled': True, 'loss': 'afcrps_psd', 'loss_weight': 1.0, 'beta': 0.7,
                                'alpha': 0.8, 'samples': 4})
    assert callable(loss)


def test_the_loader_is_RENAMED_to_load_model_module():
    """``load_regression_module`` named the one family that existed when it was written. It is now the single dispatch
    point for all three, and the name was the last place the old framing survived in code."""
    from src.utils.modeling import registry

    assert not hasattr(registry, 'load_regression_module')
    assert callable(registry.load_model_module)


@pytest.mark.source_invariant
def test_no_config_and_no_doc_still_names_the_OLD_loader(repo_root):
    """The rename had to reach the documents too: ``CLAUDE.md`` lists this function in its design invariants, so a stale
    name there sends the next reader looking for a function that does not exist."""
    import glob
    import os

    offenders = [path for path in glob.glob(os.path.join(repo_root, 'config/**/*.yaml'), recursive=True)
                 if 'load_regression_module' in open(path).read()]
    assert not offenders, offenders
    assert 'load_model_module' in open(os.path.join(repo_root, 'CLAUDE.md')).read()


def test_the_eval_override_helper_is_gone():
    """Pinned so it cannot creep back: the workaround and the raise it worked around must be removed together, or the
    next reader will assume the raise is still possible."""
    import src.utils.modeling.registry as registry

    assert not hasattr(registry, '_mc_dropout_eval_overrides')


def test_an_mc_dropout_checkpoint_trained_with_that_loss_loads_and_predicts(
        mc_trial, normalization, target_stats, save_checkpoint, batch):
    """The end-to-end version of the above: save a module whose finetune loss is the once-unbuildable name, then load
    it through the registry and run the ensemble predict_step."""
    trial = mc_trial()
    trial['finetuning'] = {**trial['finetuning'], 'loss': 'afcrps_psd', 'alpha': 0.8}
    path = save_checkpoint(MCDropoutModule(trial, 5, target_stats(), normalization), 'mc_afcrps.ckpt')

    loaded = load_model_module(path, model_family='mc_dropout')
    assert isinstance(loaded, MCDropoutEnsembleModule)
    loaded.eval_ensemble_size = MEMBERS
    loaded.eval_occurrence_event = OCCURRENCE_EVENT
    x, y = batch()
    with torch.no_grad():
        output = loaded.predict_step((x, y), 0)
    assert output['ensemble_members'].shape == (x.shape[0], MEMBERS, x.shape[2], x.shape[3])


# =====================================================================================================================
# Block 5c — the two loader internals
# =====================================================================================================================
def test_a_checkpoint_is_loaded_with_weights_only_DISABLED(unet_trial, normalization, target_stats,
                                                           save_checkpoint):
    """torch >= 2.6 flips ``weights_only`` to True by default, which refuses to unpickle anything but tensors. Our own
    checkpoints carry ``hyper_parameters`` — the trial dict, ``target_stats``, the normalization — so the default would
    make every checkpoint in the repo unloadable, and the family marker unreadable with it."""
    from src.utils.modeling.registry import _load_checkpoint

    module = DeterministicUnetModule(unet_trial(), 5, target_stats(), normalization)
    checkpoint = _load_checkpoint(save_checkpoint(module))

    assert 'state_dict' in checkpoint
    assert 'hyper_parameters' in checkpoint, 'the non-tensor payload must survive the load'
    assert 'trial' in checkpoint['hyper_parameters']


@pytest.mark.parametrize('trial,expected', [
    ({'flow': {}}, 'diffusion'),
    ({'transformer': {}}, 'diffusion'),                    # the pre-rename name, kept for source-branch checkpoints
    ({'dropout_p': 0.2}, 'mc_dropout'),
    ({'mc_inference': {}}, 'mc_dropout'),                  # ditto
    ({'unet': {}}, 'deterministic_unet'),
    ({}, 'deterministic_unet'),
])
def test_the_family_is_sniffed_from_the_DISJOINT_trial_keys(trial, expected):
    """The legacy fallback for a marker-less checkpoint. It works only because the three families' trial vocabularies
    are disjoint — ``flow`` for diffusion, ``dropout_p`` for MC-dropout, neither for the plain U-net. Adding a
    ``flow`` block to another family would silently reroute its checkpoints."""
    from src.utils.modeling.registry import _sniff_family

    assert _sniff_family({'hyper_parameters': {'trial': trial}}) == expected


@pytest.mark.parametrize('checkpoint', [
    {},                                                    # no hyper_parameters at all
    {'hyper_parameters': {}},                              # no trial
    {'hyper_parameters': {'trial': 'not-a-dict'}},         # a trial of the wrong type
    {'hyper_parameters': 'not-a-dict'},
    'not-a-dict-at-all',
])
def test_a_MALFORMED_checkpoint_sniffs_to_the_default_rather_than_raising(checkpoint):
    """Best-effort by design — it runs only when the marker is absent, which already means the checkpoint predates the
    registry. Raising here would make an old checkpoint unloadable rather than merely mis-typed, and the explicit
    ``model_family`` argument is the documented way out."""
    from src.utils.modeling.registry import DEFAULT_MODULE_CLASS, _sniff_family

    assert _sniff_family(checkpoint) == DEFAULT_MODULE_CLASS


def test_the_sniff_prefers_DIFFUSION_when_a_trial_somehow_carries_both():
    """Not reachable from any shipped search space, but pinned because the order of the two tests IS the tie-break: a
    residual diffusion trial has no ``dropout_p``, so if one ever appeared the flow block is the stronger signal."""
    from src.utils.modeling.registry import _sniff_family

    assert _sniff_family({'hyper_parameters': {'trial': {'flow': {}, 'dropout_p': 0.2}}}) == 'diffusion'
