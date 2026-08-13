"""Tests for src/utils/banner.py — the figlet-style startup banner.

**Thin by design.** Untouched plumber template code that renders ASCII art for the orchestrator's startup line: it has no
contract any other module depends on, and no failure mode that could corrupt a result. The mirror requires a file, so
this is a smoke test plus the one property worth having — that a rendering failure cannot take a pipeline down with it.

Block 5c gives its six functions real tests as part of the every-function requirement; until then this is deliberately
shallow, and says so rather than looking like coverage.
"""
import pytest

from src.utils.banner import make_banner


def test_the_module_imports_and_renders():
    banner = make_banner('lightning-stochastic-modeling')
    assert isinstance(banner, str) and banner.strip()


def test_the_name_influences_the_output():
    """Weak but not vacuous: it rules out a constant string, which is the only way a "renders successfully" test could be
    passing while the renderer does nothing."""
    assert make_banner('aaa') != make_banner('zzz')


def test_the_version_and_tagline_appear_in_the_output():
    banner = make_banner('project', version='1.2.3', tagline='a tagline')
    assert '1.2.3' in banner
    assert 'a tagline' in banner


def test_an_empty_version_and_tagline_are_omitted_cleanly():
    banner = make_banner('project')
    assert isinstance(banner, str)
    assert 'None' not in banner


@pytest.mark.parametrize('name', ['', 'x', 'a-very-long-project-name-with-hyphens', '123'])
def test_awkward_names_do_not_raise(name):
    """The banner runs before anything else in a stage, so a name it cannot render must not be the reason a pipeline
    fails to start."""
    assert isinstance(make_banner(name), str)


def test_an_unknown_font_does_not_take_the_pipeline_down():
    """The property actually worth asserting here. A missing font file is a cosmetic problem; raising would turn it into
    a failed run."""
    try:
        result = make_banner('project', font='definitely-not-a-font')
    except Exception as error:                                   # noqa: BLE001
        pytest.fail(f'an unknown font must not raise: {type(error).__name__}: {error}')
    assert isinstance(result, str)
