"""Tests for src/utils/banner.py — the figlet-style startup banner.

Untouched plumber template code that renders ASCII art for the orchestrator's startup line. It has no contract any other
module depends on and no failure mode that could corrupt a result, so the property that governs the whole file is
NEGATIVE: nothing in here may be the reason a pipeline fails to start. Every fallback path is tested for that.

The bundled FIGlet renderer (``_FigFont`` / ``_smush_chars`` / ``_smush_amount`` / ``_render``) is the one part with real
logic. pyfiglet is deliberately NOT installed in this environment, so it is also the part that actually runs — and its
horizontal smushing is what keeps the name inside ``max_width``. A ``_smush_amount`` stuck at 0 would still render
readable art, just 50 % wider, so the tests below pin that smushing HAPPENS rather than only that it does not crash.

Rule-level tests use a stub font rather than the bundled one: ``small`` sets 15 of the smushing bits (equal, lowline,
hierarchy, pair) but NOT big-X or hardblank, so those two rules are unreachable through it.
"""
import pytest

from src.utils import banner
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


def test_the_banner_never_exceeds_the_width_it_was_given():
    """``max_width`` exists because the orchestrator prints this into a terminal. The shrink loop in ``make_banner``
    re-renders with a shortened name until the art fits, so a long name is truncated rather than wrapped."""
    for name in ('lightning-stochastic-modeling', 'mmmmmmmmmmmmmmmmmm', 'x'):
        banner_text = make_banner(name, max_width=40)
        assert max(len(line) for line in banner_text.split('\n')) <= 40 + 2 * 2 + 2, name


# =====================================================================================================================
# The bundled FIGlet renderer (Block 5c)
# =====================================================================================================================
class _StubFont:
    """A font carrying only what ``_smush_chars`` reads, so each smushing rule can be enabled in isolation."""

    def __init__(self, smush, hardblank='$', height=1):
        self.smush = smush
        self.hardblank = hardblank
        self.height = height


_SMUSH = 128                                                     # _SM_SMUSH: the master "smushing allowed" bit


def test_the_bundled_font_parsed_its_header():
    """``_FigFont`` reads the height, hardblank and layout mode out of the ``.flf`` header. Everything downstream is
    driven by those three, and a mis-parsed header degrades silently into scrambled art rather than an error."""
    assert banner._FONT is not None, 'data/fonts/small.flf must be bundled'
    assert banner._FONT.height == 5
    assert banner._FONT.hardblank == '$'
    assert set(banner._FONT.chars) >= set('abcdefghijklmnopqrstuvwxyz-0123456789')
    assert all(len(rows) == banner._FONT.height for rows in banner._FONT.chars.values())


@pytest.mark.parametrize('left,right,expected', [
    (' ', 'x', 'x'),                                             # a blank always yields to its neighbour
    ('x', ' ', 'x'),
])
def test_a_blank_sub_character_always_smushes(left, right, expected):
    assert banner._smush_chars(left, right, _StubFont(_SMUSH), 5, 5) == expected


def test_two_narrow_glyphs_are_never_smushed():
    """The width guard. Sub-2-column glyphs (a space, a full stop) carry no overlap to give away, and smushing them
    would delete the only column they have."""
    assert banner._smush_chars('x', 'y', _StubFont(_SMUSH), prev_w=1, cur_w=5) is None
    assert banner._smush_chars('x', 'y', _StubFont(_SMUSH), prev_w=5, cur_w=1) is None


def test_smushing_disabled_in_the_font_refuses_every_pair():
    assert banner._smush_chars('|', '|', _StubFont(smush=0), 5, 5) is None


@pytest.mark.parametrize('bit,left,right,expected', [
    (1, '|', '|', '|'),                                          # _SM_EQUAL:     identical sub-characters merge
    (2, '_', '|', '|'),                                          # _SM_LOWLINE:   an underscore yields to a border
    (4, '|', '/', '/'),                                          # _SM_HIERARCHY: | < / \ < [] < {} < () < <>
    (8, '[', ']', '|'),                                          # _SM_PAIR:      a matched bracket pair becomes a bar
    (16, '/', '\\', '|'),                                        # _SM_BIGX
    (16, '\\', '/', 'Y'),
    (16, '>', '<', 'X'),
])
def test_each_smushing_rule_fires_only_when_its_bit_is_set(bit, left, right, expected):
    """Both directions in one assertion: with the bit the pair merges, without it the renderer keeps both columns. The
    bundled ``small`` font sets neither big-X nor hardblank, which is why these need a stub.

    The negative baseline carries a DIFFERENT rule bit rather than none at all — with no rule bits the font falls into
    universal overlapping (tested below), where every pair smushes and the negative half would be unfalsifiable."""
    other = 2 if bit == 1 else 1
    assert banner._smush_chars(left, right, _StubFont(_SMUSH | bit), 5, 5) == expected
    assert banner._smush_chars(left, right, _StubFont(_SMUSH | other), 5, 5) is None


def test_a_font_with_no_rule_bits_overlaps_UNIVERSALLY():
    """FIGlet's fallback layout: with smushing enabled but no controlled rule selected, the later glyph simply wins the
    overlapping column. Worth pinning because it is what makes "no rules set" mean *maximally* aggressive rather than
    inactive — the opposite of the natural reading."""
    universal = _StubFont(_SMUSH)
    assert banner._smush_chars('a', 'b', universal, 5, 5) == 'b'
    assert banner._smush_chars('$', 'b', universal, 5, 5) == 'b', 'a hardblank yields to a real sub-character'
    assert banner._smush_chars('a', '$', universal, 5, 5) == 'a'


def test_a_hardblank_is_only_absorbed_when_the_font_allows_it():
    """The hardblank is the font's "this space is load-bearing" marker — the gap inside a ``$`` or between quote marks.
    Merging one away without the bit set would close a gap the glyph needs."""
    assert banner._smush_chars('$', '$', _StubFont(_SMUSH | 32), 5, 5) == '$'
    assert banner._smush_chars('$', 'x', _StubFont(_SMUSH | 1), 5, 5) is None


def test_the_first_glyph_shifts_left_by_its_own_leading_blank_columns():
    """``_smush_amount`` against an empty buffer. This is the term that stops every rendered name from starting with the
    widest glyph's indentation."""
    glyph = ['   x', '  xx', '   x', '   x', '   x']
    amount = banner._smush_amount([''] * 5, glyph, banner._FONT, prev_w=0, cur_w=4)
    assert amount == 2, 'the minimum leading whitespace across the glyph rows'


def test_no_shift_is_offered_when_the_font_forbids_both_smushing_and_kerning():
    glyph = ['   x'] * 5
    assert banner._smush_amount([''] * 5, glyph, _StubFont(smush=0, height=5), 0, 4) == 0


def test_rendering_is_NARROWER_than_laying_the_glyphs_side_by_side():
    """The property the whole smushing implementation exists for, and the one a broken ``_smush_amount`` would fail
    silently: the art would still be readable, just wide enough to trip ``make_banner``'s shrink loop and truncate the
    repository name for no reason."""
    name = 'lightning'
    rows = banner._render(name, banner._FONT)
    naive = sum(banner._FONT.width[character] for character in name)

    assert len(rows) == banner._FONT.height
    assert len({len(row) for row in rows}) == 1, 'all rows must be padded to one width'
    assert max(len(row) for row in rows) < naive, f'no smushing happened: {naive} columns laid out unchanged'


def test_hardblanks_never_reach_the_rendered_output():
    """They are the font's internal marker; leaving one in prints a literal ``$`` in the middle of the banner."""
    assert banner._FONT.hardblank not in ''.join(banner._render('lightning-$-modeling', banner._FONT))


def test_a_character_the_font_lacks_is_SKIPPED_not_rendered_as_a_gap():
    """The glyph table covers printable ASCII only. A repository name with an accent or an emoji must render the rest
    rather than raising a ``KeyError``."""
    assert banner._render('ab', banner._FONT) == banner._render('aéb', banner._FONT)


@pytest.mark.parametrize('rows,expected', [
    ([' ', 'x', ' '], ['x']),
    (['', '', 'a', '', 'b', ''], ['a', '', 'b']),                # interior blanks are part of the art
    ([], ['']),
    (['   ', ''], ['']),                                         # never returns an empty list
])
def test_blank_rows_are_stripped_from_the_ENDS_only(rows, expected):
    """A blank first or last row is the font's ascender/descender space and would put a gap inside the frame; a blank row
    in the middle is part of a glyph."""
    assert banner._strip_blank_rows(rows) == expected


def test_the_art_renderer_uses_the_bundled_font_when_pyfiglet_is_absent():
    import importlib.util

    if importlib.util.find_spec('pyfiglet') is not None:
        pytest.skip('pyfiglet is installed, so the bundled path is not the one taken')
    assert banner._render_art('lightning', 'small') == banner._strip_blank_rows(
        banner._render('lightning', banner._FONT))


def test_an_unknown_font_falls_back_to_the_bundled_one_rather_than_returning_nothing():
    """Requesting any font other than ``small`` needs pyfiglet. Without it the request degrades to ``small`` — the
    banner is cosmetic, so a font that is not installed must not remove the banner."""
    assert banner._render_art('project', 'doh') == banner._render_art('project', 'small')


def test_the_art_renderer_returns_NONE_when_no_renderer_is_available(monkeypatch):
    """The last fallback rung: no pyfiglet and no bundled font. ``None`` is the signal ``make_banner`` uses to print the
    plain name instead — the path a stripped deployment would take."""
    monkeypatch.setattr(banner, '_FONT', None)
    assert banner._render_art('project', 'small') is None


def test_the_banner_degrades_to_PLAIN_TEXT_with_no_font_at_all(monkeypatch):
    """The consequence of the rung above, asserted end to end: a missing ``data/fonts/small.flf`` costs the ASCII art and
    nothing else."""
    monkeypatch.setattr(banner, '_FONT', None)
    plain = make_banner('lightning', version='1.0')

    assert 'lightning' in plain, 'the name must survive as plain text'
    assert '1.0' in plain
    assert plain.startswith('╭') and plain.endswith('╯'), 'the frame is still drawn'
