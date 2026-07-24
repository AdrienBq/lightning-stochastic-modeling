"""Self-contained ASCII-art welcome banner for the pipeline orchestrator.

Renders the repository name in a compact FIGlet font (the bundled ``data/fonts/small.flf``) inside a rounded frame,
with a version/tagline subtitle. The name is derived from the repository directory name, and both the name and
the subtitle are length-capped so the banner never overextends horizontally. If anything goes wrong (missing
font, odd characters) the banner degrades to a plain framed text box rather than disrupting the run.

The bundled FIGlet renderer below (``_FigFont`` / ``_smush_chars`` / ``_smush_amount`` / ``_render``) is a tiny,
dependency-free reimplementation of FIGlet horizontal smushing, so the banner works in any environment with no
extra packages. The control-smushing rules and the layout/amount logic were adapted from:

  * the official FIGfont v2 specification ("FIGfont and FIGdriver standard"):
    http://www.jave.de/figlet/figfont.html  (canonical mirror of figfont.txt)
  * the ``pyfiglet`` reference implementation (Peter Waller et al., a Python port of FIGlet):
    https://github.com/pwaller/pyfiglet  (see ``pyfiglet/__init__.py`` -> ``smushAmount`` / ``smushChars``)

This reimplementation was verified to reproduce ``pyfiglet``'s ``small`` font byte-for-byte across a large
random battery of inputs. The bundled font ``data/fonts/small.flf`` (next to the Inter font used by
``src/utils/plotting``) is "Small" by Glenn Chappell (figlet release 2.1, freely modifiable/redistributable per
the notice in the file's header).

> Tip: install ``pyfiglet`` (``pip install pyfiglet``) for greater customizability — when it is importable,
> ``make_banner(..., font=<name>)`` (and the pipeline's optional ``banner_font`` key) can use *any* of the
> hundreds of FIGlet fonts pyfiglet ships. Without it, only the bundled ``small`` font is available.
"""
import os
import re
import logging

from src import root_path

logger = logging.getLogger(__name__)

# FIGlet smushing-mode bit flags
_SM_EQUAL, _SM_LOWLINE, _SM_HIERARCHY, _SM_PAIR, _SM_BIGX, _SM_HARDBLANK, _SM_KERN, _SM_SMUSH = (
    1, 2, 4, 8, 16, 32, 64, 128
)
_END = re.compile(r'(.)\s*$')

# bundled alongside the Inter font used by src/utils/plotting (see data/fonts/)
_FONT_PATH = os.path.join(root_path, 'data', 'fonts', 'small.flf')


class _FigFont:
    """Minimal FIGlet font: parses the header and the printable-ASCII glyphs (codes 32-126)."""

    def __init__(self, text: str):
        lines = text.split('\n')
        header = re.sub(r'^flf2.', '', lines[0]).split()
        self.hardblank = header[0]
        self.height = int(header[1])
        old_layout, comment = int(header[4]), int(header[5])
        full = int(header[7]) if len(header) > 7 else None
        if full is None:
            full = 64 if old_layout == 0 else (0 if old_layout < 0 else (old_layout & 31) | 128)
        self.smush = full
        data = lines[1 + comment:]
        self.chars, self.width = {}, {}
        i = 0
        for code in range(32, 127):
            rows, end = [], None
            for _ in range(self.height):
                line = data[i]
                i += 1
                if end is None:
                    end = re.compile(re.escape(_END.search(line).group(1)) + r'{1,2}\s*$')
                rows.append(end.sub('', line))
            w = max(len(r) for r in rows)
            self.chars[chr(code)] = [r.ljust(w) for r in rows]
            self.width[chr(code)] = w


def _smush_chars(left, right, font, prev_w, cur_w):
    """Return the smushed sub-character for the overlapping pair, or None if they cannot be smushed."""
    if left == ' ':
        return right
    if right == ' ':
        return left
    if prev_w < 2 or cur_w < 2:
        return None
    if (font.smush & _SM_SMUSH) == 0:
        return None
    if (font.smush & 63) == 0:  # universal overlapping
        if left == font.hardblank:
            return right
        if right == font.hardblank:
            return left
        return right
    if font.smush & _SM_HARDBLANK and left == font.hardblank and right == font.hardblank:
        return left
    if left == font.hardblank or right == font.hardblank:
        return None
    if font.smush & _SM_EQUAL and left == right:
        return left
    rules = ()
    if font.smush & _SM_LOWLINE:
        rules += (('_', r'|/\[]{}()<>'),)
    if font.smush & _SM_HIERARCHY:
        rules += (('|', r'/\[]{}()<>'), (r'\/', '[]{}()<>'), ('[]', '{}()<>'), ('{}', '()<>'), ('()', '<>'))
    for a, b in rules:
        if left in a and right in b:
            return right
        if right in a and left in b:
            return left
    if font.smush & _SM_PAIR:
        for pair in (left + right, right + left):
            if pair in ('[]', '{}', '()'):
                return '|'
    if font.smush & _SM_BIGX:
        if left == '/' and right == '\\':
            return '|'
        if right == '/' and left == '\\':
            return 'Y'
        if left == '>' and right == '<':
            return 'X'
    return None


def _smush_amount(buffer, glyph, font, prev_w, cur_w):
    """How many columns the glyph can shift left into the current buffer."""
    if (font.smush & (_SM_SMUSH | _SM_KERN)) == 0:
        return 0
    max_smush = cur_w
    for row in range(font.height):
        left_line, right_line = buffer[row], glyph[row]
        linebd = len(left_line.rstrip(' ')) - 1
        if linebd < 0:
            linebd = 0
        if linebd < len(left_line):
            ch1 = left_line[linebd]
        else:
            linebd, ch1 = 0, ''
        charbd = len(right_line) - len(right_line.lstrip(' '))
        ch2 = right_line[charbd] if charbd < len(right_line) else ''
        amt = charbd + len(left_line) - 1 - linebd
        if ch1 == '' or ch1 == ' ':
            amt += 1
        elif ch2 != '' and _smush_chars(ch1, ch2, font, prev_w, cur_w) is not None:
            amt += 1
        max_smush = min(max_smush, amt)
    return max_smush


def _render(text, font):
    """Render ``text`` into a list of ``font.height`` strings."""
    buffer = [''] * font.height
    prev_w = 0
    for ch in text:
        glyph = font.chars.get(ch)
        if glyph is None:
            continue
        cur_w = font.width[ch]
        ms = _smush_amount(buffer, glyph, font, prev_w, cur_w)
        for row in range(font.height):
            add_left, add_right = buffer[row], glyph[row]
            for k in range(ms):
                idx = len(add_left) - ms + k
                left = add_left[idx] if 0 <= idx < len(add_left) else ''
                smushed = _smush_chars(left, add_right[k], font, prev_w, cur_w)
                if 0 <= idx < len(add_left) and smushed is not None:
                    add_left = add_left[:idx] + smushed + add_left[idx + 1:]
            buffer[row] = add_left + add_right[ms:]
        prev_w = cur_w
    return [row.replace(font.hardblank, ' ') for row in buffer]


def _strip_blank_rows(rows):
    rows = list(rows)
    while rows and not rows[0].strip():
        rows.pop(0)
    while rows and not rows[-1].strip():
        rows.pop()
    return rows or ['']


# load the bundled font once; tolerate any failure (the banner is non-essential)
try:
    with open(_FONT_PATH, encoding='utf-8', errors='replace') as _handle:
        _FONT = _FigFont(_handle.read())
except Exception as _error:  # pragma: no cover - defensive
    logger.debug('banner: could not load font %s (%s); using plain banner', _FONT_PATH, _error)
    _FONT = None


def _render_art(name: str, font: str):
    """Render ``name`` to ASCII-art rows. Uses pyfiglet (any font) when importable; else the bundled ``small``.

    Returns ``None`` if no renderer is available (caller falls back to plain text).
    """
    # prefer pyfiglet when a non-bundled font is requested, or whenever it is installed (greater customizability)
    if font != 'small' or _FONT is None:
        try:
            import pyfiglet
            return _strip_blank_rows(pyfiglet.Figlet(font=font, width=10000).renderText(name).split('\n'))
        except Exception as error:
            if font != 'small':
                logger.debug('banner: font %r requires pyfiglet (%s); falling back to bundled "small"',
                             font, error)
    if _FONT is not None:
        return _strip_blank_rows(_render(name, _FONT))
    return None


def make_banner(name: str, version: str = '', tagline: str = '', font: str = 'small',
                max_name: int = 18, max_width: int = 64, pad: int = 2) -> str:
    """Build a rounded-frame welcome banner with the name in compact ASCII art and a version/tagline subtitle.

    Args:
        name: The repository / pipeline name (rendered as ASCII art). Capped at ``max_name`` characters.
        version: Optional version string (shown as ``v<version>`` in the subtitle).
        tagline: Optional short description shown after the version.
        font: FIGlet font name. The bundled ``small`` works with no dependencies; any other font requires
            ``pyfiglet`` to be installed (see the module docstring), otherwise it falls back to ``small``.
        max_name: Maximum number of name characters to render (longer names are truncated with ``~``).
        max_width: Hard cap on the inner content width so the banner never overextends horizontally.
        pad: Horizontal padding (spaces) inside the frame.

    Returns:
        The multi-line banner as a single string (no trailing newline).
    """
    name = (name or 'pipeline').strip() or 'pipeline'
    if len(name) > max_name:
        name = name[:max_name - 1] + '~'

    # subtitle: "v<version>  ·  <tagline>" (omit missing parts), truncated to max_width
    parts = []
    if version:
        parts.append(f'v{version}')
    if tagline:
        parts.append(tagline)
    subtitle = '  ·  '.join(parts)
    if len(subtitle) > max_width:
        subtitle = subtitle[:max_width - 1] + '…'

    art = _render_art(name, font)
    if art is not None:
        # shrink the name until the art fits within max_width
        while max(len(line) for line in art) > max_width and len(name) > 1:
            name = name[:len(name) - 2] + '~'
            art = _render_art(name, font)
        art_width = max(len(line) for line in art)
    else:
        art = [name]
        art_width = min(len(name), max_width)

    inner = min(max_width, max(art_width, len(subtitle)))
    rule = '─' * (inner + 2 * pad)
    spacer = ' ' * pad
    out = ['╭' + rule + '╮']
    for line in art:
        out.append(f'│{spacer}{line[:inner].ljust(inner)}{spacer}│')
    if subtitle:
        out.append(f'│{spacer}{" " * inner}{spacer}│')
        out.append(f'│{spacer}{subtitle.ljust(inner)}{spacer}│')
    out.append('╰' + rule + '╯')
    return '\n'.join(out)
