import os
import re

from yaml import safe_load


def parse_config(config_path: str) -> dict:
    with open(config_path, 'r') as handle:
        config_contents = handle.read()

    replacements = set(m.group(1) for m in re.finditer(r'{{\$([^}]+)}}', config_contents))
    replacements = {r: os.getenv(r) for r in replacements}
    replacements = {r: v if v is not None else '' for r, v in replacements.items()}

    for k, v in replacements.items():
        config_contents = config_contents.replace('{{$' + k + '}}', v)

    return safe_load(config_contents)

