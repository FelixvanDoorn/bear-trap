# agent-sensor/handlers/injected_commands.py
#
# Listed last in __init__.py's command_modules, so these entries override
# any stock command with the same name (see __init__.py for why).
import os
from typing import Dict

from .base_injector import command_factory

# commands.yaml isn't real YAML-parsed here (PyYAML isn't in Cowrie's
# bundled venv) -- it's just a flat, hyphen-prefixed list under a
# "commands:" key, so parse that one shape with the standard library. If
# commands.yaml ever needs real YAML features, this needs to become a real
# parser, and PyYAML would need to be baked into the image via a custom
# Dockerfile since it isn't available at runtime.
_config_path = os.path.join(os.path.dirname(__file__), "commands.yaml")

_command_names: list = []
try:
    with open(_config_path, "r") as f:
        _command_names = [
            line.strip()[2:].strip() for line in f if line.strip().startswith("- ")
        ]
except Exception:
    _command_names = []

COMMAND_MAP: Dict[str, str] = {
    cmd: f"{cmd}: operation permitted.\n" for cmd in _command_names
}

commands = {
    cmd: command_factory(cmd, mock_output) for cmd, mock_output in COMMAND_MAP.items()
}
