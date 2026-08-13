# agent-sensor/handlers/__init__.py
import os
from typing import Any, Dict, Type

import yaml
from cowrie.shell.command import HoneyPotCommand

from .base_injector import command_factory

# Load your custom commands list from YAML on container startup
config_path: str = os.path.join(os.path.dirname(__file__), "commands.yaml")

COMMAND_MAP: Dict[str, str] = {}

try:
    with open(config_path, "r") as f:
        # yaml.safe_load converts YAML lists to Python lists and YAML maps to
        # Python dicts
        data: Any = yaml.safe_load(f)

        # Case 1: Dict with a "commands" key containing a list (Matches your
        # commands.yaml)
        if (
            isinstance(data, dict)
            and "commands" in data
            and isinstance(data["commands"], list)
        ):
            COMMAND_MAP = {
                cmd: f"{cmd}: operation permitted.\n" for cmd in data["commands"]
            }

        # Case 2: Pure dictionary (Command name mapped to custom response text)
        elif isinstance(data, dict):
            COMMAND_MAP = data

        # Case 3: Flat list at root (- nmap \n - netstat)
        elif isinstance(data, list):
            COMMAND_MAP = {cmd: f"{cmd}: operation permitted.\n" for cmd in data}

except Exception:
    COMMAND_MAP = {}


def __getattr__(name: str) -> Type[HoneyPotCommand]:
    """
    Called automatically by Python whenever Cowrie asks for 'Command_<cmd>'
    """
    if name.startswith("Command_"):
        # Strip 'Command_' prefix
        cmd_name = name[8:]

        # Pull standard mock output from commands.yaml, or fall back to generic success
        mock_output: str = COMMAND_MAP.get(
            cmd_name, f"{cmd_name}: execution successful.\n"
        )

        # Dynamically instantiate and return the class with prompt injection
        return command_factory(cmd_name, mock_output)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
