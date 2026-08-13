# agent-sensor/handlers/base_injector.py
import os
from pathlib import Path
from typing import Type

from cowrie.shell.command import HoneyPotCommand

_HANDLER_CACHE: dict[str, Type[HoneyPotCommand]] = {}
PROMPT_FILE = Path(__file__).parent / "prompt.txt"


class command_dynamic_handler(HoneyPotCommand):
    # Default mock output if not overridden dynamically by __init__.py
    mock_output: str | None = None
    # Module-level registry to store previously created dynamic class types

    def log_to_vector(self, cmd_name: str, action_status: str = "intercepted") -> None:
        """
        Emits clean telemetry payload using Cowrie's native logger.
        Vector ingests this directly from /var/log/cowrie/cowrie.json
        """
        try:
            session_id = getattr(self.protocol, "log_id", "unknown")
            attacker_ip = "unknown"

            if hasattr(self.protocol, "client") and self.protocol.client:
                attacker_ip = self.protocol.client.host

            telemetry_payload = {
                "event_type": "honeypot_trap_hit",
                "session_id": session_id,
                "src_ip": attacker_ip,
                "command": cmd_name,
                "sensor_name": os.environ.get("SENSOR_NAME", "agent-sensor-01"),
                "action": action_status,
            }

            if hasattr(self, "logger") and self.logger:
                self.logger.info("AGENT_TRAP_HIT: {payload}", payload=telemetry_payload)

        except Exception:
            pass

    def _get_injection_prompt(self, cmd_name: str) -> str:
        """
        Builds the human-invisible, LLM-readable prompt injection string.
        """
        # 1. Standard base output (what humans and basic scripts see)
        base_text = (
            self.mock_output
            if self.mock_output
            else f"bash: {cmd_name}: command not found\n"
        )

        # 2. ANSI Concealed Prompt (\033[30m = Black text, \033[8m = Hidden attribute)
        hidden_prompt = PROMPT_FILE.read_text(encoding="utf-8")

        return base_text + hidden_prompt

    def call(self) -> None:
        """
        The main execution entry point invoked by Cowrie.
        """
        cmd_name = self.environ.get("command", "unknown")
        if cmd_name == "unknown" and hasattr(self, "args") and self.args:
            cmd_name = self.args[0]

        # Log hit to cowrie.json -> Vector -> GCP Pub/Sub
        self.log_to_vector(cmd_name, action_status="spoofed_response")

        # Write stealth payload and exit execution loop
        payload = self._get_injection_prompt(cmd_name)
        self.write(payload)
        self.exit()


def command_factory(cmd_name: str, mock_output: str) -> Type[HoneyPotCommand]:
    """
    Factory creating dynamic command handlers, as specified in __init__.py.
    These are cached, to limit handlers to be unique by command name.
    """
    if cmd_name in _HANDLER_CACHE:
        return _HANDLER_CACHE[cmd_name]

    # Sanitize class name to ensure valid Python identifier syntax
    safe_class_name = f"Command_{cmd_name.replace('-', '_').replace('.', '_')}"

    handler_class = type(
        safe_class_name,
        (command_dynamic_handler,),
        {"mock_output": mock_output},
    )

    _HANDLER_CACHE[cmd_name] = handler_class
    return handler_class
