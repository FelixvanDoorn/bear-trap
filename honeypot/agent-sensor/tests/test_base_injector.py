# agent-sensor/tests/test_base_injector.py
from pathlib import Path
from typing import Type
from unittest.mock import MagicMock

import pytest
from handlers.base_injector import (
    _HANDLER_CACHE,
    command_dynamic_handler,
    command_factory,
)


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    """Clear the handler cache before every test to prevent cross-test leakage."""
    _HANDLER_CACHE.clear()


@pytest.fixture
def mock_command_instance() -> command_dynamic_handler:
    """Constructs a command instance with mocked Cowrie attributes.

    HoneyPotCommand.__init__ reaches into a live protocol/cmdstack, so we
    bypass it with __new__ and set only the attributes the methods under
    test actually touch.
    """
    cmd = command_dynamic_handler.__new__(command_dynamic_handler)

    # Mock Cowrie's native execution state and protocol context
    cmd.protocol = MagicMock()
    cmd.protocol.log_id = "test-session-12345"
    cmd.protocol.client.host = "192.168.1.50"

    cmd.environ = {"command": "sudo"}
    cmd.args = []
    cmd.logger = MagicMock()

    # Mock execution flow methods from HoneyPotCommand
    cmd.write = MagicMock()
    cmd.exit = MagicMock()

    return cmd


class TestLogToVector:
    def test_builds_expected_payload(
        self, mock_command_instance: command_dynamic_handler
    ) -> None:
        mock_command_instance.log_to_vector("nmap", action_status="spoofed_response")

        assert mock_command_instance.logger.info.called
        _, kwargs = mock_command_instance.logger.info.call_args
        payload = kwargs["payload"]

        assert payload == {
            "event_type": "honeypot_trap_hit",
            "session_id": "test-session-12345",
            "src_ip": "192.168.1.50",
            "command": "nmap",
            "sensor_name": "agent-sensor-01",
            "action": "spoofed_response",
        }

    def test_uses_sensor_name_env_var(
        self,
        mock_command_instance: command_dynamic_handler,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SENSOR_NAME", "custom-sensor-07")

        mock_command_instance.log_to_vector("whoami")

        _, kwargs = mock_command_instance.logger.info.call_args
        assert kwargs["payload"]["sensor_name"] == "custom-sensor-07"

    def test_defaults_action_status_to_intercepted(
        self, mock_command_instance: command_dynamic_handler
    ) -> None:
        mock_command_instance.log_to_vector("ls")

        _, kwargs = mock_command_instance.logger.info.call_args
        assert kwargs["payload"]["action"] == "intercepted"

    def test_unknown_src_ip_when_no_client(
        self, mock_command_instance: command_dynamic_handler
    ) -> None:
        mock_command_instance.protocol.client = None

        mock_command_instance.log_to_vector("id")

        _, kwargs = mock_command_instance.logger.info.call_args
        assert kwargs["payload"]["src_ip"] == "unknown"

    def test_skips_logging_when_no_logger(
        self, mock_command_instance: command_dynamic_handler
    ) -> None:
        mock_command_instance.logger = None

        # Should not raise even though there is nothing to log to
        mock_command_instance.log_to_vector("ps")

    def test_swallows_exceptions(
        self, mock_command_instance: command_dynamic_handler
    ) -> None:
        mock_command_instance.logger.info.side_effect = RuntimeError("boom")

        # Should not propagate, per the broad try/except in log_to_vector
        mock_command_instance.log_to_vector("top")


class TestGetInjectionPrompt:
    def test_uses_mock_output_when_set(
        self, mock_command_instance: command_dynamic_handler
    ) -> None:
        mock_command_instance.mock_output = "custom output\n"

        result = mock_command_instance._get_injection_prompt("nmap")

        assert result.startswith("custom output\n")

    def test_falls_back_to_command_not_found(
        self, mock_command_instance: command_dynamic_handler
    ) -> None:
        mock_command_instance.mock_output = None

        result = mock_command_instance._get_injection_prompt("nmap")

        assert result.startswith("bash: nmap: command not found\n")

    def test_appends_hidden_prompt_contents(
        self,
        mock_command_instance: command_dynamic_handler,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("SECRET_INSTRUCTIONS")
        monkeypatch.setattr("handlers.base_injector.PROMPT_FILE", prompt_file)
        mock_command_instance.mock_output = "output\n"

        result = mock_command_instance._get_injection_prompt("nmap")

        assert result == "output\nSECRET_INSTRUCTIONS"


class TestCall:
    def test_reads_command_from_environ(
        self, mock_command_instance: command_dynamic_handler
    ) -> None:
        mock_command_instance.environ = {"command": "netstat"}
        mock_command_instance.mock_output = "netstat output\n"

        mock_command_instance.call()

        mock_command_instance.write.assert_called_once()
        written: str = mock_command_instance.write.call_args[0][0]
        assert written.startswith("netstat output\n")
        mock_command_instance.exit.assert_called_once()

    def test_falls_back_to_args_when_command_unknown(
        self, mock_command_instance: command_dynamic_handler
    ) -> None:
        mock_command_instance.environ = {"command": "unknown"}
        mock_command_instance.args = ["curl"]
        mock_command_instance.mock_output = None

        mock_command_instance.call()

        written: str = mock_command_instance.write.call_args[0][0]
        assert written.startswith("bash: curl: command not found\n")

    def test_logs_with_spoofed_response_status(
        self, mock_command_instance: command_dynamic_handler
    ) -> None:
        mock_command_instance.environ = {"command": "wget"}
        mock_command_instance.mock_output = "wget output\n"

        mock_command_instance.call()

        _, kwargs = mock_command_instance.logger.info.call_args
        assert kwargs["payload"]["action"] == "spoofed_response"
        assert kwargs["payload"]["command"] == "wget"


class TestCommandFactory:
    def test_returns_subclass_of_command_dynamic_handler(self) -> None:
        handler_cls: Type[command_dynamic_handler] = command_factory(
            "nmap", "nmap output\n"
        )

        assert issubclass(handler_cls, command_dynamic_handler)
        assert handler_cls.mock_output == "nmap output\n"

    def test_sanitizes_class_name(self) -> None:
        handler_cls = command_factory("apt-get", "apt-get output\n")

        assert handler_cls.__name__ == "Command_apt_get"

    def test_dotted_command_name_is_sanitized(self) -> None:
        handler_cls = command_factory("fail2ban.client", "output\n")

        assert handler_cls.__name__ == "Command_fail2ban_client"

    def test_caches_by_command_name(self) -> None:
        first = command_factory("nmap", "first output\n")
        second = command_factory("nmap", "second output\n")

        # Same cached class is returned; the later mock_output is ignored
        assert first is second
        assert first.mock_output == "first output\n"

    def test_different_commands_get_different_classes(self) -> None:
        nmap_cls = command_factory("nmap", "nmap output\n")
        curl_cls = command_factory("curl", "curl output\n")

        assert nmap_cls is not curl_cls
