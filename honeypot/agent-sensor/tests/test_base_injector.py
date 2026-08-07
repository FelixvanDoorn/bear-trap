# agent-sensor/tests/test_base_injector.py
from unittest.mock import MagicMock

import pytest
from handlers.base_injector import (
    _HANDLER_CACHE,
    command_dynamic_handler,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the handler cache before every test to prevent cross-test leakage."""
    _HANDLER_CACHE.clear()


@pytest.fixture
def mock_command_instance():
    """Constructs a command instance with mocked Cowrie attributes."""
    # Instantiate the base class directly for isolated testing
    cmd = command_dynamic_handler(None)

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
