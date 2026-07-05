# agent-sensor/handlers/custom_init.py
import json
import os

# Define the absolute container runtime path to our compiled JSON file
# (Inside Docker, it sits in the same directory as this script)
config_path = os.path.join(os.path.dirname(__file__), 'commands.json')

try:
    with open(config_path, 'r') as f:
        command_list = json.load(f)
except Exception:
    command_list = []

# Cowrie relies on globals prefixed with 'command_' to route string commands to classes.
for cmd in command_list:
    globals()[f"command_{cmd}"] = "cowrie.commands.dynamic_factory.command_dynamic_handler"