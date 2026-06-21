# /// script
# dependencies = [
#     "pyyaml",
# ]
# ///
from pathlib import Path
import pickle
import sys

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# This will be used to construct our fake bin environment that an attacker will interact with.
PICKLE_PATH = PROJECT_ROOT / "agent-sensor" / "data" / "fs.pickle"
# Use YAML for listing relevant commands to be added to Cowrie Shell environment 
YAML_PATH = SCRIPT_DIR / "commands.yaml"

def main():
    try:
        with open(YAML_PATH, 'r') as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Error: Configuration file missing at '{YAML_PATH}'",  file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML syntax in '{YAML_PATH}':\n{e}",  file=sys.stderr)
        sys.exit(1)

    COMMANDS_TO_ADD = config.get('commands', [])

    # 2. If pickle doesn't exist, build a proper flat layout blueprint
    if not PICKLE_PATH.exists():
        PICKLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Cowrie expects flat dictionaries for each path layer
        base_fs = [
            {'name': '/', 'path': '', 'mode': 16877, 'uid': 0, 'gid': 0, 'size': 0, 'target': None},
            {'name': 'bin', 'path': '/', 'mode': 16877, 'uid': 0, 'gid': 0, 'size': 0, 'target': None},
            {'name': 'usr', 'path': '/', 'mode': 16877, 'uid': 0, 'gid': 0, 'size': 0, 'target': None},
            {'name': 'bin', 'path': '/usr', 'mode': 16877, 'uid': 0, 'gid': 0, 'size': 0, 'target': None},
        ]
        with open(PICKLE_PATH, 'wb') as f:
            pickle.dump(base_fs, f)
        print(f"Initialized a fresh flat filesystem at {PICKLE_PATH}",  file=sys.stderr)

    # 3. Read existing pickle
    try:
        with open(PICKLE_PATH, 'rb') as f:
            fs = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}", file=sys.stderr)
        sys.exit(1)

    # Track existing commands to prevent duplicates
    # In Cowrie, unique files are tracked by combining path + name
    existing_binaries = {item['name'] for item in fs if item.get('path') in ('/bin', '/usr/bin')}

    # 4. Inject commands directly into the main list
    updated = False
    for cmd in COMMANDS_TO_ADD:
        if cmd not in existing_binaries:
            # Append standalone file metadata straight into Cowrie's flat list
            fs.append({
                'name': cmd,
                'path': '/bin',  # Located in /bin
                'mode': 33261,   # Equivalent to standard Linux -rwxr-xr-x 
                'uid': 0,        # root
                'gid': 0,        # root
                'size': 45824,   # Give it a non-zero fake size so it looks real on 'ls -l'
                'target': None
            })
            print(f"Added {cmd} to fake /bin/")
            updated = True

    # 5. Write back out if changes occurred
    if updated:
        with open(PICKLE_PATH, 'wb') as f:
            pickle.dump(fs, f)
        print("Successfully updated fs.pickle!")
    else:
        print("No new commands to add. fs.pickle is up to date.")

if __name__ == "__main__":
    main()