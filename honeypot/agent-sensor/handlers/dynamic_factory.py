# agent-sensor/handlers/dynamic_factory.py
import os

from cowrie.core.honeypot import HoneyPotCommand

class command_dynamic_handler(HoneyPotCommand):
    def start(self):
        # Extract the precise command string that the attacker executed
        cmd_name = self.environ.get('command', 'unknown')
     
        # ─────────────────────────────────────────────────────────
        # PLACE YOUR PROMPT INJECTION / LOGGING HOOKS HERE
        # ─────────────────────────────────────────────────────────
        # This executes instantly when any of your 200+ commands are hit.
        # Examples:
        #   - Save attacker IP & command payload to a database
        #   - Trigger a prompt-injection payload response
        #   - Send an alert webhook to Discord or Slack
        # ─────────────────────────────────────────────────────────
        
        session_id = getattr(self.protocol, 'log_id', 'unknown')
        attacker_ip = "unknown"

        if hasattr(self.protocol, 'client') and self.protocol.client:
            attacker_ip = self.protocol.client.host
        
        # Return a clean, standardized, realistic system error to the attacker
        self.write(f"bash: {cmd_name}: operation not permitted by secure runtime execution layer\n")
        
        # Trigger an exit state with error code 1 to close this command's execution loop
        self.exit(1)