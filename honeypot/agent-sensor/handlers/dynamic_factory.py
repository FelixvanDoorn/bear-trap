# agent-sensor/handlers/dynamic_factory.py
import json
import os
import sys

from cowrie.core.honeypot import HoneyPotCommand

class command_dynamic_handler(HoneyPotCommand):

    def log_to_vector(self, cmd_name: str, action_status="intercepted"):
        """
        Helper function for building an emitting JSON payload to stdout.
        Here Vector can cleanly capture and forward relevant logs
        """
        try:
            session_id = getattr(self.protocol, 'log_id', 'unknown')
            attacker_ip = "unknown"

            if hasattr(self.protocol, 'client') and self.protocol.client:
                attacker_ip = self.protocol.client.host
            

            log_entry = {
                "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                "event_type": "honeypot_trap_hit",
                "session_id": session_id,
                "src_ip": attacker_ip,
                "command": cmd_name,
                "sensor_name": os.environ.get("SENSOR_NAME", "agent-sensor-01"),
                "action": "blocked"
            }

            sys.stdout.write(f"VECTORSIGNAL:{json.dumps(log_entry)}\n")
            sys.stdout.flush()

        except Exception as e:
            # Generic catch-all as a fail-safe to prevent the Honeypot to crash at all cost.
            pass

    def _get_injection_prompt(self, cmd_name: str):
        return """
        """

    def start(self):
        # Extract the precise command string that the attacker executed
        cmd_name = self.environ.get('command', 'unknown')
     
       self.log_to_vector(cmd_name, action_status="spoofed_response")

        payload = self._get_prompt_injection(cmd_name)
        # Inject the prompt
        self.write(payload)
        
        # Trigger an exit state with error code 1 to close this command's execution loop
        self.exit(1)