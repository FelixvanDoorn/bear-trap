# agent-sensor/handlers/__init__.py
#
# Cowrie's shell/protocol.py walks command_modules at class-definition time,
# importing cowrie.commands.<name> for each entry and merging that module's
# own `commands` dict into one combined registry -- that's the only
# mechanism Cowrie actually consults when resolving a typed command to a
# class. A package-level __getattr__ (the previous approach here) is never
# queried by anything in that path, so it never actually intercepted a
# single command.
#
# Keep the real stock list so builtin commands (ls, cat, wget, ...) keep
# working, and append our own submodule last so its entries -- built from
# commands.yaml in injected_commands.py -- override any stock command with
# the same name. netstat, ifconfig, uname, curl, wget, dig, free, and env
# all shadow stock implementations this way; the rest are additions.
command_modules = [
    "adduser",
    "apt",
    "awk",
    "base",
    "base64",
    "bash",
    "busybox",
    "cat",
    "chmod",
    "chpasswd",
    "crontab",
    "curl",
    "cut",
    "dd",
    "dig",
    "du",
    "env",
    "ethtool",
    "find",
    "finger",
    "free",
    "fs",
    "ftpget",
    "gcc",
    "git",
    "groups",
    "ifconfig",
    "iptables",
    "last",
    "locate",
    "ls",
    "lspci",
    "nc",
    "netstat",
    "nohup",
    "perl",
    "ping",
    "python",
    "scp",
    "service",
    "sleep",
    "ssh",
    "su",
    "sudo",
    "tar",
    "tee",
    "tftp",
    "ulimit",
    "uname",
    "uniq",
    "unzip",
    "uptime",
    "wc",
    "wget",
    "which",
    "yum",
    "injected_commands",
]
