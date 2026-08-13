# bear-trap

Experimental multi-cloud honeypot platform for capturing and analyzing SSH/Telnet attack
traffic. [Cowrie](https://github.com/cowrie/cowrie)-based honeypot nodes run on AWS and
Azure, exposing standard SSH (22) and Telnet (23) publicly while the real management SSH
access sits on a separate, IP-restricted port. Session logs are parsed and enriched by
[Vector](https://vector.dev/) and shipped to a central GCP Pub/Sub topic for downstream
analysis. The Azure node additionally runs a custom command-injection layer that embeds
hidden, LLM-readable prompt injections into command output, aimed at fingerprinting and
probing automated/LLM-driven attackers rather than just human ones. Deployment is handled
by OIDC-authenticated GitHub Actions pipelines that open cloud firewall access to the CI
runner just-in-time and revoke it after each deploy.

## Node structure

```
                  ┌──────────────────────────┐
                  │      GitHub Actions      │
                  │(OIDC auth, CI/CD deploy) │
                  └──────────────────────────┘
                                │
            just-in-time SG/NSG rule, deploy over SSH
              ┌─────────────────┴─────────────────┐
              │                                   │
┌──────────────────────────┐        ┌──────────────────────────┐
│  AWS Control Node (EC2)  │        │ Azure Agent-Sensor (VM)  │
│                          │        │                          │
│ Cowrie: 22, 23 -> shell  │        │ Cowrie: 22, 23 -> shell  │
│    (honeyfs, userdb)     │        │ + prompt-injection layer │
│                          │        │                          │
│  Vector: parse + enrich  │        │  Vector: parse + enrich  │
│    cloud_provider=aws    │        │   cloud_provider=azure   │
└──────────────────────────┘        └──────────────────────────┘
              │                                   │
              ▼                                   ▼
              └─────────────────┬─────────────────┘
                                │
                                ▼
                  ┌──────────────────────────┐
                  │    GCP Pub/Sub topic     │
                  │ (central log ingestion)  │
                  └──────────────────────────┘
                                │
                                ▼
                  ┌──────────────────────────┐
                  │   sink-node (planned)    │
                  │    storage & analysis    │
                  └──────────────────────────┘
```
