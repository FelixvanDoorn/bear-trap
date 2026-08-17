-- Query-friendly view over raw_events.
--
-- raw_events.data holds each Cowrie/Vector event whole, as JSON -- ingestion
-- never fails or drops fields no matter how event shapes vary (see
-- deploy-sink.sh for why). This view promotes the fields that are actually
-- worth filtering/grouping by into real typed columns, while `raw` keeps
-- the full original payload available for anything not promoted here.
--
-- Applied by deploy-sink.sh via `bq query`. Safe to re-run: CREATE OR REPLACE.
CREATE OR REPLACE VIEW `mineral-droplet-160709.bear_trap_logs.logs` AS
SELECT
  publish_time,
  SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*SZ', JSON_VALUE(data, '$.timestamp')) AS event_timestamp,
  JSON_VALUE(data, '$.eventid') AS eventid,
  JSON_VALUE(data, '$.cloud_provider') AS cloud_provider,
  JSON_VALUE(data, '$.node_role') AS node_role,
  JSON_VALUE(data, '$.protocol') AS protocol,
  JSON_VALUE(data, '$.session') AS session,
  JSON_VALUE(data, '$.sensor') AS sensor,
  JSON_VALUE(data, '$.src_ip') AS src_ip,
  SAFE_CAST(JSON_VALUE(data, '$.src_port') AS INT64) AS src_port,
  JSON_VALUE(data, '$.dst_ip') AS dst_ip,
  SAFE_CAST(JSON_VALUE(data, '$.dst_port') AS INT64) AS dst_port,

  -- Command execution
  JSON_VALUE(data, '$.input') AS command_input,

  -- Auth attempts
  JSON_VALUE(data, '$.username') AS username,
  JSON_VALUE(data, '$.password') AS password,

  -- Session lifecycle
  SAFE_CAST(JSON_VALUE(data, '$.duration_ms') AS INT64) AS duration_ms,

  -- Prompt-injection responses (agent-sensor only -- see
  -- handlers/prompt.txt and the parse_agent_report step in vector.toml)
  JSON_VALUE(data, '$.agent_report.agent_type') AS agent_type,
  JSON_VALUE(data, '$.agent_report.framework') AS agent_framework,
  JSON_VALUE(data, '$.agent_report.primary_objective') AS agent_primary_objective,
  JSON_VALUE(data, '$.agent_report.operator_identity') AS agent_operator_identity,

  data AS raw
FROM `mineral-droplet-160709.bear_trap_logs.raw_events`;
