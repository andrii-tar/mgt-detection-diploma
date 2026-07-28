---
type: Operations Guide
title: Operations and Monitoring
description: Operational notes for the Grafana and state-export stack. Explains how processing state is aggregated from pipeline outputs, how the dashboard is configured, and what to update when dataset aliases or state fields change.
tags: [operations, monitoring, grafana, docker]
resource: /monitoring/grafana/README.md
---

# Operations and monitoring

Monitoring is a lightweight companion to the main processing pipeline. It reads pipeline state files from `TEXT_DETECTION_DATA`, aggregates them, and exposes them to Grafana through a plain HTTP service.

## Runtime components

The monitoring stack lives in `docker-compose.yml` and includes:

- `grafana`: the UI on port `3000`.
- `progress-states`: a Python HTTP service on port `8000` that serves the aggregated processing states.

The dashboard and exporter are intentionally simple: they avoid reprocessing pipeline data and only read the state files the pipeline already wrote.

## State exporter

[`monitoring/scripts/export_processing_states.py`](../monitoring/scripts/export_processing_states.py) scans:

```text
$TEXT_DETECTION_DATA/processed/**/state.json
$TEXT_DETECTION_DATA/processed/**/similarity_state.json
```

It writes a JSON array that Grafana can query. The exporter keeps the last good output during transient filesystem errors and logs failures rather than crashing watch mode.

## Dashboard configuration

The dashboard definition is in [`monitoring/grafana/dashboards/processing-progress.json`](../monitoring/grafana/dashboards/processing-progress.json). The dataset variable is generated from the dataset aliases in [`scripts/experiment_config.py`](../scripts/experiment_config.py), which means dashboard updates are required whenever dataset aliases change.

The existing tests explicitly check that the dashboard uses `http://progress-states:8000/processing-states.json` rather than HTTPS and that the dataset variable matches the configured aliases.

## Operational commands

Start the stack:

```powershell
docker compose up -d
```

Restart only the state server after a change:

```powershell
docker compose restart progress-states
```

Force a refresh after compose environment changes:

```powershell
docker compose up -d --force-recreate progress-states
```

## Change checklist

When you modify datasets, state fields, or monitoring behavior:

- Update [`scripts/experiment_config.py`](../scripts/experiment_config.py) first.
- Keep Grafana variables aligned with the new aliases.
- Update `monitoring/grafana/dashboards/processing-progress.json` if a new dataset should be visible.
- Update exporter logic if the state file schema changes.
- Run the monitoring tests to confirm the dashboard and exporter contract still hold.

## Related pages

- [`Architecture overview`](../architecture/overview.md) for how monitoring fits into the overall system.
- [`Domain model and data layout`](../domain/data-model.md) for the state-file contract.
- [`Testing guidance`](../testing.md) for the monitoring contract tests.
