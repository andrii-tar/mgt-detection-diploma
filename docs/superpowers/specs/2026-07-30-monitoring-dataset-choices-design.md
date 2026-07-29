# Monitoring Dataset Choices Design

## Goal

Make `scripts/experiment_config.py` the sole operational source of dataset
aliases used by the Grafana processing dashboard. A newly configured dataset
must become selectable without editing a second manual list.

## Design

The monitoring exporter will derive sorted dataset aliases directly from
`experiment_config.DATASETS` and write them to a separate `datasets.json`
file. The existing `processing-states.json` response remains a JSON array of
state objects so existing dashboard panel queries and consumers are not
changed.

Grafana's dataset variable will use the existing Infinity datasource to query
the generated dataset choices endpoint. The processing panels will continue
to query `processing-states.json` through Infinity.

## Data Flow

1. The exporter imports `DATASETS`.
2. Each export cycle writes the sorted aliases to `datasets.json`.
3. The existing Python HTTP server exposes the file.
4. Grafana reads the choices through its Infinity datasource.

## Failure Behavior

Dataset output follows the exporter's existing filesystem error handling. A
failed write is reported without deleting a previous valid output. The state
file format and refresh behavior remain unchanged.

## Validation

Keep tests focused on the monitoring contract: generated choices must equal
the sorted keys of `DATASETS`, and the dashboard variable must target the
generated endpoint rather than contain a custom duplicated list.
