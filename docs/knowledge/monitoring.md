# Monitoring

Monitoring is provided by Docker Compose services in `docker-compose.yml`.

## Services

- `grafana`: Grafana UI on port `3000`.
- `progress-states`: Python HTTP server on port `8000` that serves aggregated processing states.

Core URLs:

- Browser or host shell: `http://localhost:8000/processing-states.json`.
- Grafana inside Docker: `http://progress-states:8000/processing-states.json`.
- Grafana UI: `http://localhost:3000`.

The state server is plain HTTP. Do not use HTTPS.

Start services:

```powershell
docker compose up -d
```

Restart only the state server:

```powershell
docker compose restart progress-states
```

Recreate the state server after compose environment changes:

```powershell
docker compose up -d --force-recreate progress-states
```

## State Exporter

`monitoring/scripts/export_processing_states.py` scans:

```text
$TEXT_DETECTION_DATA/processed/**/state.json
$TEXT_DETECTION_DATA/processed/**/similarity_state.json
```

It writes one JSON array for Grafana. In Docker it refreshes every 5 seconds.

The exporter preserves the last good output during transient filesystem errors and logs failures instead of killing watch mode.

## Dashboard Variables

- `dataset`: dashboard dataset selector. Keep it aligned with dataset aliases that should be visible in Grafana.
- `stage`: `logprobs` or `token_similarity`.
- `state_files_url`: default Docker URL for Grafana.

The exporter also serves `/datasets.json`, generated from `DATASETS` in `scripts/experiment_config.py`. Adding a configured dataset makes it selectable in Grafana without editing the dashboard.

Detailed Grafana setup lives in `monitoring/grafana/README.md`.
