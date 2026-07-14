# Troubleshooting

## `TEXT_DETECTION_DATA is not set`

Set the environment variable before running scripts.

PowerShell:

```powershell
$env:TEXT_DETECTION_DATA = "D:/Projects/master_diploma/text-detection-data"
```

Linux setup is documented in `README.md`.

## Missing Stage 1 Processed Directory

Error example:

```text
FileNotFoundError: Missing stage-1 processed directory
```

Run stage 1 first with the same `--dataset` and `--model` values, then rerun stage 2.

## Grafana Does Not Show Outfox

Confirm the state endpoint includes outfox:

```powershell
Invoke-RestMethod http://localhost:8000/processing-states.json | Where-Object { $_.dataset -like "*outfox*" }
```

If the endpoint has outfox but the dropdown does not, restart Grafana and hard-refresh the browser tab:

```powershell
docker compose restart grafana
```

## `progress-states` URL Does Not Work In Browser

This URL is only for Grafana inside Docker:

```text
http://progress-states:8000/processing-states.json
```

Use this URL from the browser or host shell:

```text
http://localhost:8000/processing-states.json
```

## Monitoring Logs Are Empty

The compose service sets `PYTHONUNBUFFERED=1` so exporter logs should appear. Recreate the service if compose configuration changed:

```powershell
docker compose up -d --force-recreate progress-states
docker logs --tail 100 text-detection-progress-states
```

## Docker `/data/processed` I/O Error

The state exporter catches transient `OSError` failures in watch mode and keeps retrying. If errors continue, restart Docker Desktop or recreate the monitor service.

## GPU Or Memory Problems

Use smaller batches:

```powershell
--inference-batch-size 1
```

Use small smoke counts before large runs:

```powershell
--count 10
```

## Hugging Face Model Loading Issues

Confirm the model alias exists in `scripts/experiment_config.py`. If using a gated model, make sure the Hugging Face environment has access before running processing.
