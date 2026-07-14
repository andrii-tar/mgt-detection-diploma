# Sync And Remote Workflow

The project uses `mutagen.yml` for local-to-remote and remote-to-local sync.

## Sessions

- `td-code`: syncs local code to `/workspace/text-detection` on the remote machine.
- `raw-data-upload`: syncs local raw Kaggle data to the remote data root.
- `processed-states-upload`: syncs local processing state files to the remote processed tree.
- `data-download`: syncs remote data back to the local data root.

Start sync:

```powershell
mutagen project start
```

Stop sync:

```powershell
mutagen project terminate
```

Check status:

```powershell
mutagen project list
mutagen sync list
```

## Safety

Do not run upload and download data syncs together unless that is intentional. Check direction before starting long processing jobs or data transfers.

## Remote Environment

Create `/workspace/text-detection/.env` on the remote machine:

```sh
cd /workspace/text-detection
printf 'TEXT_DETECTION_DATA=/workspace/text-detection/data\n' > .env
uv sync
```

Before running processing on Linux:

```sh
cd /workspace/text-detection
set -a
. ./.env
set +a
```
