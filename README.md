# Text Detection

Utilities for processing text-detection datasets with local Hugging Face language models.

## Data Layout

Local data root is configured with `TEXT_DETECTION_DATA`.

Windows local example:

```text
D:/Projects/master_diploma/text-detection-data
```

Linux/Vast example:

```text
/workspace/text-detection/data
```

Dataset split files are configured in `scripts/experiment_config.py`.

Processed outputs are written under:

```text
processed/{dataset}__{model}/
```

## Local Setup

In `cmd.exe`:

```bat
set "TEXT_DETECTION_DATA=D:\Projects\master_diploma\text-detection-data"
uv sync
```

In PowerShell:

```powershell
$env:TEXT_DETECTION_DATA = "D:/Projects/master_diploma/text-detection-data"
uv sync
```

For notebook and visualization workflows, install the optional notebook
dependencies instead:

```powershell
uv sync --extra notebook
```

## Linux Setup

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

## Mutagen Sync

The project uses `mutagen.yml` for sync configuration.

Start sync:

```bat
mutagen project start
```

Stop sync:

```bat
mutagen project terminate
```

Check status:

```bat
mutagen project list
mutagen sync list
```

Defined sync sessions:

```text
td-code                 local code -> /workspace/text-detection
raw-data-upload         local raw dataset files -> remote data root
processed-states-upload local processed state files -> remote processed state files
data-download           remote data root -> local data root
```

Do not run upload and download data syncs at the same time unless that is intentional.

## Processing Commands

Dataset preparation and the two staged processing commands are the supported
entrypoints. They can be run as scripts or as package modules:

```bat
uv run python scripts\extract_data\prepare_outfox_chatgpt.py
uv run python -m scripts.process_pipeline --help
uv run python -m scripts.token_similarity_stage2 --help
```

Run a small local Facebook OPT processing test, 10 stage-1 rows per split:

```bat
uv run python scripts\process_pipeline.py --count 10 --model opt_350m --dataset <dataset-alias> --top-n 20 --inference-batch-size 1
```

`process_pipeline.py` runs stage 2 token similarity processing after stage 1 by
default. Add `--no-stage-2` when only stage 1 logprob output is required.

Run Facebook OPT processing on Linux, 1000 stage-1 rows per split:

```sh
uv run python scripts/process_pipeline.py --count 1000 --model opt_350m --dataset <dataset-alias> --top-n 20 --inference-batch-size 4
```

The Linux command also runs stage 2 by default; use `--no-stage-2` for stage 1
only.

If GPU memory is low, reduce `--inference-batch-size` to `1`.
