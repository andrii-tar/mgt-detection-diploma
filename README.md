# Text Detection

Utilities for processing Kaggle text-detection data with local Hugging Face language models.

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

Expected Kaggle split files:

```text
raw/kaggle_train/kaggle_train_hg.csv
raw/kaggle_train/kaggle_train_mg.csv
```

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

Run a small local SmolLM stage 1 test, 10 rows per split:

```bat
uv run python scripts\kaggle_process_pipeline.py --count 10 --model smollm2_360m --dataset kaggle_train --top-n 20 --inference-batch-size 1
```

Run SmolLM stage 1 on Linux, 1000 rows per split:

```sh
uv run python scripts/kaggle_process_pipeline.py --count 1000 --model smollm2_360m --dataset kaggle_train --top-n 20 --inference-batch-size 4
```

Run Gemma 2 2B stage 1 on Linux, 100 rows per split:

```sh
uv run python scripts/kaggle_process_pipeline.py --count 100 --model gemma_2_2b_it --dataset kaggle_train --top-n 20 --inference-batch-size 1
```

If GPU memory is low, reduce `--inference-batch-size` to `1`.
