# Master Thesis Sneaker Matcher

Docker is the primary installation and runtime path for this project.

## Prerequisites

Install:
- Docker
- Docker Compose

Optional for GPU inference:
- NVIDIA GPU
- NVIDIA drivers
- NVIDIA Container Toolkit

Check that they are available:

```bash
docker --version
docker compose version
```

If your machine uses the older standalone Compose command, replace `docker compose`
with `docker-compose` in the commands below.

On some systems, Docker commands may also require `sudo`.

For example:

```bash
sudo docker compose version
```

## Run with Docker

Optionally if you want to use your model/assets copy `.env.example` to `.env` and adjust:
- `RUNTIME_ASSETS_REPO_URL`
- `MODEL_REPO_URL`

Start the full project:

```bash
docker compose up --build
```

If Docker requires elevated privileges on your machine:

```bash
sudo docker compose up --build
```

Or through Make:

```bash
make docker-up
```

This starts:
- backend on `http://localhost:8090`
- frontend on `http://localhost:4173`

On first startup, the backend automatically downloads:
- runtime assets from `RUNTIME_ASSETS_REPO_URL`
- the model repository from `MODEL_REPO_URL`

`SNEAKER_MODEL_CHECKPOINT` is resolved automatically from the downloaded model repository.

If you use Hugging Face rate-limited downloads, set `HF_TOKEN` in `.env`.

## Optional GPU Runtime

The recommended default setup runs on CPU:

```bash
docker compose up --build
```

This works on any machine with Docker and does not require NVIDIA container support.

GPU mode is optional and should only be used if the host machine is already configured for
NVIDIA containers.

If your host already has:
- NVIDIA drivers
- NVIDIA Container Toolkit

first verify GPU access on the host:

```bash
nvidia-smi
```

then verify GPU access in Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

If the Docker test fails, configure the NVIDIA runtime on the host, for example:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo systemctl restart docker
```

After Docker GPU access works, start the project in GPU mode:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

This project does not install NVIDIA drivers or the NVIDIA container runtime for you.
GPU support depends on the host machine setup.

## Docker Operations

Show running services:

```bash
docker compose ps
```

Show logs:

```bash
docker compose logs -f
```

Stop services:

```bash
docker compose stop
```

Stop and remove containers:

```bash
docker compose down
```

Stop and remove containers, networks, and volumes:

```bash
docker compose down -v
```

Rebuild from scratch:

```bash
docker compose up --build --force-recreate
```

## Manual Installation

Manual local installation is documented in:

[`MANUAL-INSTALLATION.md`](/home/tomas/dev/master-thesis/master-thesis/MANUAL-INSTALLATION.md)

## Crop Demo Export

To execute standalone sneaker crop export:

```bash
./venv/bin/python -m backend.export_sneaker_crops /path/to/input.jpg /tmp/crop-demo
```

The command writes:

```text
/tmp/crop-demo/input_crop_1.jpg
/tmp/crop-demo/input_crop_2.jpg
/tmp/crop-demo/input_metadata.json
```

Optional settings:

```bash
./venv/bin/python -m backend.export_sneaker_crops image.jpg out-dir --padding 12 --padding-ratio 0.15
```

Use `--return-original-if-empty` to save the original image when no sneaker crop is detected.

## Preprocessing Order

The backend preprocessing pipeline runs these steps by default:

```text
normalize_format -> crop_sneakers -> resize_limit -> grayscale
```

The default is defined in `backend/app/preprocess_service/config.py` as
`PreprocessConfig.step_order`. The order can also be overridden per request with
`RuntimeOptions.step_order`:

```python
RuntimeOptions(
    step_order=[
        "normalize_format",
        "crop_sneakers",
        "resize_limit",
        "grayscale",
    ]
)
```

Valid step names are `normalize_format`, `crop_sneakers`, `resize_limit`, and
`grayscale`.

## Training Graphs

Fine-tuning writes a training-history JSON file, for example:

```text
artifacts/finetuned_models/efficientnet_b0_training_history.json
```

Render loss, validation accuracy, and learning-rate graphs with:

```bash
./venv/bin/python -m backend.plot_training_history \
  artifacts/finetuned_models/efficientnet_b0_training_history.json \
  --output-dir artifacts/training_plots \
  --prefix efficientnet_b0
```

Or through Make:

```bash
make plot-training HISTORY=artifacts/finetuned_models/efficientnet_b0_training_history.json
```

The command writes PNG files for loss, validation accuracy, and learning rate.
The accuracy graph always uses a fixed `0..1` y-axis.

To make an underfitting/overfitting figure, render the diagnostics plot:

```bash
./venv/bin/python -m backend.plot_training_diagnostics \
  artifacts/finetuned_models/efficientnet_b0_training_history.json \
  --output-dir artifacts/training_plots \
  --prefix efficientnet_b0
```

This writes `efficientnet_b0_diagnostics.png`. The plot marks the early
underfit checkpoint candidate, the best validation epoch, the final checkpoint,
and shows whether the validation loss starts getting worse while the training
loss continues to decrease.

For thesis visual comparisons, train with epoch checkpoints enabled:

```bash
./venv/bin/python -m backend.finetune_efficientnet \
  --data-root path/to/sneakers-mixed \
  --epochs 30 \
  --checkpoint-every 1
```

The same option is available for CLIP fine-tuning:

```bash
./venv/bin/python -m backend.finetune_clip \
  --data-root path/to/sneakers-mixed \
  --epochs 30 \
  --checkpoint-every 1
```

Use the diagnostics plot to choose an early underfit checkpoint, the best
checkpoint, and a late/final overfit checkpoint. Then run Grad-CAM or feature
channels on the same image for each checkpoint.

To compare two runs in the same plots, pass multiple history files and labels:

```bash
./venv/bin/python -m backend.plot_training_history \
  artifacts/run_a/training_history.json \
  artifacts/run_b/training_history.json \
  --label clip \
  --label efficientnet \
  --output-dir artifacts/training_plots \
  --prefix clip_vs_efficientnet
```

## Evaluation Graphs

Tagged evaluation JSON reports can be visualized with:

```bash
./venv/bin/python -m backend.plot_eval_results \
  artifacts/finetuned_models/final/tests/eval_clip_sneakers-mixed_best.json \
  artifacts/finetuned_models/final/tests/eval_efficientnet_b0_sneaker-mixed_best.json \
  --label clip_mixed \
  --label efficientnet_mixed \
  --output-dir artifacts/eval_plots \
  --prefix mixed_comparison
```

The command writes:

```text
mixed_comparison_summary.png
mixed_comparison_per_tag_top1_accuracy.png
mixed_comparison_per_class_top1_accuracy.png
```

Use `--metric topk_accuracy` to render per-tag and per-class top-k graphs instead.

For older reports without tags, render only the summary comparison:

```bash
./venv/bin/python -m backend.plot_eval_results \
  artifacts/eval_finetuned_report.json \
  artifacts/eval_subset1.json \
  artifacts/finetuned_models/final/tests/eval_clip_sneakers-mixed_best.json \
  artifacts/finetuned_models/final/tests/eval_efficientnet_b0_sneaker-mixed_best.json \
  --label clip_142class \
  --label clip_13_oldtest \
  --label final_clip \
  --label final_effnet \
  --output-dir artifacts/eval_plots \
  --prefix old_vs_final \
  --summary-only
```

The summary plot intentionally compares `top1_accuracy` and
`mean_margin_vs_second`, not top-k accuracy.
