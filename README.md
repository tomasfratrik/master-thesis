# Master Thesis Sneaker Matcher

Docker is the primary installation and runtime path for this project.

## Prerequisites

Install:
- Docker
- Docker Compose

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
