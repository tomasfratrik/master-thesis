# Master Thesis Sneaker Matcher

## Development Commands

Install backend dependencies:

```bash
make install
```

Run the backend API:

```bash
make run-app-backend
```

Run the frontend:

```bash
cd frontend
npm install
npm run dev
```

## Docker Runtime

Optionally copy `.env.example` to `.env` and adjust the asset URLs, then run:

```bash
docker compose up --build
```

Or:

```bash
make docker-up
```

This starts:
- backend on `http://localhost:8090`
- frontend on `http://localhost:4173`

On first startup, the backend downloads:
- runtime assets from `RUNTIME_ASSETS_REPO_URL`
- the model checkpoint from `MODEL_REPO_URL`

`SNEAKER_MODEL_CHECKPOINT` is resolved automatically from the downloaded model repo.

## Crop Demo Export

To execute standalone sneaker crop export, run the following command:

```bash
./venv/bin/python -m backend.export_sneaker_crops /path/to/input.jpg /tmp/crop-demo
```

The command writes crop images and metadata into the output directory:

```text
/tmp/crop-demo/input_crop_1.jpg
/tmp/crop-demo/input_crop_2.jpg
/tmp/crop-demo/input_metadata.json
```

Optional crop settings can be passed from the command line:

```bash
./venv/bin/python -m backend.export_sneaker_crops image.jpg out-dir --padding 12 --padding-ratio 0.15
```

Use `--return-original-if-empty` to save the original image when no sneaker crop is detected.
