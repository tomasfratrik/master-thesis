# Manual Installation

Use this only if you do not want to run the project through Docker.

## 1. Download Runtime Assets

Clone the runtime assets repository somewhere local:

```bash
git clone git@github.com:tomasfratrik/thesis-runtime-assets.git
```

Expected contents:

```text
app_data/
  app.sqlite3
artifacts/
  image_embeddings.npy
  image_meta.json
previews/
```

Copy those into this repository:

```bash
cp -a thesis-runtime-assets/app_data/. app_data/
cp -a thesis-runtime-assets/artifacts/. artifacts/
cp -a thesis-runtime-assets/previews/. previews/
```

## 2. Download Model

Clone the model repository:

```bash
git clone git@hf.co:tomasfratrik/thesis-model
cd thesis-model
git lfs pull
cd ..
```

Find the checkpoint file inside that repository and set:

```bash
export SNEAKER_MODEL_CHECKPOINT=/absolute/path/to/thesis-model/your-checkpoint.pt
```

## 3. Install Backend

```bash
make install
```

## 4. Run Backend

```bash
make run-app-backend
```

The backend starts on:

```text
http://localhost:8090
```

## 5. Install Frontend

```bash
cd frontend
npm install
```

If needed, set the backend URL:

```bash
export PUBLIC_API_BASE_URL=http://localhost:8090
```

## 6. Run Frontend

```bash
npm run dev
```

The frontend starts on the Vite development port shown in the terminal, typically:

```text
http://localhost:5173
```
