
.PHONY: install freeze run run-app-backend docker-build docker-predict docker-up docker-up-gpu eval-finetuned plot-training plot-training-diagnostics plot-eval seed-previews

run:
	source venv/bin/activate

venv-install:
	python3 -m venv venv
	source venv/bin/activate && pip install --upgrade pip

help:
	@echo "Available commands:"
	@echo "  Note: GPU inference is much faster. Analysis can take about 1 second on GPU and 10+ seconds on CPU."
	@echo "  make freeze   - Freeze current Python dependencies into requirements.txt"
	@echo "  make install  - Install packages from requirements.txt"
	@echo "  make run-app-backend - Run the catalog/backend API"
	@echo "  make docker-build - Build prediction image"
	@echo "  make docker-up - Start backend and frontend with Docker Compose"
	@echo "  make docker-up-gpu - Start backend and frontend with Docker Compose GPU runtime"
	@echo "  make docker-predict IMAGE=path/to/image.jpg - Run prediction in Docker"
	@echo "  make eval-finetuned CHECKPOINT=... TEST_ROOT=... - Evaluate checkpoint on labeled test split"
	@echo "  make plot-training HISTORY='a.json [b.json ...]' - Render training loss/accuracy graphs"
	@echo "  make plot-training-diagnostics HISTORY=a.json - Render underfit/overfit diagnostics"
	@echo "  make plot-eval REPORTS='a.json [b.json ...]' - Render evaluation comparison graphs"
	@echo "  make seed-previews SOURCE_ROOT=... - Copy sample preview images per class"

freeze:
	pip freeze > requirements.txt

install: venv-install
	pip install -r requirements.txt
	pip install git+https://github.com/openai/CLIP.git

run-app-backend:
	./venv/bin/uvicorn backend.app.api:app --host 0.0.0.0 --port 8090

docker-build:
	docker build -t sneaker-labeler:latest .

docker-up:
	docker compose up --build

docker-up-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

docker-predict:
	docker run --rm -v $(PWD)/artifacts:/app/artifacts -v $(PWD):/workspace sneaker-labeler:latest --image /workspace/$(IMAGE)

eval-finetuned:
	./venv/bin/python -m backend.evaluate_finetuned --checkpoint $(CHECKPOINT) --test-root $(TEST_ROOT) --output-json artifacts/eval_finetuned_report.json

plot-training:
	./venv/bin/python -m backend.plot_training_history $(HISTORY) --output-dir artifacts/training_plots

plot-training-diagnostics:
	./venv/bin/python -m backend.plot_training_diagnostics $(HISTORY) --output-dir artifacts/training_plots

plot-eval:
	./venv/bin/python -m backend.plot_eval_results $(REPORTS) --output-dir artifacts/eval_plots

seed-previews:
	./venv/bin/python -m backend.seed_previews --source-root $(SOURCE_ROOT)
