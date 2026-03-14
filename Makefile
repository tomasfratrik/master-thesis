
.PHONY: install freeze run run-app-backend docker-build docker-predict eval-finetuned seed-previews

run:
	source venv/bin/activate

venv-install:
	python3 -m venv venv
	source venv/bin/activate && pip install --upgrade pip

help:
	@echo "Available commands:"
	@echo "  make freeze   - Freeze current Python dependencies into requirements.txt"
	@echo "  make install  - Install packages from requirements.txt"
	@echo "  make run-app-backend - Run the catalog/backend API"
	@echo "  make docker-build - Build prediction image"
	@echo "  make docker-predict IMAGE=path/to/image.jpg - Run prediction in Docker"
	@echo "  make eval-finetuned CHECKPOINT=... TEST_ROOT=... - Evaluate checkpoint on labeled test split"
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

docker-predict:
	docker run --rm -v $(PWD)/artifacts:/app/artifacts -v $(PWD):/workspace sneaker-labeler:latest --image /workspace/$(IMAGE)

eval-finetuned:
	./venv/bin/python -m backend.evaluate_finetuned --checkpoint $(CHECKPOINT) --test-root $(TEST_ROOT) --output-json artifacts/eval_finetuned_report.json

seed-previews:
	./venv/bin/python -m backend.seed_previews --source-root $(SOURCE_ROOT)
