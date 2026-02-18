
.PHONY: install freeze run docker-build docker-predict

run:
	source venv/bin/activate

venv-install:
	python3 -m venv venv
	source venv/bin/activate && pip install --upgrade pip

help:
	@echo "Available commands:"
	@echo "  make freeze   - Freeze current Python dependencies into requirements.txt"
	@echo "  make install  - Install packages from requirements.txt"
	@echo "  make docker-build - Build prediction image"
	@echo "  make docker-predict IMAGE=path/to/image.jpg - Run prediction in Docker"

freeze:
	pip freeze > requirements.txt

install: venv-install
	pip install -r requirements.txt
	pip install git+https://github.com/openai/CLIP.git

docker-build:
	docker build -t sneaker-labeler:latest .

docker-predict:
	docker run --rm -v $(PWD)/artifacts:/app/artifacts -v $(PWD):/workspace sneaker-labeler:latest --image /workspace/$(IMAGE)
