
.PHONY: install freeze run

run:
	source venv/bin/activate

venv-install:
	python3 -m venv venv
	source venv/bin/activate && pip install --upgrade pip

help:
	@echo "Available commands:"
	@echo "  make freeze   - Freeze current Python dependencies into requirements.txt"
	@echo "  make install  - Install packages from requirements.txt"

freeze:
	pip freeze > requirements.txt

install: venv-install
	pip install -r requirements.txt
	pip install git+https://github.com/openai/CLIP.git
