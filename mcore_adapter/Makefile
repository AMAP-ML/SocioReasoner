.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  make clean    - Remove build artifacts and temporary files"
	@echo "  make build    - Build the wheel package"
	@echo "  make build-release      - Clean and build"

clean:
	@echo "Cleaning up..."
	rm -rf dist/
	rm -rf build/
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

build:
	@echo "Building wheel package..."
	python setup.py bdist_wheel
	python setup.py sdist

build-release: clean build

.PHONY: help clean build build-release
