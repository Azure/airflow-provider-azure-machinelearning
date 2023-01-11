.PHONY: build clean dist-test dist test

build:
	python3 -m pip install --user --upgrade build
	python3 -m build
	twine check dist/*

clean:
	rm -rf dist

test:
	python3 -m unittest

dist-test: clean test build
	python3 -m pip install --user --upgrade twine
	twine upload --repository testpypi dist/*

dist: clean test build
	python3 -m pip install --user --upgrade twine
	twine upload dist/*
