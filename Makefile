.PHONY: build clean dist-test dist test

build:
	python3 -m pip install --upgrade build
	python3 -m build
	twine check dist/*

clean:
	rm -rf dist

test:
	python3 -m unittest

#dist-test:
#	python3 -m pip install --user --upgrade twine
#	python3 -m twine upload --repository testpypi dist/*

#dist:
#	python3 -m pip install --user --upgrade twine
#	python3 -m twine upload dist/*
