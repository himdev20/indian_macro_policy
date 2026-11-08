all: install run test

install:
	pip install -e ".[test]"

run:
	python -m indiapolicy

test:
	pytest -q

clean:
	rm -rf output/figures/* output/tables/* output/results.json data/processed/*

.PHONY: all install run test clean
