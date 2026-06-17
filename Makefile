.PHONY: setup data test train all

setup:
	pip install -r requirements.txt && pip install -e .

data:
	python scripts/get_data.py

test:
	pytest -q

train:
	python -m churn.train

all: test train
