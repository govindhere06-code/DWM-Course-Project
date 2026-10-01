# Usage: make <target>   (on Windows: mingw32-make <target>)
# backend/  -> ML pipeline (src, data, models, reports, notebooks, tests)
# frontend/ -> Streamlit dashboard
ifeq ($(OS),Windows_NT)
    VENV_PY := .venv/Scripts/python.exe
    BOOTSTRAP := py -3.13
else
    VENV_PY := .venv/bin/python
    BOOTSTRAP := python3
endif

ROOT := $(CURDIR)
PYTHON := "$(ROOT)/$(VENV_PY)"
BACKEND := cd backend &&

.PHONY: install prepare eda train train-quick evaluate explain dashboard test lint all

install:
	$(BOOTSTRAP) -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

prepare:
	$(BACKEND) $(PYTHON) -m src.data

eda:
	$(BACKEND) $(PYTHON) -m src.eda

# raw CSV -> validated split -> model selection -> final model, metrics, SHAP (~5 min)
train:
	$(BACKEND) $(PYTHON) -m src.train

# refit + evaluate the saved model choice / tuned params (< 1 min)
train-quick:
	$(BACKEND) $(PYTHON) -m src.train --stage final

evaluate:
	$(BACKEND) $(PYTHON) -m src.evaluate

explain:
	$(BACKEND) $(PYTHON) -m src.explain

dashboard:
	$(PYTHON) -m streamlit run frontend/app.py

test:
	$(BACKEND) $(PYTHON) -m pytest -q
	cd frontend && $(PYTHON) -m pytest -q

all: install eda train test

lint:
	$(PYTHON) -m ruff check backend frontend
	$(PYTHON) -m black --check backend frontend
