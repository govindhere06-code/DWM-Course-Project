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

.PHONY: install eda train tune evaluate explain dashboard test all

install:
	$(BOOTSTRAP) -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

eda:
	$(BACKEND) $(PYTHON) -m src.eda

train:
	$(BACKEND) $(PYTHON) -m src.train

tune:
	$(BACKEND) $(PYTHON) -m src.tune

evaluate:
	$(BACKEND) $(PYTHON) -m src.evaluate

explain:
	$(BACKEND) $(PYTHON) -m src.explain

dashboard:
	$(PYTHON) -m streamlit run frontend/app.py

test:
	$(BACKEND) $(PYTHON) -m pytest -q

all: install eda train tune evaluate explain
