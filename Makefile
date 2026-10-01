# Usage: make <target>   (on Windows: mingw32-make <target>)
ifeq ($(OS),Windows_NT)
    PYTHON := .venv/Scripts/python.exe
    BOOTSTRAP := py -3.13
else
    PYTHON := .venv/bin/python
    BOOTSTRAP := python3
endif

.PHONY: install eda train tune evaluate explain dashboard test all

install:
	$(BOOTSTRAP) -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

eda:
	$(PYTHON) -m src.eda

train:
	$(PYTHON) -m src.train

tune:
	$(PYTHON) -m src.tune

evaluate:
	$(PYTHON) -m src.evaluate

explain:
	$(PYTHON) -m src.explain

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py

test:
	$(PYTHON) -m pytest -q

all: install eda train tune evaluate explain
