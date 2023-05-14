VENV = .
PIP = ./bin/pip3
PYTHON = ./bin/python3

all: $(VENV)/bin/activate

$(VENV)/bin/activate: ./echoanalysis-workflow/pyproject.toml
	virtualenv -p /opt/homebrew/bin/python3.11 $(VENV)
	$(PIP) install -e './echoanalysis-workflow/.[develop]'

clean:
	rm -rf .Python bin/ lib/ pyvenv.cfg share/

run:
	$(PYTHON) run.py

.PHONY: venv run clean