VENV = .
PIP = ./bin/pip3
PYTHON = ./bin/python3
BRANCH  = dev-active-staging-1

$(VENV)/bin/activate: ./echonet-workflow/pyproject.toml
	virtualenv -p /usr/bin/python3 $(VENV)
	$(PIP) install -e './echonet-workflow/[develop]'

venv: $(VENV)/bin/activate

run: venv
	git pull origin $(BRANCH)

clean:
	rm -rf .Python bin/ lib/ pyvenv.cfg share/

.PHONY: venv run clean