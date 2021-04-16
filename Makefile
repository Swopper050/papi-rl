LINT_FILES = *.py papi-rl tests
TEST_PATH = tests
PYTEST = py.test $(TEST_PATH) --pythonwarnings=once
PYTEST_ARGS_COV = --cov-report=term-missing --cov-report=html --cov=papi-rl

define echotask
	@tput setaf 6
	@echo -n "  $1"
	@tput setaf 8
	@echo -n " - "
	@tput sgr0
	@echo $2
endef

help:
	@echo
	$(call echotask,"deps","installs and updates all dependencies for developing")
	$(call echotask,"doc","builds documentation using pydoc")
	$(call echotask,"format","formats code using isort and black")
	$(call echotask,"formatcheck","checks format using isort and black")
	$(call echotask,"flake8","lints code using flake8")
	$(call echotask,"lint","lints all code using flake8 isort and black")
	$(call echotask,"formatlint","formats and lints code using flake8 isort and black")
	$(call echotask,"test","runs all tests")
	$(call echotask,"test_cov","runs pytest and creates htmlcov")
	@echo

deps:
	pip install -U pip
	pip install -Ur requirements.txt

format:
	isort --filter-files $(LINT_FILES)
	black $(LINT_FILES)

formatcheck:
	isort --check-only --filter-files $(LINT_FILES)
	black --check $(LINT_FILES)

flake8:
	flake8 $(LINT_FILES)

lint: flake8 formatcheck

formatlint: format flake8

test:
	$(PYTEST)

test_cov:
	$(PYTEST) $(PYTEST_ARGS_COV)
