ENV = DSMP
YML_FILE = environment.yml
SRC = spectroscopy/src/
SCRIPTS = spectroscopy/scripts/

.PHONY: create-env update-env remove exp-env run-model lint-format lint-fix type-check

create-env:
	@echo Creating conda environment...
	conda env create -f $(YML_FILE)

update-env:
	@echo Updating conda environment...
	conda env update -f $(YML_FILE) --prune

remove-env:
	@echo Removing conda environment...
	conda env remove -n $(ENV)

exp-env:
	@echo Exporting conda environment...
	conda env export > $(YML_FILE)

run-model:
	@echo Running model...
	python spectroscopy/scripts/full_model_template.py

lint-format:
	@echo Linting src...
	ruff format $(SRC)
	@echo Linting scripts...
	ruff format $(SCRIPTS)
	@echo Done!

lint-fix: 
	@echo Linting and fixing src...
	ruff check $(SRC) --fix --ignore F401
	@echo Linting and fixing scripts...
	ruff check $(SCRIPTS) --fix --ignore F401
	@echo Done!

type-check:
	@echo Type checking...
	mypy $(SRC) 
	@echo Done!