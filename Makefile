# Reproduce every number in the write-up:  make install && make data && make all
PY      ?= python
LOGS    := results/logs
RAW_ZIP := data/raw/bbl_json.zip

.PHONY: install data all eda match powerplay players test lint format clean help

help:
	@echo "make install    - editable install with dev tools"
	@echo "make data       - download Cricsheet BBL ball-by-ball JSON and build tidy tables"
	@echo "make all        - eda + match + powerplay + players (logs in results/logs)"
	@echo "make test|lint|format"

install:
	$(PY) -m pip install -e ".[dev]"

data:
	mkdir -p data/raw
	curl -L -o $(RAW_ZIP) https://cricsheet.org/downloads/bbl_json.zip
	$(PY) -m bbl.load_cricsheet $(RAW_ZIP)

# Each stage logs its console output; later stages import earlier ones (see README, "Pipeline").
eda:
	$(PY) -m bbl.eda | tee $(LOGS)/eda.txt

match:
	$(PY) -m bbl.match_model   | tee $(LOGS)/match_backtest.txt
	$(PY) -m bbl.match_predict | tee $(LOGS)/match_predict.txt

powerplay:
	$(PY) -m bbl.powerplay_model   | tee $(LOGS)/powerplay_backtest.txt
	$(PY) -m bbl.powerplay_drift   | tee $(LOGS)/powerplay_drift.txt
	$(PY) -m bbl.powerplay_predict | tee $(LOGS)/powerplay_predict.txt

players:
	$(PY) -m bbl.player_layer     | tee $(LOGS)/player_layer.txt
	$(PY) -m bbl.player_scenarios | tee $(LOGS)/player_scenarios.txt

all: eda match powerplay players

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests
	black --check src tests

format:
	black src tests

clean:
	rm -f results/*.csv results/*.png results/*.json results/*.pkl results/*.npy results/logs/*.txt
