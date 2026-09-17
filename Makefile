.PHONY: help venv data model circuit frames web clean

help:
	@echo "make venv     - create .venv and install dependencies"
	@echo "make data     - download MaleCNS v1.0 files (~550 MB)"
	@echo "make model    - clone the flybody MuJoCo model (~140 MB)"
	@echo "make circuit  - extract the circuit -> web/circuit.json"
	@echo "make frames   - render the fly -> web/frames/"
	@echo "make web      - serve the demo at http://localhost:8777"
	@echo "make all      - everything, in order"

all: venv data model circuit frames

PY := .venv/bin/python
BASE := https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome

venv:
	python3 -m venv .venv
	.venv/bin/pip install --quiet --upgrade pip
	.venv/bin/pip install --quiet -r requirements.txt

data:
	mkdir -p data/raw
	curl -sL -o data/raw/annotations.feather       "$(BASE)/body-annotations-male-cns-v1.0-minconf-0.5.feather"
	curl -sL -o data/raw/neurotransmitters.feather "$(BASE)/body-neurotransmitters-male-cns-v1.0.feather"
	curl -sL -o data/raw/weights.feather           "$(BASE)/connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"

model:
	git clone --depth 1 --filter=blob:none --sparse \
	  https://github.com/google-deepmind/mujoco_menagerie.git vendor/menagerie
	cd vendor/menagerie && git sparse-checkout set flybody
	ln -sfn ../vendor/menagerie/flybody/assets scenes/assets

circuit:
	$(PY) extract_circuit.py
	$(PY) export_web.py

frames:
	$(PY) render_frames.py

web:
	cd web && python3 -m http.server 8777

clean:
	rm -rf data/circuit web/frames renders __pycache__
