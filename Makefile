CXX ?= clang++
CXXFLAGS ?= -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic

BIN := build/netflix
ARCHIVE := data/raw/nf_prize_dataset.tar.gz
DATASET := data/processed/netflix.bin
EXTRACTED := data/raw/extracted/.complete

.PHONY: all download verify extract ingest baseline time-baseline svdpp svdpp-f32 svdpp-seed42 time-svdpp blend test clean

all: $(BIN)

build:
	mkdir -p build

$(BIN): src/netflix.cpp src/joint.hpp src/ensemble.hpp src/neighborhood.hpp src/rbm.hpp src/integrated.hpp src/temporal_neighborhood.hpp | build
	$(CXX) $(CXXFLAGS) $< -o $@

download: $(ARCHIVE)

$(ARCHIVE):
	mkdir -p data/raw
	curl --fail --location --continue-at - --output $(ARCHIVE) \
		https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz

verify: download
	@actual=$$(md5 -q $(ARCHIVE)); \
	test "$$actual" = "a8f23d2d76461211c6b4c0ca6df2547d" || \
	  { echo "bad archive md5: $$actual" >&2; exit 1; }

extract: verify $(EXTRACTED)

$(EXTRACTED): $(ARCHIVE)
	mkdir -p data/raw/extracted
	tar -xzf $(ARCHIVE) -C data/raw/extracted
	mkdir -p data/raw/extracted/download/training_set
	tar -xf data/raw/extracted/download/training_set.tar \
		-C data/raw/extracted/download/training_set
	touch $(EXTRACTED)

ingest: $(BIN) extract
	$(BIN) ingest \
		--training-dir data/raw/extracted/download/training_set/training_set \
		--probe data/raw/extracted/download/probe.txt \
		--qualifying data/raw/extracted/download/qualifying.txt \
		--output $(DATASET)

baseline: $(BIN) $(DATASET)
	$(BIN) baseline --data $(DATASET) --model artifacts/baseline.bin

time-baseline: $(BIN) $(DATASET)
	$(BIN) time-baseline --data $(DATASET) \
		--model artifacts/time-baseline.bin --iterations 40

svdpp: $(BIN) $(DATASET)
	$(BIN) svdpp --data $(DATASET) \
		--time-model artifacts/time-baseline.bin \
		--model artifacts/svdpp-f64.bin --factors 64 --iterations 10

svdpp-f32: $(BIN) $(DATASET)
	$(BIN) svdpp --data $(DATASET) \
		--time-model artifacts/time-baseline.bin \
		--model artifacts/svdpp-f32-best.bin --factors 32 --iterations 7

svdpp-seed42: $(BIN) $(DATASET)
	$(BIN) svdpp --data $(DATASET) \
		--time-model artifacts/time-baseline.bin \
		--model artifacts/svdpp-f64-seed42.bin --factors 64 --iterations 7 \
		--seed 42

time-svdpp: $(BIN) $(DATASET)
	$(BIN) time-svdpp --data $(DATASET) \
		--time-model artifacts/time-baseline.bin \
		--svd-model artifacts/svdpp-f64.bin \
		--model artifacts/time-svdpp-f64.bin --iterations 6

blend: $(BIN) $(DATASET)
	$(BIN) blend-svdpp --data $(DATASET) \
		--time-model artifacts/time-baseline.bin \
		--model-a artifacts/svdpp-f32-best.bin \
		--model-b artifacts/svdpp-f64.bin \
		--time-svd-model-b artifacts/time-svdpp-f64.bin \
		--model-c artifacts/svdpp-f64-seed42.bin

test: $(BIN)
	bash tests/smoke.sh
	python3 tests/test_joint.py
	python3 tests/test_temporal_neighborhood.py
	python3 tests/test_timeline.py

clean:
	rm -rf build

# Isolated macOS research binary; one-thread runs recommended alongside training.
.PHONY: moonshot-gp test-moonshot-gp
moonshot-gp: build/moonshot_gp

build/moonshot_gp: src/moonshot_gp.cpp src/netflix.cpp src/joint.hpp src/ensemble.hpp src/neighborhood.hpp src/rbm.hpp | build
	$(CXX) $(CXXFLAGS) -Wno-deprecated-declarations $< -framework Accelerate -o $@

test-moonshot-gp: $(BIN) build/moonshot_gp
	VECLIB_MAXIMUM_THREADS=1 python3 tests/test_moonshot_gp.py

.PHONY: moonshot-local
moonshot-local: build/moonshot_local
build/moonshot_local: src/moonshot_local.cpp src/netflix.cpp src/joint.hpp src/ensemble.hpp src/neighborhood.hpp src/rbm.hpp | build
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY: moonshot-user-kernel
moonshot-user-kernel: build/moonshot_user_kernel
build/moonshot_user_kernel: src/moonshot_user_kernel.cpp src/netflix.cpp src/joint.hpp src/ensemble.hpp src/neighborhood.hpp src/rbm.hpp | build
	$(CXX) $(CXXFLAGS) -Wno-deprecated-declarations $< -framework Accelerate -o $@
