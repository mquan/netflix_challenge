CXX ?= clang++
CXXFLAGS ?= -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic

BIN := build/netflix
ARCHIVE := data/raw/nf_prize_dataset.tar.gz
DATASET := data/processed/netflix.bin
EXTRACTED := data/raw/extracted/.complete

.PHONY: all download verify extract ingest baseline time-baseline test clean

all: $(BIN)

build:
	mkdir -p build

$(BIN): src/netflix.cpp | build
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

test: $(BIN)
	bash tests/smoke.sh

clean:
	rm -rf build
