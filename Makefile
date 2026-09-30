.PHONY: install test leakage eval transformer all clean

install:
	pip install -r requirements.txt

test:
	cd projects/01-leakage-audit && python -m pytest tests/ -q
	cd projects/02-agent-eval-harness && python -m pytest tests/ -q
	cd projects/03-pytorch-from-scratch && python -m pytest tests/ -q

leakage:
	cd projects/01-leakage-audit && python -m src.experiment && python -m src.figures

eval:
	cd projects/02-agent-eval-harness && python -m src.harness --split heldout

transformer:
	cd projects/03-pytorch-from-scratch && python -m src.bench --steps 12 && \
	  python -m src.decode_bench --new-tokens 64 && python -m src.figures

all: test leakage eval transformer

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
