.PHONY: help setup verify train-all test clean eda
.PHONY: stage1 stage2 stage3a stage3b stage4a stage4b stage5

help:
	@echo "King Gizzard Setlist Predictor - ML Pipeline"
	@echo ""
	@echo "Setup:"
	@echo "  make setup           - Install dependencies and create directories"
	@echo "  make verify          - Verify project setup"
	@echo ""
	@echo "Training (Run all experiments):"
	@echo "  make train-all       - Run all 6 stages sequentially"
	@echo "  make stage1          - Stage 1: Baseline models (Logistic, RF, XGBoost)"
	@echo "  make stage2          - Stage 2: XGBoost hyperparameter tuning"
	@echo "  make stage3a         - Stage 3A: Deep learning tabular (MLP, DeepFM)"
	@echo "  make stage3b         - Stage 3B: Deep learning with embeddings"
	@echo "  make stage4a         - Stage 4A: Temporal GNN baseline"
	@echo "  make stage4b         - Stage 4B: GNN with feature dropout"
	@echo "  make stage5          - Stage 5B: GNN + Priors (FINAL MODEL)"
	@echo ""
	@echo "Analysis:"
	@echo "  make eda             - Generate exploratory data analysis figures"
	@echo ""
	@echo "Utilities:"
	@echo "  make test            - Run tests"
	@echo "  make clean           - Clean generated files"

setup:
	@echo "Installing Python dependencies..."
	@python3 -m pip install python-dotenv pandas pyarrow numpy joblib scikit-learn xgboost torch torch-geometric torch-scatter torch-sparse requests diskcache tqdm optuna pyyaml matplotlib seaborn --quiet
	@echo "Creating directories..."
	@mkdir -p data/{raw,curated} output/{models,logs,figures,reports} tests scripts/archive
	@if [ ! -f .env ]; then cp .env.example .env; fi
	@echo "[SUCCESS] Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Add your setlist.fm API key to .env"
	@echo "  2. make verify"
	@echo "  3. make train-all"

verify:
	@echo "Verifying project setup..."
	@python3 tests/test_basic.py

stage1:
	@echo "Running Stage 1: Logistic Regression..."
	@python3 experiments/run_stage1.py 2>&1 | tee output/logs/stage1.log

stage2:
	@echo "Running Stage 2: XGBoost A/B Test..."
	@python3 experiments/run_stage2_ab_test.py 2>&1 | tee output/logs/stage2.log

stage3a:
	@echo "Running Stage 3A: Deep Learning (Tabular)..."
	@python3 experiments/run_stage3a.py 2>&1 | tee output/logs/stage3a.log

stage3b:
	@echo "Running Stage 3B: Deep Learning (Embeddings)..."
	@python3 experiments/run_stage3b.py 2>&1 | tee output/logs/stage3b.log

stage4a:
	@echo "Running Stage 4A: Temporal GNN Baseline..."
	@python3 experiments/run_stage4a_gnn_baseline.py 2>&1 | tee output/logs/stage4a.log

stage4b:
	@echo "Running Stage 4B: GNN with Dropout..."
	@python3 experiments/run_stage4b_gnn_dropout.py 2>&1 | tee output/logs/stage4b.log

stage5:
	@echo "Running Stage 5B: GNN with Priors (Final Model)..."
	@python3 experiments/run_stage5b_final_model.py 2>&1 | tee output/logs/stage5b.log

train-all:
	@echo "Running all experiments (Stages 1-5)..."
	@$(MAKE) stage1
	@$(MAKE) stage2
	@$(MAKE) stage3a
	@$(MAKE) stage3b
	@$(MAKE) stage4a
	@$(MAKE) stage4b
	@$(MAKE) stage5
	@echo "[SUCCESS] All experiments complete!"

eda:
	@echo "Generating exploratory data analysis figures..."
	@python3 scripts/generate_eda_figures.py 2>&1 | tee output/logs/eda.log

test:
	@echo "Running tests..."
	@python3 -m pytest tests/ -v

clean:
	@echo "Cleaning generated files..."
	@rm -rf output/models/**/*.pkl
	@rm -rf output/models/**/*.pt
	@rm -rf output/logs/*.log
	@rm -rf output/figures/*.png
	@rm -rf output/reports/**/*.json
	@rm -rf **/__pycache__
	@echo "[SUCCESS] Clean complete!"
