#!/usr/bin/env bash
#
# Master pipeline script for the Loss Dynamics Noise Detection experiment.
# Usage: bash run_pipeline.sh [--skip-api] [--phase1-only] [--phase2-only]
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/config.yaml"

cd "${SCRIPT_DIR}"

DATA_DIR="${SCRIPT_DIR}/data"
TRAINING_DIR="${SCRIPT_DIR}/training"
ANALYSIS_DIR="${SCRIPT_DIR}/analysis"
DOWNSTREAM_DIR="${SCRIPT_DIR}/downstream"
RESULTS_DIR="${SCRIPT_DIR}/results"
CKPT_DIR="${SCRIPT_DIR}/checkpoints"

mkdir -p "${CKPT_DIR}" "${RESULTS_DIR}/tables" "${RESULTS_DIR}/figures"

PHASE1_ONLY=false
PHASE2_ONLY=false
PHASE3_ONLY=false
PHASE4_ONLY=false
SKIP_API=false
USE_LOCAL_FALLBACK=false
PHASE5=false

for arg in "$@"; do
    case $arg in
        --phase1-only) PHASE1_ONLY=true ;;
        --phase2-only) PHASE2_ONLY=true ;;
        --phase3-only) PHASE3_ONLY=true ;;
        --phase4-only) PHASE4_ONLY=true ;;
        --skip-api) SKIP_API=true ;;
        --use-local-fallback) USE_LOCAL_FALLBACK=true ;;
        --phase5) PHASE5=true ;;
    esac
done

run_phase1() {
    echo "========== Phase 1: Data Preparation =========="

    API_FLAGS=""
    if [ "$SKIP_API" = true ]; then
        API_FLAGS="--skip-api-calls"
    elif [ "$USE_LOCAL_FALLBACK" = true ]; then
        API_FLAGS="--use-local-fallback"
    fi
    if [ "$PHASE5" = true ]; then
        API_FLAGS="$API_FLAGS --phase5"
    fi

    python "${DATA_DIR}/prepare_data.py" \
        --config "${CONFIG}" \
        --output-dir "${DATA_DIR}" \
        ${API_FLAGS}

    echo "Running quality check..."
    python "${DATA_DIR}/quality_check.py" \
        --input "${DATA_DIR}/train.jsonl" \
        --sample-size 200

    echo "Phase 1 complete."
}

run_phase2() {
    echo "========== Phase 2: Baseline Training + Signal Collection =========="

    echo "2.2: Training holdout model (1.5B, 3k clean samples, 3 epochs)..."
    python "${TRAINING_DIR}/train_holdout.py" \
        --config "${CONFIG}" \
        --model-size 1b \
        --data-path "${DATA_DIR}/train.jsonl" \
        --output-dir "${CKPT_DIR}/holdout_model" \
        --num-epochs 3 \
        --holdout-size 3000

    echo "2.3: Training main model (1.5B, full 15k, 5 epochs)..."
    python "${TRAINING_DIR}/train_main.py" \
        --config "${CONFIG}" \
        --model-size 1b \
        --data-path "${DATA_DIR}/train.jsonl" \
        --output-dir "${CKPT_DIR}/main_model" \
        --num-epochs 5

    echo "2.4: Computing IFD scores..."
    python "${TRAINING_DIR}/compute_ifd.py" \
        --config "${CONFIG}" \
        --model-path "${CKPT_DIR}/main_model/checkpoint-*" \
        --model-size 1b \
        --data-path "${DATA_DIR}/train.jsonl" \
        --output-path "${CKPT_DIR}/ifd_scores.json"

    echo "2.5: Computing RHO scores..."
    python "${TRAINING_DIR}/compute_rho.py" \
        --config "${CONFIG}" \
        --main-model-path "${CKPT_DIR}/main_model/checkpoint-*" \
        --holdout-model-path "${CKPT_DIR}/holdout_model/final" \
        --model-size 1b \
        --data-path "${DATA_DIR}/train.jsonl" \
        --output-path "${CKPT_DIR}/rho_scores.json"

    echo "Phase 2 complete."
}

run_phase3() {
    echo "========== Phase 3: Signal Analysis =========="

    echo "3.1-3.2: Computing all signals..."
    ALPHA_FLAG=""
    if [ "$PHASE5" = true ]; then
        COMPOSITE_FLAG="--composite-mode zscore"
    else
        COMPOSITE_FLAG=""
    fi
    python "${ANALYSIS_DIR}/compute_signals.py" \
        --losses-path "${CKPT_DIR}/main_model/all_losses.json" \
        --token-losses-path "${CKPT_DIR}/main_model/token_losses_epoch_3.json" \
        --ifd-path "${CKPT_DIR}/ifd_scores.json" \
        --rho-path "${CKPT_DIR}/rho_scores.json" \
        --data-path "${DATA_DIR}/train.jsonl" \
        --output-path "${RESULTS_DIR}/signals.csv" \
        --output-json "${RESULTS_DIR}/signals.json" \
        ${COMPOSITE_FLAG}

    echo "3.3: Q1 analysis..."
    python "${ANALYSIS_DIR}/q1_analysis.py" \
        --signals-path "${RESULTS_DIR}/signals.json" \
        --output-dir "${RESULTS_DIR}/tables"

    echo "3.4: Q2 analysis..."
    python "${ANALYSIS_DIR}/q2_analysis.py" \
        --signals-path "${RESULTS_DIR}/signals.json" \
        --output-dir "${RESULTS_DIR}/tables"

    echo "3.5: Q3 analysis..."
    python "${ANALYSIS_DIR}/q3_analysis.py" \
        --signals-path "${RESULTS_DIR}/signals.json" \
        --output-dir "${RESULTS_DIR}/tables"

    echo "3.6: Q4 analysis..."
    python "${ANALYSIS_DIR}/q4_analysis.py" \
        --signals-path "${RESULTS_DIR}/signals.json" \
        --output-dir "${RESULTS_DIR}/tables"

    echo "Generating visualizations..."
    python "${ANALYSIS_DIR}/visualize.py" \
        --signals-path "${RESULTS_DIR}/signals.json" \
        --q3-results "${RESULTS_DIR}/tables/q3_results.json" \
        --q4-results "${RESULTS_DIR}/tables/q4_results.json" \
        --output-dir "${RESULTS_DIR}/figures"

    echo "Phase 3 complete."
}

run_phase4() {
    echo "========== Phase 4: Downstream Verification =========="

    echo "4.3: Training filtered models (3B)..."
    python "${DOWNSTREAM_DIR}/train_filtered.py" \
        --config "${CONFIG}" \
        --checkpoint-path "${CKPT_DIR}/main_model/checkpoint-*" \
        --data-path "${DATA_DIR}/train.jsonl" \
        --signals-path "${RESULTS_DIR}/signals.json" \
        --output-dir "${CKPT_DIR}/filtered_models" \
        --strategies full p1_filtered rho_filtered random_drop ifd_only \
        --drop-ratio 0.10 \
        --num-epochs 2

    echo "4.4-4.5: Evaluation..."
    python "${DOWNSTREAM_DIR}/evaluate.py" \
        --config "${CONFIG}" \
        --model-paths \
            "full:${CKPT_DIR}/filtered_models/full/final" \
            "p1_filtered:${CKPT_DIR}/filtered_models/p1_filtered/final" \
            "rho_filtered:${CKPT_DIR}/filtered_models/rho_filtered/final" \
            "random_drop:${CKPT_DIR}/filtered_models/random_drop/final" \
            "ifd_only:${CKPT_DIR}/filtered_models/ifd_only/final" \
        --output-dir "${RESULTS_DIR}/tables"

    echo "4.6: Manual inspection samples..."
    python "${DOWNSTREAM_DIR}/manual_inspection.py" \
        --data-path "${DATA_DIR}/train.jsonl" \
        --signals-path "${RESULTS_DIR}/signals.json" \
        --output-path "${RESULTS_DIR}/tables/inspection_samples.csv" \
        --n-samples 50

    echo "Phase 4 complete."
}

run_phase5() {
    echo "========== Phase 5: Summary =========="
    python "${SCRIPT_DIR}/summarize.py" \
        --results-dir "${RESULTS_DIR}/tables" \
        --signals-path "${RESULTS_DIR}/signals.json" \
        --evaluation-results "${RESULTS_DIR}/tables/evaluation_results.json" \
        --output-path "${RESULTS_DIR}/summary.md"

    echo "Phase 5 complete."
    echo ""
    echo "Experiment pipeline finished."
    echo "Results directory: ${RESULTS_DIR}"
    echo "Summary: ${RESULTS_DIR}/summary.md"
}

# Execution logic
if [ "$PHASE1_ONLY" = true ]; then
    run_phase1
elif [ "$PHASE2_ONLY" = true ]; then
    run_phase2
elif [ "$PHASE3_ONLY" = true ]; then
    run_phase3
elif [ "$PHASE4_ONLY" = true ]; then
    run_phase4
else
    run_phase1
    run_phase2
    run_phase3
    run_phase5
    echo ""
    echo "To run downstream verification (Phase 4):"
    echo "  bash run_pipeline.sh --phase4-only"
fi
