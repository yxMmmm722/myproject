#!/usr/bin/env bash
set -euo pipefail

# Re-run best configuration per dataset from grid_search_hcar.py output.
# Usage example:
#   # one command: auto grid-search -> rerun best
#   bash scripts/run_best_from_grid.sh
#   # or reuse an existing grid_results.csv
#   bash scripts/run_best_from_grid.sh ./logs/grid_search_xxx/grid_results.csv

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_FILE="${RUN_FILE:-run.py}"
GRID_SEARCH_SCRIPT="${GRID_SEARCH_SCRIPT:-scripts/grid_search_hcar.py}"
GRID_CSV="${1:-${GRID_CSV:-}}"
FORCE_GRID_SEARCH="${FORCE_GRID_SEARCH:-0}"  # 1 => always run grid search first even if GRID_CSV is provided

# Runtime overrides.
SEQ_LEN="${SEQ_LEN:-96}"
LABEL_LEN="${LABEL_LEN:-48}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MODEL_PREFIX="${MODEL_PREFIX:-HCARBest}"
SELECTION_METRIC="${SELECTION_METRIC:-vali_loss_min}"  # vali_loss_min|mse

# Optional filters.
DATASETS="${DATASETS:-}"        # e.g. "ETTh1,ETTm1,weather"
PRED_LEN="${PRED_LEN:-}"        # e.g. "96"
GROUP_BY_PRED_LEN="${GROUP_BY_PRED_LEN:-1}" # 1 => best per (dataset,pred_len); 0 => best per dataset
GRID_PRED_LENS="${GRID_PRED_LENS:-96,192,336,720}"  # used when PRED_LEN is empty

# Grid-search controls for auto stage.
GRID_LOOKBACK_GRID="${GRID_LOOKBACK_GRID:-}"
GRID_LEARNING_RATE_GRID="${GRID_LEARNING_RATE_GRID:-}"
GRID_TOPM_GRID="${GRID_TOPM_GRID:-10,20,40}"
GRID_COARSE_K_GRID="${GRID_COARSE_K_GRID:-40,80,160}"
GRID_N_PERIOD_GRID="${GRID_N_PERIOD_GRID:-2,3}"
GRID_CONTEXT_DIM_GRID="${GRID_CONTEXT_DIM_GRID:-32,64}"
GRID_RETRIEVAL_ALPHA_GRID="${GRID_RETRIEVAL_ALPHA_GRID:-0.7}"
GRID_MAX_TRIALS="${GRID_MAX_TRIALS:-0}"
GRID_MODEL_PREFIX="${GRID_MODEL_PREFIX:-HCARGrid}"

# Optional feature toggles.
USE_AMP="${USE_AMP:-1}"  # 1 => add --use_amp
FREEZE_CONTEXT_ENCODER="${FREEZE_CONTEXT_ENCODER:-0}"
NO_REFRESH_CONTEXT_EACH_EPOCH="${NO_REFRESH_CONTEXT_EACH_EPOCH:-0}"
RETRIEVAL_CACHE_DEVICE="${RETRIEVAL_CACHE_DEVICE:-gpu}"  # cpu|gpu
TRAFFIC_CACHE_DEVICE="${TRAFFIC_CACHE_DEVICE:-cpu}"  # cpu|gpu, used only for traffic

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-./logs/run_best_from_grid_${TIMESTAMP}}"
GRID_OUTPUT_DIR="${GRID_OUTPUT_DIR:-${LOG_DIR}/grid_search}"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${RUN_FILE}" ]]; then
  echo "[FATAL] Cannot find ${RUN_FILE}. Please run this script in RAFT-main root."
  exit 1
fi
if [[ ! -f "${GRID_SEARCH_SCRIPT}" ]]; then
  echo "[FATAL] Cannot find ${GRID_SEARCH_SCRIPT}. Please run this script in RAFT-main root."
  exit 1
fi

AUTO_PRED_LENS="${GRID_PRED_LENS}"
if [[ -n "${PRED_LEN}" ]]; then
  AUTO_PRED_LENS="${PRED_LEN}"
fi

# Stage-1: optional auto grid search
if [[ "${FORCE_GRID_SEARCH}" == "1" || -z "${GRID_CSV}" ]]; then
  echo "============================================================"
  echo "[AUTO] Running grid search first..."
  echo "[AUTO] output_dir=${GRID_OUTPUT_DIR}"
  echo "[AUTO] datasets=${DATASETS:-<default>}"
  echo "[AUTO] pred_lens=${AUTO_PRED_LENS}"
  echo "============================================================"

  grid_cmd=(
    "${PYTHON_BIN}" "${GRID_SEARCH_SCRIPT}"
    --python_bin "${PYTHON_BIN}"
    --run_file "${RUN_FILE}"
    --output_dir "${GRID_OUTPUT_DIR}"
    --pred_lens "${AUTO_PRED_LENS}"
    --model_prefix "${GRID_MODEL_PREFIX}"
    --retrieval_cache_device "${RETRIEVAL_CACHE_DEVICE}"
    --traffic_cache_device "${TRAFFIC_CACHE_DEVICE}"
    --seq_len "${SEQ_LEN}"
    --label_len "${LABEL_LEN}"
    --batch_size "${BATCH_SIZE}"
    --train_epochs "${TRAIN_EPOCHS}"
    --learning_rate "${LEARNING_RATE}"
    --num_workers "${NUM_WORKERS}"
    --selection_metric "${SELECTION_METRIC}"
    --max_trials "${GRID_MAX_TRIALS}"
    --topm_grid "${GRID_TOPM_GRID}"
    --coarse_k_grid "${GRID_COARSE_K_GRID}"
    --n_period_grid "${GRID_N_PERIOD_GRID}"
    --context_dim_grid "${GRID_CONTEXT_DIM_GRID}"
    --retrieval_alpha_grid "${GRID_RETRIEVAL_ALPHA_GRID}"
  )

  if [[ -n "${DATASETS}" ]]; then
    grid_cmd+=(--datasets "${DATASETS}")
  fi
  if [[ -n "${GRID_LOOKBACK_GRID}" ]]; then
    grid_cmd+=(--lookback_grid "${GRID_LOOKBACK_GRID}")
  fi
  if [[ -n "${GRID_LEARNING_RATE_GRID}" ]]; then
    grid_cmd+=(--learning_rate_grid "${GRID_LEARNING_RATE_GRID}")
  fi
  if [[ "${USE_AMP}" == "1" ]]; then
    grid_cmd+=(--use_amp)
  fi
  if [[ "${FREEZE_CONTEXT_ENCODER}" == "1" ]]; then
    grid_cmd+=(--freeze_context_encoder)
  fi
  if [[ "${NO_REFRESH_CONTEXT_EACH_EPOCH}" == "1" ]]; then
    grid_cmd+=(--no_refresh_context_each_epoch)
  fi

  "${grid_cmd[@]}"
  GRID_CSV="${GRID_OUTPUT_DIR}/grid_results.csv"
fi

if [[ -z "${GRID_CSV}" ]]; then
  echo "[FATAL] Missing grid_results.csv path after auto stage."
  exit 1
fi
if [[ ! -f "${GRID_CSV}" ]]; then
  echo "[FATAL] grid_results.csv not found: ${GRID_CSV}"
  exit 1
fi

declare -A ROOT_PATHS=(
  [ETTh1]="./data/ETT"
  [ETTh2]="./data/ETT"
  [ETTm1]="./data/ETT"
  [ETTm2]="./data/ETT"
  [electricity]="./data/electricity"
  [exchange_rate]="./data/exchange_rate"
  [illness]="./data/illness"
  [traffic]="./data/traffic"
  [weather]="./data/weather"
)

declare -A DATA_FILES=(
  [ETTh1]="ETTh1.csv"
  [ETTh2]="ETTh2.csv"
  [ETTm1]="ETTm1.csv"
  [ETTm2]="ETTm2.csv"
  [electricity]="electricity.csv"
  [exchange_rate]="exchange_rate.csv"
  [illness]="national_illness.csv"
  [traffic]="traffic.csv"
  [weather]="weather.csv"
)

BEST_TSV="${LOG_DIR}/best_candidates.tsv"

"${PYTHON_BIN}" - "${GRID_CSV}" "${DATASETS}" "${PRED_LEN}" "${GROUP_BY_PRED_LEN}" "${SELECTION_METRIC}" > "${BEST_TSV}" <<'PY'
import csv
import math
import sys

grid_csv = sys.argv[1]
datasets_filter = {x.strip() for x in sys.argv[2].split(",") if x.strip()}
pred_len_filter = sys.argv[3].strip()
group_by_pred_len = sys.argv[4].strip() == "1"
selection_metric = sys.argv[5].strip() or "vali_loss_min"
if selection_metric not in {"vali_loss_min", "mse"}:
    raise SystemExit(f"Unsupported selection metric: {selection_metric}. Use vali_loss_min or mse.")

def to_float(v, default=math.inf):
    try:
        return float(v)
    except Exception:
        return default

def score_of(row):
    if selection_metric == "mse":
        return to_float(row.get("mse", ""))
    return to_float(row.get("vali_loss_min", ""))

best = {}

with open(grid_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        dataset = row.get("dataset", "").strip()
        if not dataset:
            continue
        if datasets_filter and dataset not in datasets_filter:
            continue
        pred_len = row.get("pred_len", "").strip()
        if pred_len_filter and pred_len != pred_len_filter:
            continue

        rc = row.get("return_code", "").strip()
        if rc not in {"0", "0.0"}:
            continue

        cur_score = score_of(row)
        if not math.isfinite(cur_score):
            continue
        mse = to_float(row.get("mse", ""))
        mae = to_float(row.get("mae", ""))
        vali = to_float(row.get("vali_loss_min", ""))

        key = (dataset, pred_len) if group_by_pred_len else (dataset,)
        prev = best.get(key)
        if prev is None:
            best[key] = row
            continue

        prev_score = score_of(prev)
        prev_mse = to_float(prev.get("mse", ""))
        prev_mae = to_float(prev.get("mae", ""))
        prev_vali = to_float(prev.get("vali_loss_min", ""))

        better = False
        if cur_score < prev_score:
            better = True
        elif cur_score == prev_score:
            # stable tie-break: mse -> mae -> vali
            if mse < prev_mse:
                better = True
            elif mse == prev_mse and mae < prev_mae:
                better = True
            elif mse == prev_mse and mae == prev_mae and vali < prev_vali:
                better = True

        if better:
            best[key] = row

for key in sorted(best.keys()):
    row = best[key]
    # dataset, pred_len, seq_len, learning_rate, topm, coarse_k, n_period, context_dim,
    # retrieval_alpha, channels, best_score, mse, mae
    score = score_of(row)
    print(
        "\t".join(
            [
                row.get("dataset", "").strip(),
                row.get("pred_len", "").strip(),
                row.get("seq_len", "").strip(),
                row.get("learning_rate", "").strip(),
                row.get("topm", "").strip(),
                row.get("coarse_k", "").strip(),
                row.get("n_period", "").strip(),
                row.get("context_dim", "").strip(),
                row.get("retrieval_alpha", "").strip(),
                row.get("channels", "").strip(),
                str(score),
                row.get("mse", "").strip(),
                row.get("mae", "").strip(),
            ]
        )
    )
PY

if [[ ! -s "${BEST_TSV}" ]]; then
  echo "[FATAL] No valid best candidates found from ${GRID_CSV}."
  echo "Check whether this CSV is produced by scripts/grid_search_hcar.py and has successful trials."
  exit 1
fi

SUCCESS_RUNS=()
FAILED_RUNS=()
SKIPPED_RUNS=()

while IFS=$'\t' read -r dataset pred_len best_seq_len best_learning_rate topm coarse_k n_period context_dim retrieval_alpha channels best_score best_mse best_mae; do
  root_path="${ROOT_PATHS[${dataset}]:-}"
  data_file="${DATA_FILES[${dataset}]:-}"
  if [[ -z "${root_path}" || -z "${data_file}" ]]; then
    echo "[SKIP] ${dataset}: no dataset mapping."
    SKIPPED_RUNS+=("${dataset}")
    continue
  fi

  csv_path="${root_path}/${data_file}"
  if [[ ! -f "${csv_path}" ]]; then
    echo "[SKIP] ${dataset}: missing ${csv_path}"
    SKIPPED_RUNS+=("${dataset}")
    continue
  fi

  if [[ -z "${channels}" || "${channels}" == "None" ]]; then
    channels="$(head -n 1 "${csv_path}" | awk -F',' '{print NF-1}')"
  fi

  seq_len_run="${best_seq_len}"
  if [[ -z "${seq_len_run}" || "${seq_len_run}" == "None" ]]; then
    seq_len_run="${SEQ_LEN}"
  fi
  learning_rate_run="${best_learning_rate}"
  if [[ -z "${learning_rate_run}" || "${learning_rate_run}" == "None" ]]; then
    learning_rate_run="${LEARNING_RATE}"
  fi

  retrieval_cache_device_run="${RETRIEVAL_CACHE_DEVICE}"
  if [[ "${dataset}" == "traffic" ]]; then
    retrieval_cache_device_run="${TRAFFIC_CACHE_DEVICE}"
  fi

  model_id="${MODEL_PREFIX}_${dataset}_pl${pred_len}_lb${seq_len_run}_lr${learning_rate_run}_m${topm}_k${coarse_k}_p${n_period}_cd${context_dim}"
  log_file="${LOG_DIR}/${model_id}.log"

  echo "============================================================"
  echo "[RUN ] dataset=${dataset} pred_len=${pred_len} channels=${channels}"
  echo "[BEST] metric=${SELECTION_METRIC} score=${best_score} mse=${best_mse} mae=${best_mae}"
  echo "[CFG ] seq_len=${seq_len_run} lr=${learning_rate_run} topm=${topm} coarse_k=${coarse_k} n_period=${n_period} context_dim=${context_dim} retrieval_alpha=${retrieval_alpha} cache=${retrieval_cache_device_run}"
  echo "[LOG ] ${log_file}"
  echo "============================================================"

  cmd=(
    "${PYTHON_BIN}" "${RUN_FILE}"
    --data "${dataset}"
    --root_path "${root_path}"
    --data_path "${data_file}"
    --model_id "${model_id}"
    --is_training 1
    --seq_len "${seq_len_run}" --label_len "${LABEL_LEN}" --pred_len "${pred_len}"
    --enc_in "${channels}" --dec_in "${channels}" --c_out "${channels}"
    --n_period "${n_period}" --topm "${topm}"
    --retrieval_coarse_k "${coarse_k}"
    --retrieval_alpha "${retrieval_alpha}"
    --context_dim "${context_dim}"
    --retrieval_cache_device "${retrieval_cache_device_run}"
    --online_retrieval
    --batch_size "${BATCH_SIZE}" --train_epochs "${TRAIN_EPOCHS}"
    --learning_rate "${learning_rate_run}" --lradj cosine --num_workers "${NUM_WORKERS}"
  )

  if [[ "${USE_AMP}" == "1" ]]; then
    cmd+=(--use_amp)
  fi
  if [[ "${FREEZE_CONTEXT_ENCODER}" == "1" ]]; then
    cmd+=(--freeze_context_encoder)
  fi
  if [[ "${NO_REFRESH_CONTEXT_EACH_EPOCH}" == "1" ]]; then
    cmd+=(--no_refresh_context_each_epoch)
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee "${log_file}"
  status=${PIPESTATUS[0]}
  set -e

  if [[ ${status} -eq 0 ]]; then
    echo "[OK  ] ${dataset} pl${pred_len}"
    SUCCESS_RUNS+=("${dataset}:pl${pred_len}")
  else
    echo "[FAIL] ${dataset} pl${pred_len} (exit=${status})"
    FAILED_RUNS+=("${dataset}:pl${pred_len}")
  fi
done < "${BEST_TSV}"

echo
echo "========================== SUMMARY =========================="
echo "Grid CSV: ${GRID_CSV}"
echo "Candidates TSV: ${BEST_TSV}"
echo "Selection metric: ${SELECTION_METRIC}"
echo "Log directory: ${LOG_DIR}"
echo "Success (${#SUCCESS_RUNS[@]}): ${SUCCESS_RUNS[*]:-none}"
echo "Failed  (${#FAILED_RUNS[@]}): ${FAILED_RUNS[*]:-none}"
echo "Skipped (${#SKIPPED_RUNS[@]}): ${SKIPPED_RUNS[*]:-none}"
echo "============================================================"
