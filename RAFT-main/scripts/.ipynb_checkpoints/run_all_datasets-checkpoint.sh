#!/usr/bin/env bash
set -u

# One-click runner for all built-in datasets in this repo.
# It auto-detects feature dimension from each CSV header so enc_in/dec_in/c_out
# always match the dataset.

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_FILE="${RUN_FILE:-run.py}"
# Training hyper-parameters (override by env vars if needed).
SEQ_LEN="${SEQ_LEN:-720}"
LABEL_LEN="${LABEL_LEN:-48}"
PRED_LENS_STR="${PRED_LENS:-96 192 336 720}"  # default: paper-style horizons
ILLNESS_PRED_LENS_STR="${ILLNESS_PRED_LENS:-24 36 48 60}"  # dataset-specific default for illness
N_PERIOD="${N_PERIOD:-3}"
TOPM="${TOPM:-20}"
RETRIEVAL_TEMPERATURE="${RETRIEVAL_TEMPERATURE:-0.1}"
META_ONLY_RETRIEVAL="${META_ONLY_RETRIEVAL:-1}"  # 1: use one-shot meta-context retrieval only
COMPARE_RETRIEVAL_TOPM="${COMPARE_RETRIEVAL_TOPM:-${COMPARE_RETRIEVAL_TOPK:-0}}"  # 1: compare top-m overlap (default off for pure meta runs)
COMPARE_RETRIEVAL_FUTURE_QUALITY="${COMPARE_RETRIEVAL_FUTURE_QUALITY:-0}"  # 1: quantify wave/meta retrieval future quality on full split
SAVE_RETRIEVAL_CASES="${SAVE_RETRIEVAL_CASES:-0}"  # 1: save one test case panel of wave/meta top-m histories & futures
RETRIEVAL_CASE_PERIOD_IDX="${RETRIEVAL_CASE_PERIOD_IDX:--1}"  # -1 means last period scale
RETRIEVAL_CASE_CHANNEL_IDX="${RETRIEVAL_CASE_CHANNEL_IDX:--1}"  # -1 means last channel
RETRIEVAL_CASE_SAMPLE_IDX="${RETRIEVAL_CASE_SAMPLE_IDX:-0}"  # sample index in first test batch
RETRIEVAL_CASE_NUM_SAMPLES="${RETRIEVAL_CASE_NUM_SAMPLES:-1}"  # number of samples in first test batch to visualize
RETRIEVAL_CASE_ALL_PERIODS="${RETRIEVAL_CASE_ALL_PERIODS:-0}"  # 1: visualize all period scales
RETRIEVAL_CASE_ALL_CHANNELS="${RETRIEVAL_CASE_ALL_CHANNELS:-0}"  # 1: visualize all channels
RETRIEVAL_CASE_FIRST_LAST_SAMPLES="${RETRIEVAL_CASE_FIRST_LAST_SAMPLES:-0}"  # 1: visualize only first/last sample in first test batch
RETRIEVAL_CASE_FIRST_LAST_BATCHES="${RETRIEVAL_CASE_FIRST_LAST_BATCHES:-0}"  # 1: visualize first and last test batch
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
LRADJ="${LRADJ:-type1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MODEL_PREFIX="${MODEL_PREFIX:-HCAR}"
USE_TABLE6_PRESETS="${USE_TABLE6_PRESETS:-1}"  # 1: use dataset/pred specific seq_len/lr/topm from Table 6
USE_AMP="${USE_AMP:-0}"  # 1: enable --use_amp
RETRIEVAL_CACHE_DEVICE="${RETRIEVAL_CACHE_DEVICE:-gpu}"  # cpu|gpu
TRAFFIC_CACHE_DEVICE="${TRAFFIC_CACHE_DEVICE:-cpu}"  # cpu|gpu, used only for traffic

# Dataset list (override by env var if needed).
DATASETS_STR="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 electricity exchange_rate illness traffic weather}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-./logs/run_all_${TIMESTAMP}}"
mkdir -p "${LOG_DIR}"

# Guard thread env vars: libgomp requires OMP_NUM_THREADS to be a positive integer.
sanitize_thread_env() {
  local var_name="$1"
  local default_val="$2"
  local cur_val="${!var_name:-}"
  if [[ -z "${cur_val}" || ! "${cur_val}" =~ ^[0-9]+$ || "${cur_val}" -le 0 ]]; then
    export "${var_name}=${default_val}"
  fi
}

sanitize_thread_env OMP_NUM_THREADS 1
sanitize_thread_env MKL_NUM_THREADS "${OMP_NUM_THREADS}"
sanitize_thread_env OPENBLAS_NUM_THREADS "${OMP_NUM_THREADS}"
sanitize_thread_env NUMEXPR_NUM_THREADS "${OMP_NUM_THREADS}"

if [[ ! -f "${RUN_FILE}" ]]; then
  echo "[FATAL] Cannot find ${RUN_FILE}. Please run this script in RAFT-main root."
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

read -r -a DATASET_LIST <<< "${DATASETS_STR}"
read -r -a PRED_LEN_LIST <<< "${PRED_LENS_STR}"

SUCCESS_RUNS=()
FAILED_RUNS=()
SKIPPED_RUNS=()
declare -A MSE_BY_KEY

declare -A TABLE6_SEQ_LEN
declare -A TABLE6_LR
declare -A TABLE6_TOPM

set_table6_preset() {
  local dataset="$1"
  local pred_len="$2"
  local seq_len="$3"
  local lr="$4"
  local topm="$5"
  local key="${dataset}|${pred_len}"
  TABLE6_SEQ_LEN["${key}"]="${seq_len}"
  TABLE6_LR["${key}"]="${lr}"
  TABLE6_TOPM["${key}"]="${topm}"
}

# Table 6 presets (dataset x horizon): lookback(seq_len), learning rate, number of retrievals(topm).
# ETTh1
set_table6_preset ETTh1 96 720 1e-3 20
set_table6_preset ETTh1 192 720 1e-2 20
set_table6_preset ETTh1 336 720 1e-2 20
set_table6_preset ETTh1 720 720 1e-4 20
# ETTh2
set_table6_preset ETTh2 96 720 1e-2 10
set_table6_preset ETTh2 192 720 1e-3 10
set_table6_preset ETTh2 336 720 1e-3 20
set_table6_preset ETTh2 720 720 1e-4 20
# ETTm1
set_table6_preset ETTm1 96 720 1e-2 1
set_table6_preset ETTm1 192 720 1e-3 20
set_table6_preset ETTm1 336 720 1e-3 20
set_table6_preset ETTm1 720 720 1e-2 20
# ETTm2
set_table6_preset ETTm2 96 720 1e-3 5
set_table6_preset ETTm2 192 720 1e-3 20
set_table6_preset ETTm2 336 720 1e-4 20
set_table6_preset ETTm2 720 720 1e-4 20
# Electricity
set_table6_preset electricity 96 720 1e-2 1
set_table6_preset electricity 192 720 1e-3 1
set_table6_preset electricity 336 720 1e-3 1
set_table6_preset electricity 720 720 1e-3 1
# Exchange (mapped to exchange_rate here)
set_table6_preset exchange_rate 96 720 1e-4 1
set_table6_preset exchange_rate 192 720 1e-3 1
set_table6_preset exchange_rate 336 720 1e-3 10
set_table6_preset exchange_rate 720 720 1e-4 20
# Illness
set_table6_preset illness 24 96 1e-2 1
set_table6_preset illness 36 96 1e-2 1
set_table6_preset illness 48 96 1e-2 20
set_table6_preset illness 60 96 1e-2 20
# Traffic
set_table6_preset traffic 96 720 1e-2 1
set_table6_preset traffic 192 720 1e-3 1
set_table6_preset traffic 336 720 1e-3 1
set_table6_preset traffic 720 720 1e-3 1
# Weather
set_table6_preset weather 96 720 1e-2 1
set_table6_preset weather 192 720 1e-3 1
set_table6_preset weather 336 720 1e-3 1
set_table6_preset weather 720 720 1e-3 1

extract_mse_from_log() {
  local log_file="$1"
  local mse
  mse="$(
    grep -Eo 'mse:[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?' "${log_file}" 2>/dev/null \
      | tail -n 1 \
      | cut -d':' -f2
  )"
  echo "${mse}"
}

is_custom_dataset() {
  local dataset="$1"
  case "${dataset}" in
    electricity|exchange_rate|illness|traffic|weather)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_window_valid_for_custom_split() {
  local csv_path="$1"
  local seq_len="$2"
  local pred_len="$3"

  # Number of samples excluding header.
  local total_rows
  total_rows=$(( $(wc -l < "${csv_path}") - 1 ))
  if [[ ${total_rows} -le 0 ]]; then
    return 1
  fi

  local num_train num_test num_val
  num_train=$(( total_rows * 7 / 10 ))
  num_test=$(( total_rows * 2 / 10 ))
  num_val=$(( total_rows - num_train - num_test ))

  # Dataset_Custom window counts:
  # train_len = num_train - seq_len - pred_len + 1
  # val_len   = num_val - pred_len + 1
  # test_len  = num_test - pred_len + 1
  local train_len val_len test_len
  train_len=$(( num_train - seq_len - pred_len + 1 ))
  val_len=$(( num_val - pred_len + 1 ))
  test_len=$(( num_test - pred_len + 1 ))

  if [[ ${train_len} -le 0 || ${val_len} -le 0 || ${test_len} -le 0 ]]; then
    return 1
  fi
  return 0
}

run_one() {
  local dataset="$1"
  local pred_len="$2"
  local preset_key="${dataset}|${pred_len}"
  local seq_len="${SEQ_LEN}"
  local learning_rate="${LEARNING_RATE}"
  local topm="${TOPM}"
  local retrieval_cache_device_run="${RETRIEVAL_CACHE_DEVICE}"
  local num_workers_run="${NUM_WORKERS}"
  local preset_source="default"
  local root_path="${ROOT_PATHS[${dataset}]:-}"
  local data_file="${DATA_FILES[${dataset}]:-}"

  if [[ "${dataset}" == "traffic" ]]; then
    retrieval_cache_device_run="${TRAFFIC_CACHE_DEVICE}"
    num_workers_run=0
  fi

  if [[ "${USE_TABLE6_PRESETS}" == "1" ]]; then
    if [[ -n "${TABLE6_SEQ_LEN["${preset_key}"]:-}" ]]; then
      seq_len="${TABLE6_SEQ_LEN["${preset_key}"]}"
    fi
    if [[ -n "${TABLE6_LR["${preset_key}"]:-}" ]]; then
      learning_rate="${TABLE6_LR["${preset_key}"]}"
    fi
    if [[ -n "${TABLE6_TOPM["${preset_key}"]:-}" ]]; then
      topm="${TABLE6_TOPM["${preset_key}"]}"
    fi
    if [[ -n "${TABLE6_SEQ_LEN["${preset_key}"]:-}" || -n "${TABLE6_LR["${preset_key}"]:-}" || -n "${TABLE6_TOPM["${preset_key}"]:-}" ]]; then
      preset_source="table6"
    fi
  fi

  if [[ -z "${root_path}" || -z "${data_file}" ]]; then
    echo "[SKIP] ${dataset}: no mapping in script."
    SKIPPED_RUNS+=("${dataset}:pl${pred_len}")
    return 0
  fi

  local csv_path="${root_path}/${data_file}"
  if [[ ! -f "${csv_path}" ]]; then
    echo "[SKIP] ${dataset}: missing file ${csv_path}"
    SKIPPED_RUNS+=("${dataset}:pl${pred_len}")
    return 0
  fi

  if is_custom_dataset "${dataset}"; then
    if ! is_window_valid_for_custom_split "${csv_path}" "${seq_len}" "${pred_len}"; then
      echo "[SKIP] ${dataset}:pl${pred_len} invalid window config for Dataset_Custom split (seq_len=${seq_len}, pred_len=${pred_len})."
      SKIPPED_RUNS+=("${dataset}:pl${pred_len}")
      return 0
    fi
  fi

  local channels
  channels="$(head -n 1 "${csv_path}" | awk -F',' '{print NF-1}')"
  if [[ -z "${channels}" || "${channels}" -le 0 ]]; then
    echo "[FAIL] ${dataset}: cannot infer channel count from ${csv_path}"
    FAILED_RUNS+=("${dataset}:pl${pred_len}")
    return 0
  fi

  local model_id="${MODEL_PREFIX}_${dataset}_pl${pred_len}"
  local log_file="${LOG_DIR}/${model_id}.log"

  echo "============================================================"
  echo "[RUN ] dataset=${dataset} pred_len=${pred_len} channels=${channels}"
  echo "[CFG ] seq_len=${seq_len} lr=${learning_rate} lradj=${LRADJ} topm=${topm} temp=${RETRIEVAL_TEMPERATURE} meta_only=${META_ONLY_RETRIEVAL} cmp_topm=${COMPARE_RETRIEVAL_TOPM}:${topm} save_case=${SAVE_RETRIEVAL_CASES} case_n=${RETRIEVAL_CASE_NUM_SAMPLES} case_all_p=${RETRIEVAL_CASE_ALL_PERIODS} case_all_c=${RETRIEVAL_CASE_ALL_CHANNELS} case_fl_s=${RETRIEVAL_CASE_FIRST_LAST_SAMPLES} case_fl_b=${RETRIEVAL_CASE_FIRST_LAST_BATCHES} preset=${preset_source} cache=${retrieval_cache_device_run}"
  echo "[LOG ] ${log_file}"
  echo "============================================================"

  local cmd=(
    "${PYTHON_BIN}" "${RUN_FILE}"
    --data "${dataset}"
    --root_path "${root_path}"
    --data_path "${data_file}"
    --model_id "${model_id}"
    --is_training 1
    --seq_len "${seq_len}" --label_len "${LABEL_LEN}" --pred_len "${pred_len}"
    --enc_in "${channels}" --dec_in "${channels}" --c_out "${channels}"
    --n_period "${N_PERIOD}" --topm "${topm}"
    --retrieval_temperature "${RETRIEVAL_TEMPERATURE}"
    --retrieval_cache_device "${retrieval_cache_device_run}"
    --batch_size "${BATCH_SIZE}" --train_epochs "${TRAIN_EPOCHS}"
    --learning_rate "${learning_rate}" --lradj "${LRADJ}" --num_workers "${num_workers_run}"
  )

  if [[ "${USE_AMP}" == "1" ]]; then
    cmd+=(--use_amp)
  fi
  if [[ "${META_ONLY_RETRIEVAL}" == "1" ]]; then
    cmd+=(--meta_only_retrieval)
  fi
  if [[ "${COMPARE_RETRIEVAL_TOPM}" == "1" ]]; then
    cmd+=(--compare_retrieval_topm)
  fi
  if [[ "${COMPARE_RETRIEVAL_FUTURE_QUALITY}" == "1" ]]; then
    cmd+=(--compare_retrieval_future_quality)
  fi
  if [[ "${SAVE_RETRIEVAL_CASES}" == "1" ]]; then
    cmd+=(--save_retrieval_cases)
    cmd+=(--retrieval_case_period_idx "${RETRIEVAL_CASE_PERIOD_IDX}")
    cmd+=(--retrieval_case_channel_idx "${RETRIEVAL_CASE_CHANNEL_IDX}")
    cmd+=(--retrieval_case_sample_idx "${RETRIEVAL_CASE_SAMPLE_IDX}")
    cmd+=(--retrieval_case_num_samples "${RETRIEVAL_CASE_NUM_SAMPLES}")
    if [[ "${RETRIEVAL_CASE_ALL_PERIODS}" == "1" ]]; then
      cmd+=(--retrieval_case_all_periods)
    fi
    if [[ "${RETRIEVAL_CASE_ALL_CHANNELS}" == "1" ]]; then
      cmd+=(--retrieval_case_all_channels)
    fi
    if [[ "${RETRIEVAL_CASE_FIRST_LAST_SAMPLES}" == "1" ]]; then
      cmd+=(--retrieval_case_first_last_samples)
    fi
    if [[ "${RETRIEVAL_CASE_FIRST_LAST_BATCHES}" == "1" ]]; then
      cmd+=(--retrieval_case_first_last_batches)
    fi
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e

  if [[ ${status} -eq 0 ]]; then
    echo "[OK  ] ${dataset}:pl${pred_len}"
    SUCCESS_RUNS+=("${dataset}:pl${pred_len}")
    local key="${dataset}|${pred_len}"
    local mse
    mse="$(extract_mse_from_log "${log_file}")"
    if [[ -n "${mse}" ]]; then
      MSE_BY_KEY["${key}"]="${mse}"
      echo "[MSE ] ${dataset}:pl${pred_len} mse=${mse}"
    else
      echo "[WARN] ${dataset}:pl${pred_len} finished but mse was not found in ${log_file}"
    fi
  else
    echo "[FAIL] ${dataset}:pl${pred_len} (exit=${status})"
    FAILED_RUNS+=("${dataset}:pl${pred_len}")
  fi
}

set -e
for d in "${DATASET_LIST[@]}"; do
  # Dataset-specific horizon override: illness uses 24/36/48/60 by default.
  CUR_PRED_LEN_LIST=("${PRED_LEN_LIST[@]}")
  if [[ "${d}" == "illness" && -z "${PRED_LENS:-}" ]]; then
    read -r -a CUR_PRED_LEN_LIST <<< "${ILLNESS_PRED_LENS_STR}"
  fi

  for p in "${CUR_PRED_LEN_LIST[@]}"; do
    run_one "${d}" "${p}"
  done
done

MSE_BY_RUN_CSV="${LOG_DIR}/mse_by_run.csv"
MSE_AVG_CSV="${LOG_DIR}/mse_avg_over_horizons.csv"

{
  echo "dataset,pred_len,mse"
  for d in "${DATASET_LIST[@]}"; do
    CUR_PRED_LEN_LIST=("${PRED_LEN_LIST[@]}")
    if [[ "${d}" == "illness" && -z "${PRED_LENS:-}" ]]; then
      read -r -a CUR_PRED_LEN_LIST <<< "${ILLNESS_PRED_LENS_STR}"
    fi
    for p in "${CUR_PRED_LEN_LIST[@]}"; do
      key="${d}|${p}"
      echo "${d},${p},${MSE_BY_KEY["${key}"]:-}"
    done
  done
} > "${MSE_BY_RUN_CSV}"

{
  printf "dataset"
  for p in "${PRED_LEN_LIST[@]}"; do
    printf ",mse_pl%s" "${p}"
  done
  if [[ -z "${PRED_LENS:-}" ]]; then
    read -r -a __illness_pl_tmp <<< "${ILLNESS_PRED_LENS_STR}"
    for p in "${__illness_pl_tmp[@]}"; do
      found=0
      for q in "${PRED_LEN_LIST[@]}"; do
        if [[ "${q}" == "${p}" ]]; then
          found=1
          break
        fi
      done
      if [[ ${found} -eq 0 ]]; then
        printf ",mse_pl%s" "${p}"
      fi
    done
  fi
  printf ",mse_avg,valid_horizons\n"

  for d in "${DATASET_LIST[@]}"; do
    CUR_PRED_LEN_LIST=("${PRED_LEN_LIST[@]}")
    if [[ "${d}" == "illness" && -z "${PRED_LENS:-}" ]]; then
      read -r -a CUR_PRED_LEN_LIST <<< "${ILLNESS_PRED_LENS_STR}"
    fi

    # union columns for stable CSV header
    COL_PRED_LEN_LIST=("${PRED_LEN_LIST[@]}")
    if [[ -z "${PRED_LENS:-}" ]]; then
      read -r -a __illness_pl_tmp <<< "${ILLNESS_PRED_LENS_STR}"
      for p in "${__illness_pl_tmp[@]}"; do
        found=0
        for q in "${COL_PRED_LEN_LIST[@]}"; do
          if [[ "${q}" == "${p}" ]]; then
            found=1
            break
          fi
        done
        if [[ ${found} -eq 0 ]]; then
          COL_PRED_LEN_LIST+=("${p}")
        fi
      done
    fi

    sum="0"
    valid_count=0
    row="${d}"
    for p in "${COL_PRED_LEN_LIST[@]}"; do
      key="${d}|${p}"
      mse="${MSE_BY_KEY["${key}"]:-}"
      row="${row},${mse}"
      include_in_avg=0
      for pp in "${CUR_PRED_LEN_LIST[@]}"; do
        if [[ "${pp}" == "${p}" ]]; then
          include_in_avg=1
          break
        fi
      done
      if [[ -n "${mse}" && ${include_in_avg} -eq 1 ]]; then
        sum="$(awk -v a="${sum}" -v b="${mse}" 'BEGIN{printf "%.12f", a+b}')"
        valid_count=$((valid_count + 1))
      fi
    done

    avg=""
    if [[ ${valid_count} -eq ${#CUR_PRED_LEN_LIST[@]} ]]; then
      avg="$(awk -v s="${sum}" -v n="${valid_count}" 'BEGIN{printf "%.6f", s/n}')"
    fi
    printf "%s,%s,%d/%d\n" "${row}" "${avg}" "${valid_count}" "${#CUR_PRED_LEN_LIST[@]}"
  done
} > "${MSE_AVG_CSV}"

echo
echo "========================== SUMMARY =========================="
echo "Log directory: ${LOG_DIR}"
echo "Success (${#SUCCESS_RUNS[@]}): ${SUCCESS_RUNS[*]:-none}"
echo "Failed  (${#FAILED_RUNS[@]}): ${FAILED_RUNS[*]:-none}"
echo "Skipped (${#SKIPPED_RUNS[@]}): ${SKIPPED_RUNS[*]:-none}"
echo "MSE by run: ${MSE_BY_RUN_CSV}"
echo "MSE avg over horizons: ${MSE_AVG_CSV}"
echo "============================================================"
