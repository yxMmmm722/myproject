#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_FILE="${RUN_FILE:-run.py}"

SEQ_LEN="${SEQ_LEN:-720}"
LABEL_LEN="${LABEL_LEN:-48}"
PRED_LENS_STR="${PRED_LENS:-96 192 336 720}"
ILLNESS_PRED_LENS_STR="${ILLNESS_PRED_LENS:-24 36 48 60}"
N_PERIOD="${N_PERIOD:-3}"
TOPM="${TOPM:-20}"
RETRIEVAL_TEMPERATURE="${RETRIEVAL_TEMPERATURE:-0.1}"
META_ONLY_RETRIEVAL="${META_ONLY_RETRIEVAL:-0}"
COMPARE_RETRIEVAL_FUTURE_QUALITY="${COMPARE_RETRIEVAL_FUTURE_QUALITY:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
LRADJ="${LRADJ:-type1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MODEL_PREFIX="${MODEL_PREFIX:-RAFTORIGIN}"
USE_TABLE6_PRESETS="${USE_TABLE6_PRESETS:-1}"
USE_AMP="${USE_AMP:-0}"

DATASETS_STR="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 electricity exchange_rate illness traffic weather}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-./logs/run_all_${TIMESTAMP}}"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${RUN_FILE}" ]]; then
  echo "[FATAL] Cannot find ${RUN_FILE}. Please run this script in RAFT-origin root."
  exit 1
fi

declare -A ROOT_PATHS=(
  [etth1]="./data/ETT"
  [etth2]="./data/ETT"
  [ettm1]="./data/ETT"
  [ettm2]="./data/ETT"
  [electricity]="./data/electricity"
  [exchange_rate]="./data/exchange_rate"
  [exchange]="./data/exchange_rate"
  [illness]="./data/illness"
  [traffic]="./data/traffic"
  [weather]="./data/weather"
)

declare -A DATA_FILES=(
  [etth1]="ETTh1.csv"
  [etth2]="ETTh2.csv"
  [ettm1]="ETTm1.csv"
  [ettm2]="ETTm2.csv"
  [electricity]="electricity.csv"
  [exchange_rate]="exchange_rate.csv"
  [exchange]="exchange_rate.csv"
  [illness]="national_illness.csv"
  [traffic]="traffic.csv"
  [weather]="weather.csv"
)

declare -A FREQS=(
  [etth1]="h"
  [etth2]="h"
  [ettm1]="t"
  [ettm2]="t"
  [electricity]="h"
  [exchange_rate]="d"
  [exchange]="d"
  [illness]="w"
  [traffic]="h"
  [weather]="t"
)

declare -A DATA_ARGS=(
  [etth1]="ETTh1"
  [etth2]="ETTh2"
  [ettm1]="ETTm1"
  [ettm2]="ETTm2"
  [electricity]="custom"
  [exchange_rate]="custom"
  [exchange]="custom"
  [illness]="custom"
  [traffic]="custom"
  [weather]="custom"
)

declare -A CANONICAL_NAMES=(
  [etth1]="ETTh1"
  [etth2]="ETTh2"
  [ettm1]="ETTm1"
  [ettm2]="ETTm2"
  [electricity]="electricity"
  [exchange_rate]="exchange_rate"
  [exchange]="exchange_rate"
  [illness]="illness"
  [traffic]="traffic"
  [weather]="weather"
)

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

set_table6_preset ETTh1 96 720 1e-3 20
set_table6_preset ETTh1 192 720 1e-2 20
set_table6_preset ETTh1 336 720 1e-2 20
set_table6_preset ETTh1 720 720 1e-4 20
set_table6_preset ETTh2 96 720 1e-2 10
set_table6_preset ETTh2 192 720 1e-3 10
set_table6_preset ETTh2 336 720 1e-3 20
set_table6_preset ETTh2 720 720 1e-4 20
set_table6_preset ETTm1 96 720 1e-2 1
set_table6_preset ETTm1 192 720 1e-3 20
set_table6_preset ETTm1 336 720 1e-3 20
set_table6_preset ETTm1 720 720 1e-2 20
set_table6_preset ETTm2 96 720 1e-3 5
set_table6_preset ETTm2 192 720 1e-3 20
set_table6_preset ETTm2 336 720 1e-4 20
set_table6_preset ETTm2 720 720 1e-4 20
set_table6_preset electricity 96 720 1e-2 1
set_table6_preset electricity 192 720 1e-3 1
set_table6_preset electricity 336 720 1e-3 1
set_table6_preset electricity 720 720 1e-3 1
set_table6_preset exchange_rate 96 720 1e-4 1
set_table6_preset exchange_rate 192 720 1e-3 1
set_table6_preset exchange_rate 336 720 1e-3 10
set_table6_preset exchange_rate 720 720 1e-4 20
set_table6_preset illness 24 96 1e-2 1
set_table6_preset illness 36 96 1e-2 1
set_table6_preset illness 48 96 1e-2 20
set_table6_preset illness 60 96 1e-2 20
set_table6_preset traffic 96 720 1e-2 1
set_table6_preset traffic 192 720 1e-3 1
set_table6_preset traffic 336 720 1e-3 1
set_table6_preset traffic 720 720 1e-3 1
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

normalize_dataset() {
  echo "$1" | tr '[:upper:]' '[:lower:]'
}

is_custom_dataset() {
  local data_arg="$1"
  [[ "${data_arg}" == "custom" ]]
}

is_window_valid_for_custom_split() {
  local csv_path="$1"
  local seq_len="$2"
  local pred_len="$3"

  local total_rows
  total_rows=$(( $(wc -l < "${csv_path}") - 1 ))
  if [[ ${total_rows} -le 0 ]]; then
    return 1
  fi

  local num_train num_test num_val
  num_train=$(( total_rows * 7 / 10 ))
  num_test=$(( total_rows * 2 / 10 ))
  num_val=$(( total_rows - num_train - num_test ))

  local train_len val_len test_len
  train_len=$(( num_train - seq_len - pred_len + 1 ))
  val_len=$(( num_val - pred_len + 1 ))
  test_len=$(( num_test - pred_len + 1 ))

  if [[ ${train_len} -le 0 || ${val_len} -le 0 || ${test_len} -le 0 ]]; then
    return 1
  fi
  return 0
}

read -r -a DATASET_LIST <<< "${DATASETS_STR}"
read -r -a PRED_LEN_LIST <<< "${PRED_LENS_STR}"

SUCCESS_RUNS=()
FAILED_RUNS=()
SKIPPED_RUNS=()
declare -A MSE_BY_KEY

run_one() {
  local dataset_raw="$1"
  local pred_len="$2"
  local dataset_key
  dataset_key="$(normalize_dataset "${dataset_raw}")"
  local dataset="${CANONICAL_NAMES[${dataset_key}]:-}"

  if [[ -z "${dataset}" ]]; then
    echo "[SKIP] ${dataset_raw}: no mapping in script."
    SKIPPED_RUNS+=("${dataset_raw}:pl${pred_len}")
    return 0
  fi

  local root_path="${ROOT_PATHS[${dataset_key}]}"
  local data_file="${DATA_FILES[${dataset_key}]}"
  local freq="${FREQS[${dataset_key}]}"
  local data_arg="${DATA_ARGS[${dataset_key}]}"
  local seq_len="${SEQ_LEN}"
  local learning_rate="${LEARNING_RATE}"
  local topm="${TOPM}"
  local preset_key="${dataset}|${pred_len}"
  local preset_source="default"

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

  local csv_path="${root_path}/${data_file}"
  if [[ ! -f "${csv_path}" ]]; then
    echo "[SKIP] ${dataset}: missing file ${csv_path}"
    SKIPPED_RUNS+=("${dataset}:pl${pred_len}")
    return 0
  fi

  if is_custom_dataset "${data_arg}"; then
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
  echo "[RUN ] dataset=${dataset} data_arg=${data_arg} pred_len=${pred_len} channels=${channels}"
  echo "[CFG ] seq_len=${seq_len} lr=${learning_rate} lradj=${LRADJ} topm=${topm} temp=${RETRIEVAL_TEMPERATURE} meta_only=${META_ONLY_RETRIEVAL} cmp_retr_quality=${COMPARE_RETRIEVAL_FUTURE_QUALITY} preset=${preset_source}"
  echo "[LOG ] ${log_file}"
  echo "============================================================"

  local cmd=(
    "${PYTHON_BIN}" "${RUN_FILE}"
    --data "${data_arg}"
    --root_path "${root_path}"
    --data_path "${data_file}"
    --freq "${freq}"
    --model_id "${model_id}"
    --is_training 1
    --seq_len "${seq_len}" --label_len "${LABEL_LEN}" --pred_len "${pred_len}"
    --enc_in "${channels}" --dec_in "${channels}" --c_out "${channels}"
    --n_period "${N_PERIOD}" --topm "${topm}" --retrieval_temperature "${RETRIEVAL_TEMPERATURE}"
    --batch_size "${BATCH_SIZE}" --train_epochs "${TRAIN_EPOCHS}"
    --learning_rate "${learning_rate}" --lradj "${LRADJ}" --num_workers "${NUM_WORKERS}"
  )

  if [[ "${USE_AMP}" == "1" ]]; then
    cmd+=(--use_amp)
  fi
  if [[ "${META_ONLY_RETRIEVAL}" == "1" ]]; then
    cmd+=(--meta_only_retrieval)
  fi
  if [[ "${COMPARE_RETRIEVAL_FUTURE_QUALITY}" == "1" ]]; then
    cmd+=(--compare_retrieval_future_quality)
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
  CUR_PRED_LEN_LIST=("${PRED_LEN_LIST[@]}")
  if [[ "$(normalize_dataset "${d}")" == "illness" && -z "${PRED_LENS:-}" ]]; then
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
    dataset_key="$(normalize_dataset "${d}")"
    dataset="${CANONICAL_NAMES[${dataset_key}]:-${d}}"
    CUR_PRED_LEN_LIST=("${PRED_LEN_LIST[@]}")
    if [[ "${dataset_key}" == "illness" && -z "${PRED_LENS:-}" ]]; then
      read -r -a CUR_PRED_LEN_LIST <<< "${ILLNESS_PRED_LENS_STR}"
    fi
    for p in "${CUR_PRED_LEN_LIST[@]}"; do
      key="${dataset}|${p}"
      echo "${dataset},${p},${MSE_BY_KEY["${key}"]:-}"
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
    dataset_key="$(normalize_dataset "${d}")"
    dataset="${CANONICAL_NAMES[${dataset_key}]:-${d}}"
    CUR_PRED_LEN_LIST=("${PRED_LEN_LIST[@]}")
    if [[ "${dataset_key}" == "illness" && -z "${PRED_LENS:-}" ]]; then
      read -r -a CUR_PRED_LEN_LIST <<< "${ILLNESS_PRED_LENS_STR}"
    fi

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

    sum=0
    cnt=0
    printf "%s" "${dataset}"
    for p in "${COL_PRED_LEN_LIST[@]}"; do
      key="${dataset}|${p}"
      mse="${MSE_BY_KEY["${key}"]:-}"
      printf ",%s" "${mse}"
      if [[ -n "${mse}" ]]; then
        sum=$(awk -v a="${sum}" -v b="${mse}" 'BEGIN{printf "%.12f", a+b}')
        cnt=$((cnt+1))
      fi
    done
    if [[ ${cnt} -gt 0 ]]; then
      avg=$(awk -v s="${sum}" -v c="${cnt}" 'BEGIN{printf "%.12f", s/c}')
      printf ",%s,%d\n" "${avg}" "${cnt}"
    else
      printf ",,%d\n" "${cnt}"
    fi
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
