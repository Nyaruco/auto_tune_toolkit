#!/bin/bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════
#  auto_tune 通用启动脚本
#  1) 填下面配置区
#  2) 选模式：MODE=full（默认）/ donly / ponly
#  3) bash auto_tune.sh
# ══════════════════════════════════════════════════════════════════════

base="$(cd "$(dirname "$0")" && pwd)"
cd "${base}"

# ── 配置区 ────────────────────────────────────────────────────────────
host_ip=""                 # 服务端 IP；donly 指 D 实例，ponly 指单实例
port=""
model_path=""              # bench_serving 的 --model（模型目录）
model_name=""              # 日志目录 / CSV 用的模型短名
bench_python="python3"     # 装了 sglang 的解释器
bench_dataset=""           # 本机 ShareGPT json；离线必填，传空则不传
scenes_file=""             # 空 = scenes.py；也可写 test_scenes_cmp_1010.py
latency_stat="p90"         # p90 / p95 / p99 / mean ...

MODE="${MODE:-full}"               # full / donly / ponly
RATE_SEARCH="${RATE_SEARCH:-0}"    # 1 = donly/ponly 也搜 request rate
NO_RATE_SEARCH="${NO_RATE_SEARCH:-0}"  # 1 = full 打满，只搜 batch

# ── 检查 ──────────────────────────────────────────────────────────────
if [ -z "${host_ip}" ] || [ -z "${port}" ] || [ -z "${model_name}" ]; then
    echo "先填 host_ip / port / model_name" >&2
    exit 1
fi
if { [ "${MODE}" = "donly" ] || [ "${MODE}" = "ponly" ]; } && [ -z "${model_path}" ]; then
    echo "${MODE} 走 bench_serving：请填 model_path / bench_python / bench_dataset" >&2
    exit 1
fi

scenes_args=()
if [ -n "${scenes_file}" ]; then
    scenes_args=(--scenes-file "${scenes_file}")
fi

python3 auto_tune.py --list "${scenes_args[@]}"

run_dir="${base}/runs/${model_name}"   # state / report / CSV 都落这
log_dir="${base}/logs"                 # 日志落 logs/<模型>/<mode>/
mkdir -p "${run_dir}" "${log_dir}"

common=(--host "${host_ip}" --port "${port}"
        --bench-python "${bench_python}" --bench-model "${model_path}"
        --model-name "${model_name}" --bench-dataset "${bench_dataset}"
        --state-dir "${run_dir}" --log-dir "${log_dir}"
        --latency-stat "${latency_stat}")
if [ ${#scenes_args[@]} -gt 0 ]; then
    common+=("${scenes_args[@]}")
fi

extra=()
if [ "${RATE_SEARCH}" = "1" ]; then
    extra+=(--rate-search)
fi
if [ "${NO_RATE_SEARCH}" = "1" ]; then
    extra+=(--no-rate-search)
fi

case "${MODE}" in
    full|donly|ponly) ;;
    *) echo "MODE 只支持 full / donly / ponly" >&2; exit 2 ;;
esac

python3 auto_tune.py --mode "${MODE}" "${extra[@]}" "${common[@]}" \
    2>&1 | tee -a "${log_dir}/auto_tune_${MODE}.log"
