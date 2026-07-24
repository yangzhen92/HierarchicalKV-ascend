#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SOURCE_GIT_WORKTREE_STATUS="unknown"
if git -C "$PROJECT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=all)" ]]; then
        SOURCE_GIT_WORKTREE_STATUS="dirty"
    else
        SOURCE_GIT_WORKTREE_STATUS="clean"
    fi
fi

CANN_ROOT="${ASCEND_INSTALL_PATH:-${ASCEND_HOME_PATH:-}}"
BUILD_DIR="${PROJECT_ROOT}/build-cannsim"
OUTPUT_DIR=""
SOC_VERSION="Ascend950"
GTEST_FILTER="test_find_or_insert_ptr.empty_table_for_sim"
CORE_ID="0"
BUILD_JOBS="${HKV_BUILD_JOBS:-16}"
RECORD_TIMEOUT="${HKV_CANNSIM_RECORD_TIMEOUT:-30m}"
REPORT_TIMEOUT="${HKV_CANNSIM_REPORT_TIMEOUT:-10m}"
BUILD_ONLY=0
SKIP_BUILD=0
ALLOW_UNSUPPORTED_ARM=0
REPORT_ON_RECORD_FAILURE=0

usage() {
    cat <<EOF
Usage: ${0##*/} [OPTIONS]

Build the board-compatible test binary, run one GTest with CANN Simulator,
and generate a Chrome Tracing JSON report from the recorded instructions.

Options:
  -h, --help                  Show this help and exit.
  -i, --install-path PATH     CANN installation root.
  -b, --build-dir PATH        Build directory (default: build-cannsim).
  -o, --output-dir PATH       New or empty simulation output directory.
  -s, --soc-version SOC       cannsim SoC name (default: Ascend950).
  -f, --gtest-filter FILTER   GTest filter.
  -n, --core-id ID            Report core selection (default: 0).
  -j, --jobs N                Parallel build jobs (default: 16).
      --record-timeout TIME    record timeout (default: 30m).
      --report-timeout TIME    report timeout (default: 10m).
  -c, --build-only            Build the cannsim smoke binary only.
      --skip-build             Reuse an existing, unverified smoke binary.
      --allow-unsupported-arm Allow aarch64 although public CANN Simulator
                              documentation currently marks ARM unsupported.
      --report-on-record-failure
                              Try to generate diagnostic report after GTest
                              passed but record failed; final exit stays nonzero.
EOF
}

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

while (($# > 0)); do
    case "$1" in
    -h | --help)
        usage
        exit 0
        ;;
    -i | --install-path)
        (($# >= 2)) || fail "$1 requires a value"
        CANN_ROOT="$2"
        shift 2
        ;;
    -b | --build-dir)
        (($# >= 2)) || fail "$1 requires a value"
        BUILD_DIR="$2"
        shift 2
        ;;
    -o | --output-dir)
        (($# >= 2)) || fail "$1 requires a value"
        OUTPUT_DIR="$2"
        shift 2
        ;;
    -s | --soc-version)
        (($# >= 2)) || fail "$1 requires a value"
        SOC_VERSION="$2"
        shift 2
        ;;
    -f | --gtest-filter)
        (($# >= 2)) || fail "$1 requires a value"
        GTEST_FILTER="$2"
        shift 2
        ;;
    -n | --core-id)
        (($# >= 2)) || fail "$1 requires a value"
        CORE_ID="$2"
        shift 2
        ;;
    -j | --jobs)
        (($# >= 2)) || fail "$1 requires a value"
        BUILD_JOBS="$2"
        shift 2
        ;;
    --record-timeout)
        (($# >= 2)) || fail "$1 requires a value"
        RECORD_TIMEOUT="$2"
        shift 2
        ;;
    --report-timeout)
        (($# >= 2)) || fail "$1 requires a value"
        REPORT_TIMEOUT="$2"
        shift 2
        ;;
    -c | --build-only)
        BUILD_ONLY=1
        shift
        ;;
    --skip-build)
        SKIP_BUILD=1
        shift
        ;;
    --allow-unsupported-arm)
        ALLOW_UNSUPPORTED_ARM=1
        shift
        ;;
    --report-on-record-failure)
        REPORT_ON_RECORD_FAILURE=1
        shift
        ;;
    --)
        shift
        break
        ;;
    *)
        fail "unknown option: $1"
        ;;
    esac
done

[[ "$SOC_VERSION" =~ ^[A-Za-z0-9_-]+$ ]] ||
    fail "invalid SoC name: ${SOC_VERSION}"
[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] ||
    fail "jobs must be a positive integer"
[[ "$CORE_ID" == "all" ||
    "$CORE_ID" =~ ^(0|[1-9][0-9]*)(-(0|[1-9][0-9]*))?(,(0|[1-9][0-9]*)(-(0|[1-9][0-9]*))?)*$ ]] ||
    fail "core-id must be 'all', a number, a range, or a comma-separated list"
[[ "$RECORD_TIMEOUT" =~ ^[1-9][0-9]*[smhd]?$ ]] ||
    fail "record-timeout must be a positive integer with optional s/m/h/d suffix"
[[ "$REPORT_TIMEOUT" =~ ^[1-9][0-9]*[smhd]?$ ]] ||
    fail "report-timeout must be a positive integer with optional s/m/h/d suffix"
[[ -n "$GTEST_FILTER" ]] || fail "gtest-filter must not be empty"
if [[ "$CORE_ID" != "all" ]]; then
    IFS=, read -r -a CORE_RANGES <<<"$CORE_ID"
    for core_range in "${CORE_RANGES[@]}"; do
        if [[ "$core_range" == *-* ]]; then
            core_first="${core_range%-*}"
            core_last="${core_range#*-}"
            ((core_first <= core_last)) ||
                fail "invalid descending core range: ${core_range}"
        fi
    done
fi
if ((BUILD_ONLY == 1 && SKIP_BUILD == 1)); then
    fail "--build-only and --skip-build cannot be used together"
fi

if [[ -z "$CANN_ROOT" ]]; then
    for candidate in \
        "${HOME:-}/Ascend/cann" \
        "${HOME:-}/Ascend/ascend-toolkit/latest" \
        /usr/local/Ascend/cann \
        /usr/local/Ascend/ascend-toolkit/latest \
        /usr/local/Ascend/cann-*; do
        if [[ -d "$candidate" ]]; then
            CANN_ROOT="$candidate"
            break
        fi
    done
fi
[[ -n "$CANN_ROOT" && -d "$CANN_ROOT" ]] ||
    fail "CANN installation was not found; pass --install-path"
CANN_ROOT="$(cd -- "$CANN_ROOT" && pwd -P)"

SET_ENV_SCRIPT=""
for candidate in \
    "${CANN_ROOT}/set_env.sh" \
    "${CANN_ROOT}/bin/setenv.bash" \
    "${CANN_ROOT}/bin/set_env.sh"; do
    if [[ -f "$candidate" ]]; then
        SET_ENV_SCRIPT="$candidate"
        break
    fi
done
[[ -n "$SET_ENV_SCRIPT" ]] ||
    fail "CANN environment script was not found under ${CANN_ROOT}"

# Some CANN environment scripts read optional variables without defaults.
set +u
source "$SET_ENV_SCRIPT"
set -euo pipefail

CANNSIM="${CANN_ROOT}/bin/cannsim"
if [[ ! -x "$CANNSIM" ]]; then
    CANNSIM="$(command -v cannsim || true)"
fi
[[ -n "$CANNSIM" && -x "$CANNSIM" ]] ||
    fail "cannsim was not found after loading ${SET_ENV_SCRIPT}"

case "$(uname -m)" in
x86_64 | amd64)
    PLATFORM_DIR="x86_64-linux"
    HOST_ARCH="x86_64"
    PLATFORM_SUPPORT="official"
    ;;
aarch64 | arm64)
    ((ALLOW_UNSUPPORTED_ARM == 1)) ||
        fail "public CANN Simulator documentation marks ARM unsupported; use --allow-unsupported-arm only for a verified CANN image"
    PLATFORM_DIR="aarch64-linux"
    HOST_ARCH="aarch64"
    PLATFORM_SUPPORT="unsupported_override"
    ;;
*)
    fail "unsupported host architecture: $(uname -m)"
    ;;
esac

DEVLIB=""
for candidate in \
    "${CANN_ROOT}/${PLATFORM_DIR}/devlib" \
    "${CANN_ROOT}/${PLATFORM_DIR}/devlib/linux/${HOST_ARCH}"; do
    if [[ -d "$candidate" ]]; then
        DEVLIB="$candidate"
        break
    fi
done

RUNTIME_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

path_without_devlib() {
    local input="$1"
    local output=""
    local item
    local -a items
    IFS=: read -r -a items <<<"$input"
    for item in "${items[@]}"; do
        [[ -n "$item" ]] || continue
        [[ "$item" == *"/devlib"* ]] && continue
        output="${output:+${output}:}${item}"
    done
    printf '%s' "$output"
}

if ((SKIP_BUILD == 0)); then
    [[ -n "$DEVLIB" ]] ||
        fail "architecture-matching CANN devlib was not found under ${CANN_ROOT}"
    [[ -f "${PROJECT_ROOT}/3rdparty/gtest/CMakeLists.txt" ]] ||
        fail "GTest submodule is missing; run: git submodule update --init --recursive"

    BUILD_LD_LIBRARY_PATH="${DEVLIB}${RUNTIME_LD_LIBRARY_PATH:+:${RUNTIME_LD_LIBRARY_PATH}}"
    BUILD_LIBRARY_PATH="${DEVLIB}${LIBRARY_PATH:+:${LIBRARY_PATH}}"

    echo "[INFO] Configuring board-compatible binary for cannsim"
    env \
        LD_LIBRARY_PATH="$BUILD_LD_LIBRARY_PATH" \
        LIBRARY_PATH="$BUILD_LIBRARY_PATH" \
        cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
        -DRUN_MODE=npu \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DENABLE_TEST=1

    echo "[INFO] Building test_cannsim_smoke with ${BUILD_JOBS} jobs"
    env \
        LD_LIBRARY_PATH="$BUILD_LD_LIBRARY_PATH" \
        LIBRARY_PATH="$BUILD_LIBRARY_PATH" \
        cmake --build "$BUILD_DIR" --target test_cannsim_smoke -j "$BUILD_JOBS"
fi

TEST_BINARY="${BUILD_DIR}/tests/test_cannsim_smoke"
[[ -x "$TEST_BINARY" ]] || fail "test binary was not found: ${TEST_BINARY}"
if ((SKIP_BUILD == 1)); then
    CMAKE_CACHE="${BUILD_DIR}/CMakeCache.txt"
    [[ -f "$CMAKE_CACHE" ]] ||
        fail "CMake cache was not found for the reused build: ${CMAKE_CACHE}"
    grep -Fxq "CMAKE_HOME_DIRECTORY:INTERNAL=${PROJECT_ROOT}" "$CMAKE_CACHE" ||
        fail "reused build was configured from a different source directory"
    grep -Eq '^RUN_MODE:[^=]+=npu$' "$CMAKE_CACHE" ||
        fail "reused build was not configured with RUN_MODE=npu"
fi
if ((BUILD_ONLY == 1)); then
    echo "[INFO] Build completed: ${TEST_BINARY}"
    exit 0
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${PROJECT_ROOT}/cannsim_output/$(date +%Y%m%d_%H%M%S)_$$"
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    [[ -d "$OUTPUT_DIR" ]] || fail "output path is not a directory: ${OUTPUT_DIR}"
    if [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        fail "output directory must be empty: ${OUTPUT_DIR}"
    fi
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd -P)"

# devlib contains link-time stubs. It must not shadow the simulator/runtime
# libraries while the recorded program is running.
RUNTIME_LD_LIBRARY_PATH="$(path_without_devlib "$RUNTIME_LD_LIBRARY_PATH")"
export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${CANN_ROOT}/lib64${RUNTIME_LD_LIBRARY_PATH:+:${RUNTIME_LD_LIBRARY_PATH}}"

RECORD_CONSOLE="${OUTPUT_DIR}/cannsim-record.console.log"
echo "[INFO] Recording ${GTEST_FILTER}"
echo "[INFO] Output directory: ${OUTPUT_DIR}"
command -v timeout >/dev/null 2>&1 ||
    fail "GNU timeout is required"
set +e
timeout --kill-after=30s "$RECORD_TIMEOUT" "$CANNSIM" record \
    -s "$SOC_VERSION" \
    -n "$CORE_ID" \
    --user-option="--gtest_filter=${GTEST_FILTER}" \
    -o "$OUTPUT_DIR" \
    "$TEST_BINARY" 2>&1 | tee "$RECORD_CONSOLE"
RECORD_RC=${PIPESTATUS[0]}
set -e

mapfile -d '' ARCHIVES < <(
    find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d \
        -name 'cannsim_*' -print0
)
((${#ARCHIVES[@]} == 1)) ||
    fail "expected exactly one new simulation archive, found ${#ARCHIVES[@]}"
ARCHIVE="${ARCHIVES[0]}"

CANNSIM_LOG="$(find "$ARCHIVE" -type f -name cannsim.log -size +0c -print -quit)"
INSTR_BIN="$(find "$ARCHIVE" -type f -name instr.bin -size +0c -print -quit)"
[[ -n "$CANNSIM_LOG" ]] || fail "non-empty cannsim.log was not generated"
[[ -n "$INSTR_BIN" ]] || fail "non-empty instr.bin was not generated"

if grep -Eq '\[ +FAILED +\]' "$CANNSIM_LOG"; then
    fail "GTest reported a failed test; see ${CANNSIM_LOG}"
fi
if ! grep -Fq "Note: Google Test filter = ${GTEST_FILTER}" "$CANNSIM_LOG"; then
    fail "the requested GTest filter was not observed; see ${CANNSIM_LOG}"
fi
if ! grep -Eq '\[ +RUN +\]' "$CANNSIM_LOG"; then
    fail "GTest did not start a test; see ${CANNSIM_LOG}"
fi
if ! grep -Eq '\[ +PASSED +\] +[1-9][0-9]* tests?' "$CANNSIM_LOG"; then
    fail "GTest success marker was not found; see ${CANNSIM_LOG}"
fi
ACL_SHUTDOWN_STATUS="passed"
MISSING_SHUTDOWN_MARKERS=""
for shutdown_marker in \
    "[ACL] Resetting device" \
    "[ACL] Finalizing runtime" \
    "[ACL] Runtime shutdown complete"; do
    if ! grep -Fq "$shutdown_marker" "$CANNSIM_LOG"; then
        ACL_SHUTDOWN_STATUS="failed"
        MISSING_SHUTDOWN_MARKERS+="${shutdown_marker};"
    fi
done
if [[ "$ACL_SHUTDOWN_STATUS" != "passed" ]]; then
    if ((RECORD_RC == 0)); then
        fail "ACL shutdown markers were not found: ${MISSING_SHUTDOWN_MARKERS}"
    fi
    echo "[WARN] ACL shutdown markers were not found: ${MISSING_SHUTDOWN_MARKERS}"
fi

if ((RECORD_RC != 0)); then
    if ((REPORT_ON_RECORD_FAILURE == 0)); then
        fail "cannsim record exited ${RECORD_RC}; use --report-on-record-failure only to collect diagnostics"
    fi
    echo "[WARN] cannsim record exited ${RECORD_RC} after GTest passed."
    echo "[WARN] Generating a diagnostic report; the final command will still fail."
fi

CORE_LABEL="${CORE_ID//,/__}"
CORE_LABEL="${CORE_LABEL//-/_}"
REPORT_DIR="${ARCHIVE}/report-${CORE_LABEL}"
REPORT_CONSOLE="${OUTPUT_DIR}/cannsim-report.console.log"
[[ ! -e "$REPORT_DIR" ]] ||
    fail "report directory already exists: ${REPORT_DIR}"

echo "[INFO] Generating report for core selection ${CORE_ID}"
set +e
timeout --kill-after=30s "$REPORT_TIMEOUT" "$CANNSIM" report \
    -e "$ARCHIVE" \
    -o "$REPORT_DIR" \
    -n "$CORE_ID" 2>&1 | tee "$REPORT_CONSOLE"
REPORT_RC=${PIPESTATUS[0]}
set -e
((REPORT_RC == 0)) || fail "cannsim report exited ${REPORT_RC}"

mapfile -d '' TRACE_FILES < <(
    find "$REPORT_DIR" -type f -name 'trace_core*.json' -size +0c -print0
)
((${#TRACE_FILES[@]} > 0)) ||
    fail "no non-empty trace_core*.json report was generated"

if [[ "$CORE_ID" != "all" ]]; then
    IFS=, read -r -a CORE_RANGES <<<"$CORE_ID"
    for core_range in "${CORE_RANGES[@]}"; do
        if [[ "$core_range" == *-* ]]; then
            core_first="${core_range%-*}"
            core_last="${core_range#*-}"
        else
            core_first="$core_range"
            core_last="$core_range"
        fi
        for ((core = core_first; core <= core_last; core++)); do
            trace_for_core_found=0
            for trace in "${TRACE_FILES[@]}"; do
                if [[ "${trace##*/}" == "trace_core${core}.json" ]]; then
                    trace_for_core_found=1
                    break
                fi
            done
            ((trace_for_core_found == 1)) ||
                fail "requested core ${core} report was not generated"
        done
    done
fi

command -v python3 >/dev/null 2>&1 ||
    fail "python3 is required to validate the generated JSON"
TRACE_SUMMARY="$(
    python3 - "${TRACE_FILES[@]}" <<'PY'
import json
import pathlib
import sys

event_total = 0
duration_total = 0
for name in sys.argv[1:]:
    path = pathlib.Path(name)
    data = json.loads(path.read_text())
    events = data.get("traceEvents") if isinstance(data, dict) else data
    if not isinstance(events, list) or not events:
        raise SystemExit(f"empty or invalid trace event list: {path}")
    if any(not isinstance(event, dict) for event in events):
        raise SystemExit(f"trace contains a non-object event: {path}")
    duration_events = [
        event
        for event in events
        if event.get("ph") == "X"
        and isinstance(event.get("name"), str)
        and isinstance(event.get("ts"), (int, float))
        and isinstance(event.get("dur"), (int, float))
    ]
    if not duration_events:
        raise SystemExit(f"trace contains no complete duration events: {path}")
    event_total += len(events)
    duration_total += len(duration_events)
print(f"trace_files={len(sys.argv) - 1}")
print(f"trace_events={event_total}")
print(f"trace_duration_events={duration_total}")
PY
)"

SUMMARY_FILE="${OUTPUT_DIR}/validation-summary.txt"
VALIDATION_STATUS="passed"
if ((RECORD_RC != 0)); then
    VALIDATION_STATUS="failed_record_exit_report_generated"
fi
{
    echo "status=${VALIDATION_STATUS}"
    echo "project_commit=$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "source_git_worktree_status=${SOURCE_GIT_WORKTREE_STATUS}"
    if ((SKIP_BUILD == 1)); then
        echo "build_provenance=reused_unverified"
    else
        echo "build_provenance=built_by_this_run"
    fi
    echo "host_arch=${HOST_ARCH}"
    echo "platform_support=${PLATFORM_SUPPORT}"
    echo "cann_root=${CANN_ROOT}"
    echo "cannsim=${CANNSIM}"
    echo "soc_version=${SOC_VERSION}"
    echo "core_id=${CORE_ID}"
    echo "gtest_filter=${GTEST_FILTER}"
    echo "record_timeout=${RECORD_TIMEOUT}"
    echo "report_timeout=${REPORT_TIMEOUT}"
    echo "gtest_status=passed"
    echo "acl_shutdown_status=${ACL_SHUTDOWN_STATUS}"
    echo "record_exit_code=${RECORD_RC}"
    echo "report_exit_code=${REPORT_RC}"
    echo "test_binary=${TEST_BINARY}"
    echo "test_binary_sha256=$(sha256sum "$TEST_BINARY" | awk '{print $1}')"
    echo "archive=${ARCHIVE}"
    echo "cannsim_log=${CANNSIM_LOG}"
    echo "cannsim_log_sha256=$(sha256sum "$CANNSIM_LOG" | awk '{print $1}')"
    echo "instr_bin=${INSTR_BIN}"
    echo "instr_bin_sha256=$(sha256sum "$INSTR_BIN" | awk '{print $1}')"
    printf '%s\n' "$TRACE_SUMMARY"
    for trace in "${TRACE_FILES[@]}"; do
        echo "trace=${trace}"
        echo "trace_sha256=$(sha256sum "$trace" | awk '{print $1}')"
    done
} >"$SUMMARY_FILE"

echo "[INFO] GTest passed and the simulation report is valid."
echo "[INFO] Summary: ${SUMMARY_FILE}"
for trace in "${TRACE_FILES[@]}"; do
    echo "[INFO] Trace: ${trace}"
done

if ((RECORD_RC != 0)); then
    fail "cannsim record exited ${RECORD_RC}; diagnostic report was generated"
fi
