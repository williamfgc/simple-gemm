#!/bin/bash

M=10000
OUTDIR=""
EXECUTABLE=""

if [ $# -lt 2 ]; then
    echo "Usage: run_threads.sh [matrix-size] out_dir executable"
    exit 1
elif [ $# -eq 3 ]; then
    M=$1
    OUTDIR="$2"
    EXECUTABLE="$3"
else
    OUTDIR="$1"
    EXECUTABLE="$2"
fi

mkdir -p ${OUTDIR}
EXEC_DIR="$(dirname "$EXECUTABLE")"
EXEC_FILE=$(basename -- "${EXECUTABLE}")
EXEC_EXT="${EXEC_FILE##*.}"
CMD_PREFIX=""

echo "ext: ${EXEC_EXT}"

case "${EXEC_EXT}" in
    "jl")
        CMD_PREFIX="julia --project=${EXEC_DIR}"
        ;;
    "py")
        CMD_PREFIX="python3"
        ;;
esac

threads=(1 2 4 8 12 16 20 22 24)

for t in "${threads[@]}"
do
    echo
    echo "Running with ${t} threads for M=${M}"
    CMD_OPT=""
    if [ "${EXEC_EXT}" = "jl" ]; then
        CMD_OPT="-t ${t}"
    else
        export OMP_NUM_THREADS=${t}
    fi
    CMD="${CMD_PREFIX} ${CMD_OPT} ${EXECUTABLE} $M $M $M"
    echo ">${CMD}"
    time ${CMD} > ${OUTDIR}/M10K_${t}t.log
done
