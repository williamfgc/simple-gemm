#!/bin/bash

if [ $# != 2 ]; then
    echo "Usage: run_threads.sh <out_dir> <executable>"
    exit 1
fi

OUTDIR="$1"
mkdir -p ${OUTDIR}
EXECUTABLE="$2"
EXEC_DIR="$(dirname "$EXECUTABLE")"
EXEC_FILE=$(basename -- "${EXECUTABLE}")
EXEC_EXT="${EXEC_FILE##*.}"
CMD_PREFIX=""
CMD_NEEDS_THREADS=0

echo "ext: ${EXEC_EXT}"

case "${EXEC_EXT}" in
    "jl")
        CMD_PREFIX="julia --project=${EXEC_DIR}"
        CMD_NEEDS_THREADS=1
        ;;
    "py")
        # CMD_PREFIX="python3 --project=${EXEC_DIR}"
        CMD_PREFIX="python3"
        ;;
esac

M=100
threads=(1 2 4 8 12 16 20 22 24)

for t in "${threads[@]}"
do
    echo
    echo "Running ${EXECUTABLE} with ${t} threads for M=${M}"
    CMD_OPT=""
    if [ ${CMD_NEEDS_THREADS} -eq 1 ]; then
        CMD_OPT="-t ${t}"
    else
        export OMP_NUM_THREADS=$t
    fi
    CMD="${CMD_PREFIX} ${CMD_OPT} ${EXECUTABLE} $M $M $M"
    echo "Running ${CMD}"
    time ${CMD} > ${OUTDIR}/M10K_${t}t.log
done
