#!/bin/bash
set -e

BIN=bin
RESULTS=results
EXEC=$BIN/test

mkdir -p $RESULTS
CSV_FILE=$RESULTS/measurements.csv

echo "collective,procs,my_mpi_time,mpi_time" > $CSV_FILE

PROCS_LIST=(1 2 4 8)

for p in "${PROCS_LIST[@]}"; do
    
    echo "Running with $p processes..."
    output=$(mpirun -np $p $EXEC)
    echo "$output" | grep -E '^[a-z]+,[0-9]+,' >> $CSV_FILE

done

echo
echo "----------------------------------------------"
printf "%-10s | %-5s | %-10s | %-12s\n" "Collective" "Procs" "My_MPI Time" "MPI Time"
echo "----------------------------------------------"

tail -n +2 "$CSV_FILE" | while IFS=, read -r coll procs my mpi; do
    coll_cap="$(tr '[:lower:]' '[:upper:]' <<< ${coll:0:1})${coll:1}"
    printf "%-10s | %-5s | %-10s | %-12s\n" "$coll_cap" "$procs" "$my" "$mpi"
done

echo
echo "Measurements saved in $CSV_FILE"