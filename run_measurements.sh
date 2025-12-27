#!/bin/bash
set -e

BIN=bin
RESULTS=results
EXEC=$BIN/test

mkdir -p $RESULTS
CSV_FILE=$RESULTS/measurements.csv

echo "procs,my_mpi_time,mpi_time" > $CSV_FILE

PROCS_LIST=(1 2 4 8)

for p in "${PROCS_LIST[@]}"; do

    echo "Running with $p processes..."
    output=$(mpirun -np $p $EXEC)
    lastline=$(printf "%s\n" "$output" | awk -F, 'NF==2 {print $0}' | tail -n1)
    
    if [ -z "$lastline" ]; then
        echo "ERROR: Could not parse timing output from program. Full output:"
        echo "$output"
        exit 1
    fi

    my_mpi_time=$(echo $lastline | cut -d',' -f1)
    mpi_time=$(echo $lastline | cut -d',' -f2)
    echo "$p,$my_mpi_time,$mpi_time" >> $CSV_FILE

done

echo
printf "%-5s | %-10s | %-12s\n" "Procs" "My_MPI_Time" "MPI_Time"
echo "---------------------------------"
tail -n +2 $CSV_FILE | while IFS=, read -r procs mympit mpit; do
    printf "%-5s | %-10s | %-12s\n" "$procs" "$mympit" "$mpit"
done
echo
echo "Measurements saved in $CSV_FILE"