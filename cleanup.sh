#!/bin/bash

qselect -u "$(whoami)" -s R | xargs -r qdel

declare -a files_to_delete=("paceEngineJobSubmit*.sh" "*.err" "*.out" "myPickleFile[0-9]*.dat" "gridEngineJobSubmit_master[0-9]*.sh" "slurmEngineJobSubmit_master[0-9]*.sh" "out[0-9]*.txt" "err[0-9]*.txt" "hypervolume[0-9]*.txt" "*.o[0-9]*" "*.e[0-9]*")
for file in "${files_to_delete[@]}"
do
    find . -name "$file" -type f -delete
done
