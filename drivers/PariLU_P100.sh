#!/bin/bash

MATRIX=../testing/matrices/30p30n.mtx

DIR=PariLU_P100
threads=(20 40 1024)
sweeps=(1 2 3 4 5 10 20 40 60)

mkdir -p ${DIR}

for t in ${threads[@]}; do
  export OMP_NUM_THREADS=${t}
  echo OMP_NUM_THREADS = $OMP_NUM_THREADS
  for s in ${sweeps[@]}; do
    ./generate_iLU --matrix ${MATRIX} --sweeps ${s} --outDir ${DIR}  
  done
done
