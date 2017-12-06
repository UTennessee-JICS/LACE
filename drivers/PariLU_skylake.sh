#!/bin/bash

MATRIXNAME=30p30n
MATRIXSUFFIX=.mtx
MATRIXPATH=../testing/matrices/
MATRIX=${MATRIXPATH}${MATRIXNAME}${MATRIXSUFFIX}
echo ${MATRIX}
#MATRIX=../testing/matrices/30p30n.mtx

DIR=PariLU_Skylake
threads=(20 40)
sweeps=(1 2 3 4 5 10 20 40 60)

mkdir -p ${DIR}

for t in ${threads[@]}; do
  export OMP_NUM_THREADS=${t}
  echo OMP_NUM_THREADS = $OMP_NUM_THREADS
  for s in ${sweeps[@]}; do
    ./generate_iLU --matrix ${MATRIX} --sweeps ${s} --outDir ${DIR} > ${DIR}/log_Skylake_${MATRIXNAME}_${s}sweeps_${t}threads.m 2>&1 
  done
done
