#!/bin/bash

MATRIXNAME=${1}
MATRIXSUFFIX=.mtx
MATRIXPATH=${2}
MATRIX=${MATRIXPATH}${MATRIXNAME}${MATRIXSUFFIX}
echo ${MATRIX}

DIR=${3}
threads=(2 4)
sweeps=(1 2 3 4 5)

mkdir -p ${DIR}

for t in ${threads[@]}; do
  export OMP_NUM_THREADS=${t}
  echo OMP_NUM_THREADS = $OMP_NUM_THREADS
  for s in ${sweeps[@]}; do
    ./generate_iLU --matrix ${MATRIX} --sweeps ${s} --outDir ${DIR} > ${DIR}/log_laptop_${MATRIXNAME}_${s}sweeps_${t}threads.m 2>&1 
  done
done
