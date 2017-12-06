#!/bin/bash

MATRIXNAME=30p30n
MATRIXSUFFIX=.mtx
MATRIXPATH=../testing/matrices/
MATRIX=${MATRIXPATH}${MATRIXNAME}${MATRIXSUFFIX}
echo ${MATRIX}
#MATRIX=../testing/matrices/30p30n.mtx

DIR=PariLU_P100
threads=(20 40 1024)
sweeps=(1 2 3 4 5 10 20 40 60)

mkdir -p ${DIR}

for t in ${threads[@]}; do
  echo GPU threads = ${t}
  for s in ${sweeps[@]}; do
    ./generate_iLU_gpu --matrix ${MATRIX} --sweeps ${s} --threads ${t} --outDir ${DIR} > ${DIR}/log_P100_${MATRIXNAME}_${s}sweeps_${t}threads.m 2>&1
  done
done
