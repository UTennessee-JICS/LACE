#!/bin/bash

RUNNAME="MKLFGMRES"
REPETITION="01"
MACHINE="beacon"
DATE=$(date +'%Y%m%d')
LOGDIR=log_${RUNNAME}_${MACHINE}_${DATE}_${REPETITION}
threads=(1 16 32)
#threads=(1 68 136 204 272)
#threads=(1 `seq 2 2 272`)
#threads=(1 `seq 2 2 60`)
#precommand=""
precommand="numactl --interleave=all"

MATDIR="testing/matrices/"
MATEXT=".mtx"
MATRIX="30p30n"
RHS="30p30n-b"
#OUTDIR="out_test/"
#SCALING=(NONE UNITDIAG)
SCALING=(NONE)
#GMRESRELTOL=(1.0e-3 1.0e-6)
GMRESRELTOL=(1.0e-3)
#GMRESRESTART=(100 1000 2000)
GMRESRESTART=(2000)
#GMRESMAXITER=(100 1000 2000)
GMRESMAXITER=(2000)
PRECONDCHOICE=(0)
REDUCTION=(1.0e-1 1.0e-3 1.0e-10 1.0e-15)

MATRIXFILE=${MATDIR}${MATRIX}${MATEXT}
RHSFILE=${MATDIR}${RHS}${MATEXT}

mkdir -p ${LOGDIR}
#mkdir -p ${OUTDIR}
for thread in ${threads[@]}; do
  export OMP_NUM_THREADS=${thread}
  echo OMP_NUM_THREADS = $OMP_NUM_THREADS
  for scale in ${SCALING[@]}; do
    echo SCALING = ${scale}
    for restart in ${GMRESRESTART[@]}; do
      echo GMRESRESTART = ${restart}
      for maxiter in ${GMRESMAXITER[@]}; do
        echo GMRESMAXITER = ${maxiter}
        for gtol in ${GMRESRELTOL[@]}; do
          echo GMRESRELTOL = ${gtol}
          for ptol in ${REDUCTION[@]}; do
            echo REDUCTION = ${ptol}
            LOGFILE=${LOGDIR}/log_${MATRIX}_${RHS}_${thread}_${restart}_${maxiter}_${gtol}_ParilU0v02_${ptol}.m
            echo writing to ${LOGFILE}
            ${precommand} ./test_solve_pariLU0_MKL_FGMRES \
            ${MATRIXFILE} ${RHSFILE} ${LOGDIR} ${scale} ${gtol} ${restart} \
            ${maxiter} ${PRECONDCHOICE} ${ptol} > ${LOGFILE} 2>&1
          done
        done
      done
    done
  done
done

PRECONDCHOICE=(1)
LOGFILE=${LOGDIR}/log_${MATRIX}_${RHS}_${thread}_${restart}_${maxiter}_${gtol}_MKLcsriLU0_${ptol}.m
echo writing to ${LOGFILE}
${precommand} ./test_solve_pariLU0_MKL_FGMRES \
${MATRIXFILE} ${RHSFILE} ${LOGDIR} ${scale} ${gtol} ${restart} \
${maxiter} ${PRECONDCHOICE} ${ptol} > ${LOGFILE} 2>&1
