#!/bin/bash

MATRIXNAME=30p30n
MATRIXSUFFIX=.mtx
MATRIXPATH=../testing/matrices/
MATRIX=${MATRIXPATH}${MATRIXNAME}${MATRIXSUFFIX}
echo ${MATRIX}
RHSNAME=../testing/matrices/30p30n-b.mtx
echo ${RHSNAME}

MACHINE=Skylake
PRECONDIR=PariLU_${MACHINE}
OUTDIR=Solver_${MACHINE}

SOLVERS=(FGMRES FGMRESH)
TOL=(1.0e-10)
SEARCHMAX=(2000)

mkdir -p ${OUTDIR}

Lfiles=("$( ls ${PRECONDIR}/*_L*.mtx )")
Ufiles=("$( ls ${PRECONDIR}/*_U*.mtx )")

IFS=' ' read -r -a precond_L <<< $Lfiles
IFS=' ' read -r -a precond_U <<< $Ufiles

echo length of precond_L is ${#precond_L[@]}

for s in ${SOLVERS}; do
  for ((i=0; i<${#precond_L[@]}; i++)); do
    outfile=$(printf '%s/log_%s_%s_%s_%02d.m' "${OUTDIR}" "${s}" "${MACHINE}" "${MATRIXNAME}" "${i}")
    echo ${outfile}
    echo % ${precond_L[${i}]} > ${outfile} 
    echo % ${precond_U[${i}]} >> ${outfile}
    ./solver --solver ${s} --matrix ${MATRIX} --L ${precond_L[${i}]} --U ${precond_U[${i}]} --RHS ${RHSNAME} --outDir ${OUTDIR} --tolType 0 --tol ${TOL} --searchMax ${SEARCHMAX} --csrtrsvType 0 --monitorOrthog 1 >> ${outfile} 2>&1
  echo ${i}
  done
done
