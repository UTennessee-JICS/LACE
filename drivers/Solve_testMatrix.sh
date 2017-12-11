#!/bin/bash

echo arg1 ${1}
echo arg2 ${2}
echo arg3 ${3}
echo arg4 ${4}
echo arg5 ${5}

MATRIXNAME=${1}
MATRIXSUFFIX=.mtx
MATRIXPATH=${2}
MATRIX=${MATRIXPATH}${MATRIXNAME}${MATRIXSUFFIX}
echo ${MATRIX}
RHSNAME=${3}
RHSPATH=${4}
if [ "${RHSNAME}" == "ONES" ]; then
  RHS=ONES
  MACHINE=${4}
else 
  RHS=${RHSPATH}/${RHSNAME}${MATRIXSUFFIX}
  MACHINE=${5}
fi
echo ${RHS} 

PRECONDIR=PariLU_${MACHINE}
OUTDIR=Solver_${MACHINE}

SOLVERS=(FGMRES FGMRESH)
TOL=(1.0e-10)
SEARCHMAX=(1000)

mkdir -p ${OUTDIR}

Lfiles=("$( ls ${PRECONDIR}/${MATRIXNAME}_L*.mtx )")
Ufiles=("$( ls ${PRECONDIR}/${MATRIXNAME}_U*.mtx )")

IFS=' ' read -r -a precond_L <<< $Lfiles
IFS=' ' read -r -a precond_U <<< $Ufiles

echo length of precond_L is ${#precond_L[@]}

for s in ${SOLVERS[@]}; do
  for ((i=0; i<${#precond_L[@]}; i++)); do
    outfile=$(printf '%s/log_%s_%s_%s_%02d.m' "${OUTDIR}" "${s}" "${MACHINE}" "${MATRIXNAME}" "${i}")
    echo ${outfile}
    echo % ${precond_L[${i}]} > ${outfile} 
    echo % ${precond_U[${i}]} >> ${outfile}
    ./solver --solver ${s} --matrix ${MATRIX} --L ${precond_L[${i}]} --U ${precond_U[${i}]} --RHS ${RHS} --outDir ${OUTDIR} --tolType 0 --tol ${TOL} --searchMax ${SEARCHMAX} --csrtrsvType 0 --monitorOrthog 1 >> ${outfile} 2>&1
  echo ${i}
  done
done
