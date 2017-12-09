#!/bin/bash

#./PariLU_laptop_testMatrices.sh airfoil_2d ../testing/matrices/airfoil_2d/ PariLU_laptop/ 

GENERATOR=(./PariLU_testMatrix.sh)
MATRIXNAMES=(airfoil_2d DK01R GT01R olafu raefsky3 young3c)
MATRIXPATHS=(../testing/matrices/airfoil_2d/ ../testing/matrices/DK01R/ ../testing/matrices/GT01R/ ../testing/matrices/olafu/ ../testing/matrices/raefsky3/ ../testing/matrices/young3c/)
OUTPUTDIR=PariLU_laptop/


for ((i=0; i<${#MATRIXNAMES[@]}; i++)); do
  echo ${MATRIXNAMES[${i}]}
  echo ${MATRIXPATHS[${i}]}
  echo ${GENERATOR} ${MATRIXNAMES[${i}]} ${MATRIXPATHS[${i}]} ${OUTPUTDIR}
  ${GENERATOR} ${MATRIXNAMES[${i}]} ${MATRIXPATHS[${i}]} ${OUTPUTDIR}
done

