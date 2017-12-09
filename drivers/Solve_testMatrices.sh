#!/bin/bash

#./PariLU_laptop_testMatrices.sh airfoil_2d ../testing/matrices/airfoil_2d/ PariLU_laptop/ 

GENERATOR=(./Solve_testMatrix.sh)
MATRIXNAMES=(airfoil_2d DK01R GT01R olafu raefsky3 young3c dRdQ_sm)
MATRIXPATHS=(../testing/matrices/airfoil_2d/ ../testing/matrices/DK01R/ ../testing/matrices/GT01R/ ../testing/matrices/olafu/ ../testing/matrices/raefsky3/ ../testing/matrices/young3c/ ../testing/matrices/)
RHSNAMES=(ONES DK01R_b GT01R_b olafu_b raefsky3_b ONES ONES )
RHSPATHS=(" " ../testing/matrices/DK01R/ ../testing/matrices/GT01R/ ../testing/matrices/olafu/ ../testing/matrices/raefsky3/ " " " ") 
MACHINE=KNL


for ((i=0; i<${#MATRIXNAMES[@]}; i++)); do
#for ((i=0; i<2; i++)); do
  echo ${MATRIXNAMES[${i}]}
  echo ${MATRIXPATHS[${i}]}
  echo ${GENERATOR} ${MATRIXNAMES[${i}]} ${MATRIXPATHS[${i}]} ${RHSNAMES[${i}]} ${RHSPATHS[${i}]} ${MACHINE}
  ${GENERATOR} ${MATRIXNAMES[${i}]} ${MATRIXPATHS[${i}]} ${RHSNAMES[${i}]} ${RHSPATHS[${i}]} ${MACHINE}
done

