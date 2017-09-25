clc; clear all; close all;

[x_mgs, rows, cols, entries] = mmread('../out_test_fgmres_10/30p30n_solution.mtx');
[x_house, rows, cols, entries] = mmread('../out_test_fgmresh_10/30p30n_solution.mtx');                        


discrepancy = norm(x_mgs - x_house)

figure(3)
plot(x_mgs)
hold all
plot(x_house)
xlabel('Degrees of Freedom')
ylabel('State values')
title({'Solution vector found by FGMRES for one psuedo-time step of the 30p30n benchmark'})
legend('FGMRES(MGS)+PariLU(1.5e+5)+MKLCSRTRSV','FGMRES(HouseHolder)+PariLU(1.5e+5)+MKLCSRTRSV')

figure(4)
plot(abs(x_mgs - x_house))
xlabel('Degrees of Freedom')
ylabel('State values abs(x_{MGS} - x_{Householder})')
title({'Difference in solution vectors found by'; 'FGMRES for one psuedo-time step of the 30p30n benchmark'})


rho_mgs = x_mgs(1:5:end);
rho_house=x_house(1:5:end);
u_mgs = x_mgs(2:5:end);
u_house = x_house(2:5:end);
v_mgs = x_mgs(3:5:end);
v_house = x_house(3:5:end);
temp_mgs = x_mgs(4:5:end);
temp_house = x_house(4:5:end);
nut_mgs = x_mgs(5:5:end);
nut_house = x_house(5:5:end);

figure(5)
plot(rho_mgs)
hold all
plot(rho_house)
xlabel('Degrees of Freedom')
ylabel('State values')
title({'Solution vector found by FGMRES for one psuedo-time step of the 30p30n benchmark'})
legend('FGMRES(MGS)+PariLU(1.5e+5)+MKLCSRTRSV','FGMRES(HouseHolder)+PariLU(1.5e+5)+MKLCSRTRSV')

figure(6)
plot(abs(rho_mgs - rho_house))
xlabel('Degrees of Freedom')
ylabel('Density discrepancy abs(\rho_{MGS} - \rho_{Householder})')
title({'Difference in density found by'; 'FGMRES for one psuedo-time step of the 30p30n benchmark'})


max_rho_diff = max(rho_mgs - rho_house)
max_u_diff = max(u_mgs - u_house)
max_v_diff = max(v_mgs - v_house)
max_temp_diff = max( temp_mgs(1:100) - temp_house(1:100) )
max_nut_diff = max( nut_mgs(1:100) - nut_house(1:100) )