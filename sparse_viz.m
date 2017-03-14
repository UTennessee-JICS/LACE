clc; clear all; close all;
format longE;

%[A, rows, cols, entries] = mmread('Trefethen_20.mtx');
[A, rows, cols, entries] = mmread('out_test30p30n_LpariLUv0_2_diff.mtx');
[i,j,s] = find(A);
maxabs = max(abs(s));
pltsizes = ((abs(s)+1)/maxabs);
pltcolors = abs(s);

figure(1)
scatter3(j,i,pltcolors,[],pltcolors)
set(gca,'Ydir','reverse')
xlabel('j')
ylabel('i')
zlabel('|L_{MKL}-L_{pariLU(tol=0.1)}|')
colorbar
% see link below to automate data tip placement at maximum value
% http://stackoverflow.com/questions/29882186/set-data-tips-programmatically


[A, rows, cols, entries] = mmread('out_test30p30n_UpariLUv0_2_diff.mtx');
[i,j,s] = find(A);
maxabs = max(abs(s));
pltsizes = ((abs(s)+1)/maxabs);
pltcolors = abs(s);

figure(2)
scatter3(j,i,pltcolors,[],pltcolors)
set(gca,'Ydir','reverse')
xlabel('j')
ylabel('i')
zlabel('|U_{MKL}-U_{pariLU(tol=0.1)}|')
colorbar