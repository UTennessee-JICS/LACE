clc; clear all; close all;

tic
[Aani5, rows, cols, entries] = mmread('testing/matrices/paper1_matrices/ani5_crop.mtx');
toc(tic)
tic
condest(Aani5)
toc(tic)
tic
plot_eigs(Aani5,1,'ani5\_crop')
toc(tic)
tic
invAani5 = inv(Aani5);
toc(tic)
tic
mmwrite('ani5_crop_inv.mtx',invAani5)
toc(tic)

tic
[A30p30n, rows, cols, entries] = mmread('testing/matrices/30p30n.mtx');
toc(tic)
tic
condest(A30p30n)
toc(tic)
tic
plot_eigs(A30p30n,2,'30p30n')
toc(tic)
tic
invA30p30n = inv(A30p30n);
toc(tic)
tic
mmwrite('30p30n_inv.mtx',invA30p30n)
toc(tic)