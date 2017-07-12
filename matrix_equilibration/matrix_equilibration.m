clc; clear all; close all;

tolerance = 1e-3
maxiter = 2

% m = 1000;
% n = m;
% density = 0.1;
% orgdiagscal = 0.05;
% rng(1,'twister');
% A = sprand(m,n,density) + sparse([1:m],[1:n],orgdiagscal*rand(min(m,n),1),m,n);

% nx = 10
% ny = 10
% n = (3*nx*ny + 2*nx + 2*ny + 1)
% m = n
% A = gallery('wathen',nx,ny,1);

% nx = 10
% A = gallery('tridiag',nx);
% m = size(A,1)
% n = size(A,2)

%[A, N, cols, entries] = mmread('testing/matrices/Trefethen_20.mtx');
%[A, N, cols, entries] = mmread('testing/matrices/paper1_matrices/apache2_rcm.mtx');L2D_1024_5pt.mtx
%[A, N, cols, entries] = mmread('testing/matrices/2cubes_sphere.mtx');
[A, N, cols, entries] = mmread('testing/matrices/30p30n-A3.mtx');
entries
n = N
m = n;

B = rand(m,1);
cond_A = condest(A)

Dr = speye(m,m);
Dc = speye(n,n);

DrADc = Dr*A*Dc;

iter = 0;
while ( check_convegence( m, n, DrADc, tolerance ) ~= 1 && iter < maxiter )
    
    % calculate row norms and update Dr 
    for i = 1:m
       Dr(i,i) = Dr(i,i) * 1/sqrt(norm(DrADc(i,:),'inf'));
    end
    
    % calculate col norms and update Dc
    for j = 1:n
       Dc(j,j) = Dc(j,j) * 1/sqrt(norm(DrADc(:,j),'inf'));
    end
    
    DrADc = Dr*A*Dc;
    cond_DrADc = condest(DrADc)

    iter = iter + 1
    
end

% cond_DrADc = condest(DrADc)

x = A\B;

x_scaled = Dc*((DrADc)\(Dr*B));

err = norm(x - x_scaled)

DrB = Dr*B;

mmwrite('30p30n-A3_me2_DrADc.mtx',DrADc,'DrADc');
mmwrite('30p30n-A3_me2_DrRHS.mtx',DrB,'DrB');
mmwrite('30p30n-A3_me2_Dc.mtx',Dc,'Dc');



