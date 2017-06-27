% Condition number estimation of sparse matrices
%
% based on William W. Hager, "Condition Estimates" 
% http://epubs.siam.org/doi/pdf/10.1137/0905023
% http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=napack%2Fcon.f
%
clc; clear all; close all;


% N = 10000
% 
% density = 0.01;
% diagscale = 1.;
% A = sprand(N,N,density) + diagscale*speye(N,N);

% [A, N, cols, entries] = mmread('testing/matrices/Trefethen_20.mtx');
% [A, N, cols, entries] = mmread('testing/matrices/paper1_matrices/ani5_crop.mtx');
 [A, N, cols, entries] = mmread('testing/matrices/paper1_matrices/apache2_rcm.mtx');
% [A, N, cols, entries] = mmread('testing/matrices/paper1_matrices/ecology2_rcm.mtx');
% [A, N, cols, entries] = mmread('testing/matrices/paper1_matrices/G3_circuit_rcm.mtx');
% [A, N, cols, entries] = mmread('testing/matrices/paper1_matrices/L2D_1024_5pt.mtx');
% [A, N, cols, entries] = mmread('testing/matrices/paper1_matrices/L3D_64_27pt.mtx');
% [A, N, cols, entries] = mmread('testing/matrices/paper1_matrices/offshore_rcm.mtx');
% [A, N, cols, entries] = mmread('testing/matrices/paper1_matrices/parabolic_fem_rcm.mtx');
% [A, N, cols, entries] = mmread('testing/matrices/paper1_matrices/thermal2_rcm.mtx');
% [A, N, cols, entries] = mmread('testing/matrices/30p30n.mtx');

tic
setup.type = 'nofill';
[L, U] = ilu(A, setup);

% setup and initialize vector workspace for condition estimation
B = zeros(N,1);
M = 0; 
C = 1./N;
for j = 1:N
   B(j) = C; 
end

% calculate norm of pivot row 
R = 0.;
O = N + 1;
P = O + 1;
LL = 5 + N*P;
I = -N -3;
LL = LL - O;
LL = N*N
if ( LL ~= 4) 
    S = 0.;
    for k = 1:N
       %j = LL - k;
       %T = A(I+j);
       j = LL - (k-1)*N;
       T = A(j);
       S = S + abs(T);
    end
    if ( R < S ) 
        R = S;
    end
end

D = -1e-10;
test = 1;
iter = 0;
while (test>0 && iter < 10) 
    display('prepare to solve system')
    % Solve
    y = L\B;
    B = U\y;
    %
    C = 0.;
    display('prepare to solve transposed system')
    for j = 1:N
       C = C + abs(B(j));
       if (B(j) < 0.)
          B(j) = -1.; 
       else
          B(j) = 1.; 
       end

    end
    % Solve Transposed system
    y = U'\B; 
    B = L'\y;
    %
    i = 1;
    for j = 1:N
       if ( abs(B(i)) < abs(B(j)) )
           i = j;
       end
    end
    %M
    %i
    %D
    %C
    
    if ( (M == i) || (D >= C) )
       display('(M == i) || (D >= C)')
       C = C*R; %
       if ( C < 1. )
           C = 1.;
       end
       CON = C;
       test = 0;
    else %if ( M == 0 ) 
       display('else prepare to continue iterating')
       M = i;
       D = C;
       for j = 1:N
           B(j) = 0.; 
       end 
       B(M) = 1.;
       % Solve
       test = 1;
    end
    iter = iter + 1;
end
hagerconesttime = toc;
iter
if (iter == 10)
    CON = -1;
end
CON
tic
mlcon = condest(A)
mlconesttime = toc;
% if (CON > mlcon)
%     estimation_error = CON/mlcon
% else
%     estimation_error = mlcon/CON
% end
estimation_error = CON/mlcon
if (CON > 0 && 0.1 <=estimation_error && estimation_error <= 10)
    display('success')
    success = 1;
else
    display('failure')
    M
    i
    D
    C
    R
    success = 0;
end

hagerconesttime
mlconesttime

format longE
[ iter; CON; mlcon; estimation_error; hagerconesttime; mlconesttime; mlconesttime/hagerconesttime; success ]
