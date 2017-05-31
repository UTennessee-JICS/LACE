% Condition number estimation of sparse matrices
%
% based on William W. Hager, "Condition Estimates" 
% http://epubs.siam.org/doi/pdf/10.1137/0905023
% http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=napack%2Fcon.f
%
clc; clear all; close all;

N = 10000

density = 0.01;
diagscale = 1.;
A = sprand(N,N,density) + diagscale*speye(N,N);
%[L,U] = lu(A);
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
if ( LL ~= 4) 
    S = 0.;
    for k = 1:N
       j = LL - k;
       T = A(I+j);
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

iter
if (iter == 10)
    CON = -1;
end
CON
mlcon = condest(A)
if (CON > mlcon)
    estimation_error = CON/mlcon
else
    estimation_error = mlcon/CON
end
if (CON > 0 && estimation_error <= 10)
    display('success')
else
    display('failure')
    M
    i
    D
    C
    R
end
