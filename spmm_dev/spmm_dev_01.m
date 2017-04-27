% Preliminary testing and development of parallel sparse-matrix multiply
% Stephen Wood, Ryan Glasby
% 20170427

clc; clear all; close all;

N = 5

A = magic(N)

B = magic(N)

C_expected = A*B

C = zeros(N,N);

c = [4, 1, 5, 3, 2];

display('Dense-matrix multiply')
step = 1e10;
iter = 0;
tol = 1e-8;
%while (step > tol)
    
    step = 0;
    for ii = 1:N
        i = c(ii);
        for j = 1:N
            tmp = 0;
            for k=1:N
                tmp = tmp + A(i,k)*B(k,j);  
            end
            tmp_step = (C(i,j) - tmp).^2;
            step = step + tmp_step;
            C(i,j) = tmp;
        end
    end
    
    iter = iter + 1;
%end

tmp_step
iter
C
C_expected
error = C_expected-C

