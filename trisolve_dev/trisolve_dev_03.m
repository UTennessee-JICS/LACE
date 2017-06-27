% Preliminary testing and development of parallel tri-solves
% Stephen Wood, Ryan Glasby
% 20170426

clc; clear all; close all;
format longE;

[A_all, rows, cols, entries] = mmread('Trefethen_20.mtx');
%[A_all, rows, cols, entries] = mmread('../testing/matrices/paper1_matrices/ani5_crop.mtx');

N = rows
A = tril(A_all,-1);
A = A + speye(N);
figure(1)
spy(A)
%A = full(A);
hold all;


x = zeros(N,1);
R = ones(N,1);
% x = ones(N,1);
% R = A*x;

x_expected = A\R;
sys_expected_error = norm(R - A*x_expected)
x_fexp = x_expected;

% display('Serial Forward Solve')
% for i=1:N
%    tmp = 0;
%    for j=2:i
%      tmp = tmp + A(i,j-1)*x(j-1);  
%    end
%    x(i) = (R(i) - tmp)/A(i,i);
% end
% 
% %x
% error = x_expected-x;

x = zeros(N,1);

%c = N:-1:1;
c = randperm(N);


display('Parallel Forward Solve')
step = 1e10;

iter = 0;
tol = 1e-15;
while (step > tol)
    
    step = 0;
    for ii = 1:N
        i = c(ii);
        tmp = 0;
        for j=1:i-1
            tmp = tmp + A(i,j)*x(j);  
        end
%         tmp_step = (x(i) - (R(i) - tmp)/A(i,i)).^2;
%         step = step + tmp_step;
%         x(i) = (R(i) - tmp)/A(i,i);
        tmp = (R(i) - tmp)/A(i,i);
        tmp_step = (x(i) - tmp).^2;
        step = step + tmp_step;
        x(i) = tmp;
    end
    %[iter step]
    iter = iter + 1;
    c = randperm(N);
end

step
iter

x;
error = x_expected-x;
x_error = norm(error)
sys_error = norm(R - A*x)
x_forward = x;

A = triu(A_all,0);
spy(A,'r+')
%A = full(A);
x = zeros(N,1);
R = ones(N,1);
x_expected = A\R;
sys_expected_error = norm(R - A*x_expected)
x_bexp = x_expected;

% display('Serial Backward Solve')
% 
% for i=N:-1:1
%    tmp = 0;
%    for j=i+1:N
%      tmp = tmp + A(i,j)*x(j);  
%    end
%    x(i) = (R(i) - tmp)/A(i,i);
% end
% 
% x;

x = zeros(N,1);
c = randperm(N);
%c = N:-1:1;

display('Parallel Backward Solve')
step = 1e10;
iter = 0;
tol = 1e-15;
while (step > tol)
    
    step = 0;
    for ii=1:N
        i = c(ii);
        tmp = 0;
        for j=i+1:N
            tmp = tmp + A(i,j)*x(j);  
        end
        tmp = (R(i) - tmp)/A(i,i);
        tmp_step = (x(i) - tmp).^2;
        step = step + tmp_step;
        x(i) = tmp;
    end
    %[iter step]
    iter = iter + 1;
    c = randperm(N);
end

step
iter
x;
error = x_expected-x;
x_error = norm(error)
sys_error = norm(R - A*x)
x_backward = x;