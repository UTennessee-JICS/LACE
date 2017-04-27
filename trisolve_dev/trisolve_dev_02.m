% Preliminary testing and development of parallel tri-solves
% Stephen Wood, Ryan Glasby
% 20170426

clc; clear all; close all;

N = 5

A = tril(magic(N),0)

x = zeros(N,1)

R = ones(N,1)

Ainv = inv(A)

x_expected = Ainv*R

err = 1e10;
iter = 0;
tol = 1e-8;
omega = 0.05;
while (err > tol)
    x_err = R - A*x
    err = sum( (x_err).^2 );
    
    x = x + omega*x_err
    
    iter = iter + 1;
end

iter
err
x
x_expected


% x(1) = R(1) / A(1,1)
% x(2) = ( R(2) - A(2,1)*x(1) ) / A(2,2)

for i=1:N
   tmp = 0;
   for j=2:i
     tmp = tmp + A(i,j-1)*x(j-1);  
   end
   x(i) = (R(i) - tmp)/A(i,i);
end

x

x = zeros(N,1);

c = [4, 1, 5, 3, 2];

display('Parallel Forward Solve')
step = 1e10;
iter = 0;
tol = 1e-8;
while (step > tol)
    
    step = 0;
    %for i=N:-1:1
    %for i=1:N
    for ii = 1:N
        i = c(ii);
        tmp = 0;
        for j=2:i
            tmp = tmp + A(i,j-1)*x(j-1);  
        end
        %tmp_step = (R(i) - tmp)/A(i,i);
        tmp_step = (x(i) - (R(i) - tmp)/A(i,i)).^2;
        step = step + tmp_step;
        x(i) = (R(i) - tmp)/A(i,i);
    end
    
    iter = iter + 1;
end

tmp_step
iter
err
x
x_expected
error = x_expected-x



display('Parallel Backward Solve')
A = triu(magic(N),0)
x = zeros(N,1)
R = ones(N,1)
Ainv = inv(A)
x_expected = Ainv*R

% x(1) = R(1) / A(1,1)
% x(2) = ( R(2) - A(2,1)*x(1) ) / A(2,2)

% x(5) = R(5) / A(5,5)
% x(4) = ( R(4) - A(4,5)*x(5) ) / A(4,4)

for i=N:-1:1
   tmp = 0;
   for j=i+1:N
     tmp = tmp + A(i,j)*x(j);  
   end
   x(i) = (R(i) - tmp)/A(i,i);
end

x

x = zeros(N,1);

step = 1e10;
iter = 0;
tol = 1e-8;
while (step > tol && iter < 20)
    
    step = 0;
    %for i=N:-1:1
    for ii=1:N
    %for i = N:-1:1
        i = c(ii);
        tmp = 0;
        for j=i+1:N
            tmp = tmp + A(i,j)*x(j);  
        end
        %tmp_step = (R(i) - tmp)/A(i,i);
        tmp_step = (x(i) - (R(i) - tmp)/A(i,i)).^2;
        step = step + tmp_step;
        x(i) = (R(i) - tmp)/A(i,i);
    end
    
    iter = iter + 1;
end

tmp_step
iter
err
x
x_expected
error = x_expected-x