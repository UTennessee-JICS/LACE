clc; clear all; close all;



[A, N, cols, entries] = mmread('testing/matrices/bcsr_test.mtx');
[B, N, cols, entries] = mmread('testing/matrices/diagbcsr_scaled.mtx');

[i,j,s] = find(A);
[m,n] = size(A);
D = abs(diag(A));
s1 = s;
for k=1:entries
   s1(k) = s(k)/D( i(k) );
end
A1 = sparse(i,j,s1,m,n);

[i,j,s] = find(A);
[m,n] = size(A);
D = 1.0./sqrt(abs(diag(A)));
s2 = s;
for k=1:entries
    s2(k) = s(k)* D( i(k) ) * D( j(k) );
end
A2 = sparse(i,j,s2,m,n);

rows = zeros(n, 1);
cols = zeros(n, 1);
maxrows = zeros(n, 1);
maxcols = zeros(n, 1);
for i = 1 : n
    rows(i)  = 1 / norm(A(i,:));
    cols(i)  = 1 / norm(A(:,i));
    maxrows(i)  = 1 / max(A(i,:));
    maxcols(i)  = 1 / max(A(:,i));
end

rA = diag(rows) * A;
rAr = diag(rows) * A * diag(rows);
rAc = diag(rows) * A * diag(cols);
cA = diag(cols) * A;
cAc = diag(cols) * A * diag(cols);
cAr = diag(cols) * A * diag(rows);


coA = condest(A)
cdA = condest(A1)
cdAd = condest(A2)
crA = condest( rA ) 
crAr = condest( rAr ) 
crAc = condest( rAc ) 
ccA = condest( cA )  
ccAc = condest( cAc ) 
ccAr = condest( cAr ) 

cBA = condest(B)

figure(1)
spy(A)
hold all
spy(B,'r+')
spy(A1,'ko')

figure(2)
[i,j,s] = find(A);
maxabs = max(abs(s));
pltsizes = ((abs(s))/maxabs);
pltcolors = abs(s);

scatter3(j,i,pltcolors,[],pltcolors)
set(gca,'Ydir','reverse')
xlabel('j')
ylabel('i')
zlabel('val')
str = sprintf('A condest = %e', coA)
title(str)
colorbar



figure(3)
[i,j,s] = find(rAc);
maxabs = max(abs(s));
pltsizes = ((abs(s)+1)/maxabs);
pltcolors = abs(s);

scatter3(j,i,pltcolors,[],pltcolors)
set(gca,'Ydir','reverse')
xlabel('j')
ylabel('i')
zlabel('val')
str = sprintf('rAc condest = %e', crAc)
title(str)
colorbar



figure(4)
[i,j,s] = find(B);
maxabs = max(abs(s));
pltsizes = ((abs(s)+1)/maxabs);
pltcolors = abs(s);

scatter3(j,i,pltcolors,[],pltcolors)
set(gca,'Ydir','reverse')
xlabel('j')
ylabel('i')
zlabel('val')
str = sprintf('BA condest = %e', cBA)
title(str)
colorbar