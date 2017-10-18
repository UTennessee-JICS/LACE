function [ f, sm, lm, sr, lr, li, si ] = plot_eigs( A, n, matname )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here


f = figure(n);
hold on
sm = eigs(A,3,'sm');
plot(sm,'r*')

lm = eigs(A,3,'lm');
plot(lm,'b*')

sr = eigs(A,3,'sr');
plot(sr,'mo')

lr = eigs(A,3,'lr');
plot(lr,'co')

opts.p = 45;
li = eigs(A,3,'li',opts);
plot(li,'g^')

si = eigs(A,3,'si',opts);
plot(si,'k^')

legend('Smallest magnitude SM','Largest magnitude LM','Smallest real SR',...
    'Largest real LR','Largest imaginary LI','Smallest imaginary SI')

xlabel('Real axis')
ylabel('Imaginary axis')
str = sprintf("Six types of eigenvalues in the %s matrix", matname); 
%title('Six types of eigenvalues')
title(str)

end

