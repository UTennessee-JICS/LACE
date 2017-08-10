%#codegen
function [x, reserror, ortherr_mgs, total_iters] = gmres_orthog_est(x0, b, atv, params)
% GMRES linear equation solver
% Implementation following Saad-Schultz
%
% C. T. Kelley, July 10, 1994
%
% This code comes with no guarantee or warranty of any kind.
%
% function [x, error, total_iters] = gmres(x0, b, atv, params)
%
%
% Input: x0 = initial iterate
%        b = right hand side
%        atv, a matrix-vector product routine
%            atv must return Ax when x is input
%            the format for atv is
%            function ax = atv(x)
%            Note that for GMRES we incorporate any 
%            preconditioning into the atv routine.
%        params = three dimensional vector to control iteration
%            params(1) = relative residual reduction factor
%            params(2) = max number of iterations
%            params(3) (Optional) = reorthogonalization method
%                   1 -- Brown/Hindmarsh condition (default)
%                   2 -- Never reorthogonalize (not recommended)
%                   3 -- Always reorthogonalize (not cheap!)
%
% Output: x=solution
%         error = vector of residual norms for the history of
%            the iteration
%         total_iters = number of iterations
%
% Requires givapp.m

%
% initialization
%
n=length(b);
errtol=params(1);
kmax=params(2);
reorth=1;
if length(params) == 3
    reorth=params(3);
end
x=x0;
I = eye(kmax+1); % for orthoganalization error estimate
%
%
h=zeros(kmax);
v=zeros(n,kmax);
c=zeros(kmax+1,1);
s=zeros(kmax+1,1);
if norm(x) ~=0
   r = b-feval(atv,x);
else
   r = b;
end
rho=norm(r);
g=rho*eye(kmax+1,1);
errtol=errtol*norm(b);
reserror=[];
ortherr_mgs=[];
%
% test for termination on entry
%
reserror=[reserror,rho];
total_iters=0;
if(rho < errtol) 
    return
end
%
v(:,1)=r/rho;
beta=rho;
k=0;
%
% GMRES iteration
%
while((rho > errtol) && (k < kmax))
    k=k+1;
    v;
    v(:,k+1)=feval(atv,v(:,k));
    normav=norm(v(:,k+1));
   
%
% Modified Gram-Schmidt
%
    for j=1:k
        h(j,k)=v(:,j)'*v(:,k+1);
        v(:,k+1)=v(:,k+1)-h(j,k)*v(:,j);
    end
    h(k+1,k)=norm(v(:,k+1));
    normav2=h(k+1,k);
%
% Reorthogonalize?
%
    if  (reorth == 1 && normav + .001*normav2 == normav) || reorth ==  3
        display('Reorthogonalize')
        k
        for j=1:k
            hr=v(:,j)'*v(:,k+1);
            h(j,k)=h(j,k)+hr;
            v(:,k+1)=v(:,k+1)-hr*v(:,j);
        end
        h(k+1,k)=norm(v(:,k+1));
    end
  
%
%   watch out for happy breakdown 
%
    if(h(k+1,k) ~= 0)
         v(:,k+1)=v(:,k+1)/h(k+1,k);
    end

%
%   Assess orthogonality error
%
    ortherr_mgs = [ortherr_mgs, norm(v(:,1:k+1)'*v(:,1:k+1)-I(1:k+1,1:k+1),inf)];    
    
%
%   Form and store the information for the new Givens rotation
%
    if k > 1
        h(1:k,k)=givapp(c(1:k-1),s(1:k-1),h(1:k,k),k-1);
    end
    nu=norm(h(k:k+1,k));
    if nu~=0
%        c(k)=h(k,k)/nu;
        c(k)=conj(h(k,k)/nu);
        s(k)=-h(k+1,k)/nu;
        h(k,k)=c(k)*h(k,k)-s(k)*h(k+1,k);
        h(k+1,k)=0;
        g(k:k+1)=givapp(c(k),s(k),g(k:k+1),1);
    end
%
% Update the residual norm
%
    rho=abs(g(k+1));
    reserror=[reserror,rho];
end
%
% At this point either k > kmax or rho < errtol.
% It's time to compute x and leave.
%
y=h(1:k,1:k)\g(1:k);
total_iters=k;
x = x0 + v(1:n,1:k)*y;
