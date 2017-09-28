function [u,nu] = housegen(x)
    % [u,nu] = housegen(x)
    % Generate Householder reflection.
    % G. W. Stewart, "Matrix Algorithms, Volume 1", SIAM, 1998.
    % [u,nu] = housegen(x).
    % H = I - uu' with Hx = -+ nu e_1
    %    returns nu = norm(x).
    u = x;
    nu = norm(x);
    if nu == 0
        u(1) = sqrt(2);
        return
    end
    u = x/nu;
    if u(1) >= 0
        u(1) = u(1) + 1;
        nu = -nu;
    else
        u(1) = u(1) - 1;
    end
    u = u/sqrt(abs(u(1)));
end

function u = house_gen(x)
    % u = house_gen(x)
    % Generate Householder reflection.
    % u = house_gen(x) returns u with norm(u) = sqrt(2), and
    % H(u,x) = x - u*(u'*x) = -+ norm(x)*e_1.
    
    % Modify the sign function so that sign(0) = 1.
    sig = @(u) sign(u) + (u==0);
    
    nu = norm(x);
    if nu ~= 0
        u = x/nu;
        u(1) = u(1) + sig(u(1));
        u = u/sqrt(abs(u(1)));
    else
        u = x;
        u(1) = sqrt(2);
    end
end

function w = house_reflect(r,j)
    % w = house_reflect(r)
    % Generate Householder reflection, w, for the vector r.
    % w = house_gen(r) returns u with norm(w) = 1, and
    % P(w,r) = r - 2*w*(w'*r) = -+ norm(r)*e_1.
    
    nu = norm(r);
    beta = sign(r(j))*nu;
    eta = r;
    for i = 1:j-1
      eta(i) = 0;      
    end
    eta(j) = beta + r(j);
    nu = norm(eta);
    w = eta/nu;
    
end