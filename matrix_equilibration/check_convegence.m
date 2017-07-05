function [ converged ] = check_convegence( m, n, matrix, tolerance )
%check_convegence : monitor convergence of matrix equilibration
%   convergence is achieved when all row and column norms are within
%   tolerance of 1
    converged = 0;
    % check row norms
    for i = 1:m
       tmp = 1 - norm(matrix(i,:),'inf'); 
       if ( abs( tmp ) < tolerance )
           converged = 1;
       else
           display('row')
           [ i, tmp ]
           converged = 0;
           break;
       end
       
    end
    % check column norms
    if ( converged == 1 )
        for j = 1:n
           tmp = 1 - norm(matrix(:,j),'inf');
           if ( abs( tmp ) < tolerance )
               converged = 1;
           else
               display('col')
               [ j, tmp ]
               converged = 0;
               break;
           end
        end
    end
    
end

