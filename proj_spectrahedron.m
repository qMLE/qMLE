function X=proj_spectrahedron(Y)

%PROJ_SPECTRAHEDRON Project a matrix onto the spectrahedron
% X=PROJ_SPECTRAHEDRON(Y) returns a positive semidefinite matrix X such
% that the trace of X is equal to 1 and the Frobenius norm between X and
% Hermitian matrix Y is minimized.

    % perform eigenvalue decomposition and remove the imaginary components
    % that arise from numerical precision errors
    [Q,L] = eig(Y);
    v = real(diag(L));
    
    % project the eigenvalues onto the probability simplex
    u = sort(v,'descend');
    sv = cumsum(u);
    rho = find(u > (sv - 1) ./ (1:length(u))', 1, 'last');
    theta_ = (sv(rho)-1)/rho;
    w = max(v - theta_, 0);
    
    % reconstitute the matrix while ensuring positive semidefiniteness
    X = bsxfun(@times, Q, sqrt(w(:).'));
    X = X*X';

end
