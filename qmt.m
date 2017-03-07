function X = qmt(X, operators, type, allow_negative)
%QMT Quantum measurement transform
%   P = QMT(RHO, OPERATORS) applies the Born rule to the density matrix RHO
%   to obtain measurement probabilities for each of the operators specified
%   by OPERATORS.  RHO is a D-by-D Hermitian matrix, P is a vertical vector
%   of length K that stores the results, and OPERATORS can be one of the 
%   following (and will dictate how computation will be performed):
%
%    (1) (for near-perfect measurements) A D-by-R-by-K multidimensional
%        array where R<D, and the operator for the Ith measurement is given
%        by OPERATORS(:,:,I)*OPERATORS(:,:,I)'.  When R==1, then this can
%        also be given as a D-by-K matrix.
%
%    (2) (for imperfect measurements) A D-by-D-by-K multidimensional 
%        array where OPERATORS(:,:,I) is a matrix corresponding to the 
%        Ith measurement operator, e.g. |I><I|
%
%    (3) (for product measurements) An N-element cell array, where each 
%        element is of the form specified in either (1) or (2).  The 
%        resulting operators are Kronecker products.  
%
%        For example, suppose OPERATORS = {X, Y, Z}, where X is a
%        D1-by-D1-by-2 array, Y is a D2-by-D2-by-5 array, and Z is a
%        D3-by-D3-by-3 array, with D=D1*D2*D3 and K=2*5*3.  Then, the
%        operator corresponding to measurement number (A-1)*5*3+(B-1)*3+C
%        is equal to kron(kron(X(:,:,A),Y(:,:,B)),Z(:,:,C))
%
%   R = QMT(COEFFS, OPERATORS, 'adjoint') performs the adjoint operation,
%   where the resulting matrix R is a linear mixture of the operator
%   matrices with coefficients specified by COEFFS, a vector of length K.
%   OPERATORS are specified in the same fashion as above.
%
%   See also QSE_APG.

%% setup

% wrap "naked" operator specifications
if ~isa(operators,'cell') 
    operators = {operators};
end

% specify default type of operation
if ~exist('type','var')
    type = 'forward';
end

if ~exist('allow_negative','var')
    allow_negative = false;
end

% determine sizes
n = numel(operators);
Ds = zeros(1, n);
Ks = zeros(1, n);
Rs = zeros(1, n);
factored = false(1, n);
for i=1:n
    dims=size(operators{i});
    if numel(dims) > 3
        error('qmt:operators','OPERATORS must be either 2-dimensional or 3-dimensional arrays');
    end
    Ds(i)=dims(1);
    Ks(i)=dims(end);
    factored(i) = (length(dims)==2 || dims(2) ~= dims(1));
    if numel(dims) == 2
        Rs(i) = 1;
    else
        Rs(i) = dims(2);
    end
    if factored(i)
        operators{i} = reshape(operators{i},Ds(i),Rs(i),Ks(i));
    else
        operators{i} = reshape(operators{i},Ds(i)*Ds(i),Ks(i));
    end
end

%% main processing

switch(type)
    case 'forward'
        % shuffle rho
        X = shuffle_forward(X, fliplr(Ds));
        % main loop
        for i=n:-1:1
            P = operators{i};
            if factored(i)
                P = reshape(P, Ds(i),Rs(i)*Ks(i));
                X = reshape(X, Ds(i), []);
                X = P'*X;
                X = reshape(X, Ks(i), Rs(i)*Ds(i), []);
                X = sum(bsxfun(@times, reshape(P,Ds(i)*Rs(i),Ks(i)).',X),2);
                X = reshape(X, Ks(i), []);
                X = X.';
            else
                X = reshape(X, Ds(i)*Ds(i), []);
                X = P'*X;
                X = X.';
            end
        end
        % post-process probabilities
        X = real(X(:));
        if ~allow_negative
            X = max(0,X);
        end
    case 'adjoint'
        % main loop
        for i=n:-1:1
            P = operators{i};
            if factored(i)
                P = reshape(P, Ds(i)*Rs(i), Ks(i));
                X = reshape(X, Ks(i), 1, []);
                X = bsxfun(@times, P', X);
                X = reshape(X, Ks(i)*Rs(i), []);
                P = reshape(P, Ds(i), Rs(i)*Ks(i));
                X = P*X;
                X = reshape(X, Ds(i)^2, []);
                X = X.';
            else
                X = reshape(X, Ks(i), []);
                X = P*X;
                X = X.';
            end
        end
        % post-process matrix
        X = shuffle_adjoint(X, fliplr(Ds));
        X = 0.5*(X+X');
    otherwise
        error('qmt:type','TYPE must be either ''forward'' or ''adjoint''');
end

end

%% helper functions

function rho = shuffle_forward(rho, dims)
    n = numel(dims);
    rho = reshape(rho, [dims dims]);
    ordering = reshape([1:n; (n+1):(2*n)],1,[]);
    rho = permute(rho, ordering);
end

function R = shuffle_adjoint(R, dims)
    n = numel(dims);
    R = reshape(R, reshape([dims; dims],1,2*n));
    ordering = [1:2:(2*n),2:2:(2*n)];
    R = permute(R, ordering);
    R = reshape(R, prod(dims), prod(dims));
end
