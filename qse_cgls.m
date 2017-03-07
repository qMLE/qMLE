function [rho, stats] = qse_cgls(operators, f, opts)
%QSE_CGLS Quantum state estimation via conjugate gradients with line search
%   RHO = QSE_CGLS(OPERATORS, F) returns the maximum-likelihood estimator 
%   RHO, numerically computed using a conjugate gradient with line search 
%   (CGLS) algorithm, for a quantum tomography problem with outcome 
%   operators specified by OPERATORS and measured outcome frequencies given 
%   by a vector F.  For more info on specifying OPERATORS, please consult 
%   <a href="matlab:help qmt">qmt</a>.
%
%   [RHO, STATS] = QSE_CGLS(...) saves additional per-iteration statistics 
%   in the STATS structure.
%
%   See also QSE_APG.

% grab dimensions
if ~isa(operators,'cell') % nonseparable
    operators = {operators};
end

dims = zeros(numel(operators),2);
for i=1:numel(operators)
    temp = size(operators{i});
    dims(i,1) = temp(1);
    dims(i,2) = temp(end);
end
d = prod(dims(:,1));
K = prod(dims(:,2));
assert(K==numel(f));

% process options
defaults = struct;
defaults.threshold_step = 0.5*sqrt(d)*d*eps; % stop when trace distance upper bound between current and previous iterate is less than this^M
defaults.threshold_fval = -f(f~=0)'*log(f(f~=0)); % stop when negative log likeilhood less than this^M
defaults.threshold_dist = 0; % stop when trace distance between rho_star and rho is less than or equal to this^M
defaults.rho_star = [];
defaults.imax = 20000;
defaults.rho0 = [];
defaults.save_rhos = false;

defaults.adjustment = 0.5;
defaults.mincondchange = -Inf;
defaults.step_adjust = 2;
defaults.a2 = 0.1;
defaults.a3 = 0.2;

if ~exist('opts','var')
    opts = defaults;
else
    % scan for invalid options
    names = fieldnames(opts);
    for i=1:numel(names)
        if ~isfield(defaults, names{i})
            error('CGLS:opts','unknown option %s',names{i});
        end
    end
    % populate defaults if not in opts
    names = fieldnames(defaults);
    for i=1:numel(names)
        if ~isfield(opts, names{i})
            opts.(names{i}) = defaults.(names{i});
        end
    end
end

if ~isempty(opts.rho0)
    [V,D] = eig(opts.rho0);
    temp = full(diag(sparse(sqrt(max(0,diag(D))))))*V';
    A = temp/norm(temp,'fro');
    rho = A'*A;
else
    A = eye(d)/sqrt(d);
    rho = eye(d)/d;
end

% line search stuff
a2 = opts.a2;
a3 = opts.a3;

stats = struct;
stats.steps = zeros(opts.imax,1); % upper bound on trace distance between current iterate and previous iterate
stats.fvals = zeros(opts.imax,1);
stats.dists = zeros(opts.imax,1); % lower bound or actual trace distance between current iterate and rho_star
stats.dists_true = false(opts.imax,1); % set to true if actual trace distance
stats.comp_prob = 0;
stats.comp_grad = 0;
stats.comp_fval = 0;
stats.comp_proj = 0;
stats.times = zeros(opts.imax,1);
stats.satisfied_step = false;
stats.satisfied_fval = false;
stats.satisfied_dist = false;

stats.best_rho = [];
stats.best_fval = Inf;

if opts.save_rhos
    stats.rhos = {};
end

start = tic;

probs = qmt(rho, operators);
stats.comp_prob = stats.comp_prob+1;
adj = f./probs;
adj(f==0)=0;
rmatrix = qmt(adj, operators, 'adjoint');

condchange = Inf;

if opts.mincondchange > 0
    hessian_proxy = f./(probs).^2;
    hessian_proxy(f==0) = 0;
end

stats.comp_grad = stats.comp_grad+1;
fval = -f(f~=0)'*log(probs(f~=0));

stats.best_rho = rho;
stats.best_fval = fval;

for i=1:opts.imax
    
    curvature_too_large = false;
    
    if opts.mincondchange > 0
        if i>1
            condchange = real(acos(real(old_hessian_proxy(:)'*hessian_proxy(:))/norm(old_hessian_proxy(:))/norm(hessian_proxy(:))));
        end
        stats.cond_change_angle(i,1) = condchange;
    end

    % the gradient
    if i==1
        % gradient
        G = A*(rmatrix - eye(d)); 
        % conjugate-gradient
        H = G; 
    else
        G_next = A*(rmatrix - eye(d));
        polakribiere = real(G_next(:)'*(G_next(:)-opts.adjustment*G(:)))/(G(:)'*G(:));
        gamma = max(polakribiere, 0);
        % conjugate-gradient
        H = G_next + gamma*H;
        % gradient
        G = G_next;
    end
    
    % line search
    A2 = A + a2*H;
    A3 = A + a3*H;
    rho2 = A2'*A2; rho2=rho2/trace(rho2);
    rho3 = A3'*A3; rho3=rho3/trace(rho3);
    probs2 = qmt(rho2, operators);
    probs3 = qmt(rho3, operators);
    stats.comp_prob = stats.comp_prob+2;
    l1 = fval;
    l2 = -f(f~=0)'*log(probs2(f~=0));
    l3 = -f(f~=0)'*log(probs3(f~=0));
    stats.comp_fval = stats.comp_fval+2;
    alphaprod = 1/2*((l3-l1)*a2^2-(l2-l1)*a3^2)/((l3-l1)*a2-(l2-l1)*a3);
    if isnan(alphaprod) || alphaprod > 1/eps || alphaprod < 0
        % fallback, find which one gives the smallest value out of 0, a2 and a3
        candidates = [0 a2 a3];
        [~,index] = min([l1 l2 l3]);
        if opts.step_adjust > 1
            if isnan(alphaprod) || alphaprod > 1/eps
                % curvature too small to estimate properly
                a2 = opts.step_adjust*a2;
                a3 = opts.step_adjust*a3;
            elseif alphaprod < 0
                % curvature too large, so steps overshoot parabola
                a2 = a2/opts.step_adjust;
                a3 = a3/opts.step_adjust;
                curvature_too_large = true;
            end
        end
        alphaprod=candidates(index);
    end
    
    % update
    A = A + alphaprod*H;
    A = A/norm(A, 'fro');
    old_rho = rho;
    rho = A'*A;

    if opts.save_rhos
        stats.rhos{i,1} = rho;
    end
    
    stats.alphas(i) = alphaprod;
    
    probs = qmt(rho, operators);
    stats.comp_prob = stats.comp_prob+1;
    fval = -f(f~=0)'*log(probs(f~=0));
    stats.comp_fval = stats.comp_fval+1;
    stats.fvals(i) = fval; 

    if stats.fvals(i) < stats.best_fval
        stats.best_fval = stats.fvals(i);
        stats.best_rho = rho;
    end
    
    stats.times(i) = toc(start);

    % check threshold
    stats.steps(i) = 0.5*sqrt(d)*norm(rho-old_rho,'fro');
    stats.satisfied_step = stats.steps(i) <= opts.threshold_step;
    stats.satisfied_fval = fval <= opts.threshold_fval;
    if ~isempty(opts.rho_star)
        stats.dists(i) = 0.5*norm(rho-opts.rho_star,'fro');
        if stats.dists(i) <= opts.threshold_dist
            % do additional check with actual trace distance
            stats.dists(i) = 0.5*sum(svd(rho-opts.rho_star));
            stats.dists_true(i) = true;
            stats.satisfied_dist = stats.dists(i) <= opts.threshold_dist;
        end
    end
    if (~curvature_too_large && stats.satisfied_step) || stats.satisfied_fval || condchange < opts.mincondchange || stats.satisfied_dist
        break;
    end

    if i~=opts.imax
        adj = f./probs;
        adj(f==0)=0;
        rmatrix = qmt(adj, operators, 'adjoint');
        stats.comp_grad = stats.comp_grad+1;
        if opts.mincondchange > 0
            old_hessian_proxy = hessian_proxy;
            hessian_proxy = f./(probs).^2;
            hessian_proxy(f==0) = 0;
        end
    end
end

stats.fvals = stats.fvals(1:i);
stats.steps = stats.steps(1:i);
stats.dists = stats.dists(1:i);
stats.dists_true = stats.dists_true(1:i);
stats.times = stats.times(1:i);

end
