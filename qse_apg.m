function [rho, stats] = qse_apg(operators, f, opts)
%QSE_APG Quantum state estimation via accelerated projected gradient (APG)
%   RHO = QSE_APG(OPERATORS, F) returns the maximum-likelihood estimator 
%   RHO, numerically computed using an accelerated projected gradient (APG)
%   algorithm, for a quantum tomography problem with outcome operators 
%   specified by OPERATORS and measured outcome frequencies given by a 
%   vector F.  For more info on specifying OPERATORS, please consult <a href="matlab:help
%   qmt">qmt</a>.
%
%   [RHO, STATS] = QSE_APG(...) saves additional per-iteration statistics 
%   in the STATS structure.
%
%   RHO = QSE_APG(OPERATORS, F, OPTS) uses additional options in the struct
%   OPTS.  The following fields are supported:
%
%    rho0                 Initial guess for RHO.  Acceptable values include 
%                         matrices as well as the strings 'white' and 
%                         'bootstrap'.  The former initializes APG with the 
%                         maximally mixed density matrix; the latter uses 
%                         <a href="matlab:help qt_cgls">CG</a> to first find an estimate for RHO before 
%                         switching to APG. 
%                         [default:'bootstrap']
%
%    threshold_step       Terminate when trace distance between current and
%                         previous iterate is guaranteed to be smaller.
%                         [default:0.5*sqrt(d)*d*eps where d is the number
%                         of rows in rho and eps is machine precision]
% 
%    threshold_fval       Terminate when the merit function, i.e., the
%                         scaled negative log-likelihood is below this
%                         value.
%                         [default:the function value when the computed
%                         probabilities matches the measured frequencies
%                         exactly] 
%
%    threshold_dist       Terminate when the current trace distance between
%                         RHO and the value given in rho_star is less than
%                         this.
%                         [default:0]
%
%    rho_star             A density matrix for reference.
%                         [default:[]]
%
%    imax                 Maximum number of iterations to run.
%                         [default:2000]
%
%    save_rhos            Store intermediate values of RHO.
%                         [default:false]
%
%    guard                Prevent varrho from generating negative
%                         probabilities.
%                         [default:true]
%
%    bb                   Use Barzilai-Borwein to initialize the step size
%                         at every iteration if possible. [Ref 1]
%                         [default:true]
%
%    accel                Acceleration method.  This can either be
%                         'fista' (the original FISTA formula),
%                         'fista_tfocs' (with updates for changing step
%                         sizes in the TFOCS paper [Ref 2]) or 'none' (no
%                         acceleration).
%                         [default:'fista_tfocs']
%
%    t0                   Initial step size for the first iteration.  Set
%                         to [] for automatic estimation.
%                         [default:[]]
%
%    restart_grad_param   Specify extra sensitivity for gradient restart.
%                         [default:0.01]
%
%    bfactor              Backtracking factor.
%                         [default:0.5]
%
%    afactor              How much to grow step sizes when Barzilai-Borwein
%                         is not used.
%                         [default:1.1]
%
%    minimum_t            Minimum step size to prevent infinite
%                         backtracking
%                         [default:eps]
%
%    bootstrap_threshold  Controls how quickly we transition from CG to APG
%                         [default:0.01]
%
%   See also QMT, PROJ_SPECTRAHEDRON, QSE_CGLS.
%
%   References
%   ----------
%
%   1. Barzilai, J., & Borwein, J. M. (1988). Two-point step size gradient 
%      methods. IMA journal of numerical analysis, 8(1), 141-148.
%
%   2. Becker, S. R., Candès, E. J., & Grant, M. C. (2011). Templates for 
%      convex cone problems with applications to sparse signal recovery. 
%      Mathematical programming computation, 3(3), 165.


%% Setup

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

% common options
defaults.rho0 = 'bootstrap'; % initial density matrix
defaults.threshold_step = 0.5*sqrt(d)*d*eps; % stop when trace distance upper bound between current and previous iterate is less than this
defaults.threshold_fval = -f(f~=0)'*log(f(f~=0)); % stop when negative log likeilhood less than this
defaults.threshold_dist = 0; % stop when trace distance between rho_star and rho is less than or equal to this
defaults.rho_star = [];
defaults.imax = 2000; % maximum number of iterations

% instrumentation
defaults.save_rhos = false; % save intermediate rho values

% tweak behavior type of algorithm
defaults.guard = true; % guard against varrho exiting
defaults.bb = true; % use Barzilai-Bowein to initialize step sizes instead of afactor increases
defaults.accel = 'fista_tfocs'; % different ways to grow and compute beta

% tweak behavior parameters 
defaults.t0 = []; % set to a scalar if we know a good starting t, set to [] for Lipschitz estimation
defaults.restart_grad_param = 0.01; % how much extra sensitivity (>=0.0, <1.0)
defaults.bfactor = 0.5; % t backtracks by this much
defaults.afactor = 1.1; % t grows by this much
defaults.minimum_t = eps; % stop if t gets smaller than this
defaults.bootstrap_threshold = 0.01; % threshold for bootstrap using CG (smaller value -> longer to run CG)

if ~exist('opts','var')
    opts = defaults;
else
    % scan for invalid options
    names = fieldnames(opts);
    for i=1:numel(names)
        if ~isfield(defaults, names{i})
            error('qt_apg:opts','unknown option %s',names{i});
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

% discard zero-valued frequencies (they have no effect on the merit
% function)
fmap = (f~=0);
f = f(fmap);
coeff_temp = zeros(K,1);

start = tic;

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
stats.thetas = zeros(opts.imax,1);
stats.ts = zeros(opts.imax,1);
stats.satisfied_step = false;
stats.satisfied_fval = false;
stats.satisfied_dist = false;

stats.best_rho = [];
stats.best_fval = Inf;

if opts.save_rhos
    stats.rhos = {};
end

if ischar(opts.rho0)
    switch(opts.rho0)
        case 'white'
            rho = eye(d)/d;
        case 'bootstrap'
            % run CG with line search until "condition number" of
            % adjustment vector stops changing by much
            opts2=struct; opts2.imax=opts.imax; opts2.mincondchange=opts.bootstrap_threshold; 
            opts2.threshold_fval=opts.threshold_fval;
            opts2.threshold_dist=opts.threshold_dist;
            opts2.rho_star = opts.rho_star;
            opts2.save_rhos = opts.save_rhos;
            coeff_temp(fmap) = f;
            time_offset = toc(start);
            [rho, boot_stats] = qse_cgls(operators,coeff_temp,opts2);
            opts.imax = opts.imax-numel(boot_stats.fvals); % reduce number of iterations we can run
            boot_stats.times = boot_stats.times+time_offset;
            if boot_stats.satisfied_fval || boot_stats.satisfied_dist
                % CG run was good enough
                stats = boot_stats;
                return;
            end
        otherwise
            error('qt_apg:initializer',['unknown initializer specification: ',opts.rho0]);
    end
else
    rho = opts.rho0;
end

varrho = rho;
varrho_changed = true;
gradient = [];
theta = 1;
t = opts.t0;
fval = Inf;
bb_okay = false; % don't do two-step Barzilai-Borwein on first step

% compute initial probabilities
probs_rho = qmt(rho, operators, 'forward');
probs_rho = probs_rho(fmap);
stats.comp_prob = stats.comp_prob+1;
stats.comp_fval = stats.comp_fval+1;
probs_varrho = probs_rho;

%% Main loop

for i=1:opts.imax
    % compute new gradient if varrho has changed
    if varrho_changed
        if opts.bb && i>1
            old_gradient = gradient;
        end
        coeff_temp(fmap) = f./probs_varrho;
        
        gradient = -qmt(coeff_temp, operators, 'adjoint');
        
        stats.comp_grad = stats.comp_grad+1;
        fval_varrho = -f'*log(probs_varrho);
        stats.comp_fval = stats.comp_fval+1;
        
        if opts.bb && bb_okay
            varrho_diff = varrho(:)-old_varrho(:);
            gradient_diff = gradient(:)-old_gradient(:);
            denominator = gradient_diff'*gradient_diff;
            if denominator > 0
                t = abs(real(varrho_diff'*gradient_diff))/denominator;
            end
        end
    end
    
    if i==1 && exist('boot_stats','var')
        boot_stats.fvals(end) = fval_varrho;
    end
    
    if isempty(t)
        % compute local Lipschitz constant from derivatives
        probs_gradient = qmt(-gradient, operators, 'forward', true);
        probs_gradient = probs_gradient(fmap);
        first_deriv = -f'*(probs_gradient./probs_varrho);
        second_deriv = f'*(probs_gradient.^2./(probs_varrho.^2));
        stats.comp_prob = stats.comp_prob+1;
        t = -first_deriv/second_deriv;
    else
        if ~(opts.bb && bb_okay)
            t = t * opts.afactor;
        end
    end
    
    % backtrack for finding a good value of t
    t_good = false;
    fval_new = [];
    t_threshold = opts.minimum_t / norm(gradient(:));
    while ~t_good
        if ~isempty(fval_new)
            new_t_estimate = second_order/max(0,fval_new-fval_varrho-first_order);
            % this comparison is false if any term is NaN (namely
            % new_t_estimate)
            if new_t_estimate > t_threshold
                t = min(t*opts.bfactor, new_t_estimate);
            else
                % if new_t_estimate is NaN or less than or equal to t_threshold
                t = t*opts.bfactor;
            end
        end
        rho_new = proj_spectrahedron(varrho - t*gradient);
        stats.comp_proj = stats.comp_proj + 1;
        probs_rho_new = qmt(rho_new, operators, 'forward', true);
        probs_rho_new = probs_rho_new(fmap); 
        fval_new = -f'*log(probs_rho_new);
        stats.comp_prob = stats.comp_prob+1;
        stats.comp_fval = stats.comp_fval+1;
        stats.fvals(i) = fval_new;
        delta = rho_new(:) - varrho(:);
        first_order = real(gradient(:)'*delta(:));
        second_order = 0.5*delta(:)'*delta(:);
        % multiplied by t so that we don't get NaN or Inf if t is too small
        t_good = ~(t*fval_new > t*fval_varrho + t*first_order + 0.9*second_order); % not greater than catches NaNs
    end
    
    if fval_new < stats.best_fval
        stats.best_fval = fval_new;
        stats.best_rho = rho_new;
    end

    if opts.save_rhos
        stats.rhos{i,1} = rho_new;
    end
    
    stats.ts(i) = t;
    stats.thetas(i) = theta;

    % check threshold
    stats.steps(i) = 0.5*sqrt(d)*norm(rho_new-rho,'fro');
    stats.satisfied_step = stats.steps(i) <= opts.threshold_step;
    stats.satisfied_fval = fval_new <= opts.threshold_fval;
    
    if ~isempty(opts.rho_star)
        stats.dists(i) = 0.5*norm(rho_new-opts.rho_star,'fro');
        if stats.dists(i) <= opts.threshold_dist
            % do additional check with actual trace distance
            stats.dists(i) = 0.5*sum(svd(rho_new-opts.rho_star));
            stats.dists_true(i) = true;
            stats.satisfied_dist = stats.dists(i) <= opts.threshold_dist;
        end
    end
    
    if t < t_threshold || stats.satisfied_step || stats.satisfied_fval || stats.satisfied_dist
        stats.times(i) = toc(start);
        rho = rho_new;
        break;
    end

    % record previous value of varrho for Barzilai-Borwein
    if opts.bb
        old_varrho = varrho;
    end
    
    % check whether to do adaptive restart
    vec1 = varrho(:)-rho_new(:);
    vec2 = rho_new(:)-rho(:);
    do_restart = real(vec1'*vec2) > -opts.restart_grad_param*norm(vec1)*norm(vec2);

    % enable Barzilai-Borwein for next iteration if no restart  
    bb_okay = ~do_restart;
    
    % perform the restart if needed
    if do_restart
        varrho = rho;
        probs_varrho = probs_rho;
        varrho_changed = (theta>1);
        theta = 1;
        stats.times(i) = toc(start);
        stats.fvals(i) = fval;
        if opts.save_rhos
            stats.rhos{i,1} = rho;
        end
        continue;
    end
        
    % acceleration
    if i>1 && stats.ts(i) > eps
        Lfactor = stats.ts(i-1)/stats.ts(i);
    else
        Lfactor = 1;
    end
    switch(opts.accel)
        case 'fista'
            theta_new = (1+sqrt(1+4*theta^2))/2;
            beta = (theta-1)/theta_new;
            theta = theta_new;
        case 'fista_tfocs'
            theta_hat = sqrt(Lfactor)*theta;
            theta_new = (1+sqrt(1+4*theta_hat^2))/2;
            beta = (theta_hat-1)/theta_new;
            theta = theta_new;
        case 'none'
            beta = 0;
        otherwise
            error('qt_apg:accel','unknown acceleration scheme');
    end
    
    % update    
    varrho = rho_new + beta*(rho_new-rho);
    probs_varrho = probs_rho_new + beta*(probs_rho_new-probs_rho);
    if opts.guard && min(probs_varrho) <= 0
        % discard momentum if momentum causes varrho to become infeasible
        % retain theta to keep estimate of current condition number
        varrho = rho_new;
        probs_varrho = probs_rho_new;
    end
    varrho_changed = true;
    rho = rho_new;
    probs_rho = probs_rho_new;
    fval = fval_new;
    stats.times(i) = toc(start);
end

%% Collect stats

stats.fvals = stats.fvals(1:i);
stats.steps = stats.steps(1:i);
stats.dists = stats.dists(1:i);
stats.dists_true = stats.dists_true(1:i);
stats.times = stats.times(1:i);
stats.thetas = stats.thetas(1:i);
stats.ts = stats.ts(1:i);

if exist('boot_stats','var')
    if opts.save_rhos
        stats.rhos = [boot_stats.rhos;stats.rhos];
    end
    stats.fvals = [boot_stats.fvals;stats.fvals];
    stats.steps = [boot_stats.steps;stats.steps];
    stats.dists = [boot_stats.dists;stats.dists];
    stats.dists_true = [boot_stats.dists_true;stats.dists_true];
    stats.times = [boot_stats.times;stats.times];
    stats.comp_prob = stats.comp_prob + boot_stats.comp_prob;
    stats.comp_grad = stats.comp_grad + boot_stats.comp_grad;
    stats.comp_fval = stats.comp_fval + boot_stats.comp_fval;
    stats.comp_proj = stats.comp_proj + boot_stats.comp_proj;
end

end