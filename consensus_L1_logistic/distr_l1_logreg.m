function [z, history] = distr_l1_logreg(A, b, mu, N, rho, alpha)
% distr_l1_logreg   Solve distributed L1 regularized logistic regression 
%
% [x, history] = distr_l1_logreg(A, b, mu, N, rho, alpha)
%
% solves the following problem via ADMM:
%
%   minimize   sum( log(1 + exp(-b_i*(a_i'w + v)) ) + m*mu*norm(w,1)
%
% where A is a feature matrix and b is a response vector. The scalar m is
% the number of examples in the matrix A. 
%
% This solves the L1 regularized logistic regression problem. It uses
% a custom Newton solver for the x-step. This version solves a distributed
% version of L1 regularized logistic regression.
%
% The solution is returned in the vector x = (v,w).
%
% history is a structure that contains the objective value, the primal and 
% dual residual norms, and the tolerances for the primal and dual residual 
% norms at each iteration.
%
% N is the number of subsystems to use to split the examples. This code
% will (serially) solve N x-updates with m / N examples per subsystem.
% Therefore, the number of examples, m, should be divisible by N. No error
% checking is done.
%
% rho is the augmented Lagrangian parameter. 
%
% alpha is the over-relaxation parameter (typical values for alpha are 
% between 1.0 and 1.8).
%
% This example requires the "MATLAB Interface for L-BFGS-B" and L-BFGS-B
% installed. These can be acquiured at
% http://www.cs.ubc.ca/~pcarbo/lbfgsb-for-matlab.html.
%
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%


t_start = tic;

%% Global constants and defaults

QUIET    = 0;
MAX_ITER = 200;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

%% Preprocessing
[m, n] = size(A);
m      = m / N;  % should be divisible


%% ADMM solver

x = zeros(n+1,N);
z = zeros(n+1,N);
u = zeros(n+1,N);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', '# bfgs', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end
p = size(z,1);
C = [-b -A]';


for k = 1:MAX_ITER
        
    % x-update 
    bfgs_iters = zeros(N,1);    
    for i = 1:N  % This update can be parallized/distributed among the nodes
        Ai      = A(1+(i-1)*m:i*m,:);
        bi      = b(1+(i-1)*m:i*m);       
        [x(:,i), newton_internal_iter] = update_x(Ai, bi, u(:,i), z(:,i), rho);        
        bfgs_iters(i)                  = newton_internal_iter;
    end    

    % z-update with relaxation at the central server
    zold          = z;
    x_hat         = alpha*x + (1-alpha)*zold;
    ztilde        = mean(x_hat + u,2);
    ztilde(2:end) = shrinkage( ztilde(2:end), (m*N)*mu/(rho*N) );
    
    z = ztilde*ones(1,N);

    % u-update
    u = u + (x_hat - z);
    
    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, b, mu, x, z(:,1));
    
    history.r_norm(k)  = norm(x - z, 'fro');
    history.s_norm(k)  = norm(rho*(z - zold),'fro');

    history.LBFGS_iters(:,k) = bfgs_iters;
        
    history.eps_pri(k) = sqrt(p*N)*ABSTOL + RELTOL*max(norm(x,'fro'), norm(z,'fro'));
    history.eps_dual(k)= sqrt(p*N)*ABSTOL + RELTOL*norm(rho*u,'fro');
 
    if ~QUIET
        fprintf('%3d\t%10d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, sum(bfgs_iters), ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end
    

    if history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k)
        %break;
    end
end

if ~QUIET
    toc(t_start);
end
z = z(:,1);
end

function obj = objective(A, b, mu, x, z)
    m   = size(A,1);
    obj = sum(log(1 + exp(-A*z(2:end) -b*z(1)))) + m*mu*norm(z(2:end),1);
end

function [obj, grad] = objective_x_update(A, b, rho, x, z, u)
    %m   = size(A,1);
    obj  = (sum(log(1 + exp(-A*x(2:end,:) -b*x(1,:))))) + rho*0.5*norm(x - z + u,2)^2;
    % gradient
    C    = [b, A];
    C_T  = C.';
    t    = ones(size(C, 1), 1);
    p    = exp( t .* (C*x) );
    grad = -C_T * (t ./ (1 + p) ) + rho*(x - z + u);
end

function [x, f, info] = bfgs_update(C, u, z, rho, N, x0)
    % solve the x update
    %   minimize [ -logistic(x_i) + (rho/2)||x_i - z^k + u^k||^2 ]
    % via L-BFGS

    [m n] = size(C);

    auxdata{1} = C;
    auxdata{2} = z;
    auxdata{3} = u;
    auxdata{4} = rho;

    %x = lbfgsb(x0, -Inf*ones(n,1), +Inf*ones(n,1), 'l2_log', 'l2_log_grad', auxdata, 'record_bfgs_iters');
    b       = C(:,1);
    A       = C(:,2:end);
    opts.x0 = x0;
    opts.printEvery = 0;
    [x, f, info]   = lbfgsb(@(x) objective_x_update(A, b, rho, x, z, u), -Inf*ones(n,1), +Inf*ones(n,1), opts);
end


function z = shrinkage(a, kappa)
    z = max(0, a-kappa) - max(0, -a-kappa);
end

function u = solv_logreg_consensus(A, b, tol, rho, z, u, u_prev)
%global warm_start_u0;
options = optimoptions(@fminunc,'Algorithm','quasi-newton', 'GradObj', 'on', 'MaxIter', 100, 'TolFun', tol, 'TolX', tol, 'Display','off');
%u = fminunc(@(x) logreg_consensus(x, A, N, v, l, t), warm_start_u0, options);
u = fminunc(@(x) objective_x_update(A, b, rho, x, z, u), u_prev, options);
% opts.Method = 'lbfgs';
% opts.Display = 'off';
% opts.MaxIter = 200;
% opts.Corr = 50;
% opts.optTol = tol;
% u = minFunc(@(x) logreg_consensus(x, A, N, v, l, t), warm_start_u0, opts);
%warm_start_u0 = u;
end


function [x, iter] = update_x(A, b, u, z, rho, x0)
    % solve the x update
    %   minimize [ -logistic(x_i) + (rho/2)||x_i - z^k + u^k||^2 ]
    % via Newton's method; for a single subsystem only.
    alpha = 0.1;
    BETA  = 0.5;
    TOLERANCE = 1e-5;
    MAX_ITER = 50;
    [m n] = size(A);
    I = eye(n+1);
    if exist('x0', 'var')
        x = x0;
    else
        x = zeros(n+1,1);
    end
    C = [-b -A];
    f = @(w) (sum(log(1 + exp(C*w))) + (rho/2)*norm(w - z + u).^2);
    for iter = 1:MAX_ITER
        fx = f(x);
        g = C'*(exp(C*x)./(1 + exp(C*x))) + rho*(x - z + u);
        H = C' * diag(exp(C*x)./(1 + exp(C*x)).^2) * C + rho*I;
        dx = -H\g;   % Newton step
        dfx = g'*dx; % Newton decrement
        if abs(dfx) < TOLERANCE
            break;
        end
        % backtracking
        t = 1;
        while f(x + t*dx) > fx + alpha*t*dfx
            t = BETA*t;
        end
        x = x + t*dx;
    end
end
