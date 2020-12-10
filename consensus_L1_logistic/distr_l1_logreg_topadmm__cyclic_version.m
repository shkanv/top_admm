function [z, history] = distr_l1_logreg_topadmm__cyclic_version(A, b, mu, N, rho, alpha, gamma, step_size__x_m, step_size__z)
% distr_l1_logreg   Solve distributed L1 regularized logistic regression 
%
% [x, history] = distr_l1_logreg_topadmm(A, b, mu, N, rho, alpha, gamma, step_size__x_m, step_size__z)
%
% solves the following problem via TOP-ADMM:
%
%   minimize   sum( log(1 + exp(-b_i*(a_i'w + v)) ) + m*mu*norm(w,1)
%
% where A is a feature matrix and b is a response vector. The scalar m is
% the number of examples in the matrix A. 
%
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
% gamma is the other variant of over-relaxation parameter that only relaxes
% the dual update. 
%
% step_size__x_m is a step-size that is used in the x_m updates---that is referred to as vartheta whereas
% step_size__z is a step-size that is used in the z update---that is referred to as tau
% 

t_start = tic;

%% Global constants and defaults

QUIET    = 0;
MAX_ITER = 1000;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

%% Preprocessing
[m, n] = size(A);
m      = m / N;  % should be divisible


tol = 1e-5; 
%% ADMM solver

x_hat = zeros(n+1,N);
x     = zeros(n+1,N);
z     = zeros(n+1,N);
y     = zeros(n+1,N);


if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      '\Delta r norm', 'eps pri', 's==\Delta z norm', 'eps dual', 'objective');
end
p = size(z,1);
C = [-b -A]';

%global BFGS_ITERS;  % global variable to keep track of bfgs iterations
bfgs_iters = zeros(N,1);

%x_prev = zeros(n+1,N);

for k = 1:MAX_ITER
    
    % z-update with relaxation
    zold          = z;
    x_hat         = alpha*x + (1-alpha)*zold;
    ztilde        = mean(x_hat + y/rho - step_size__z.*grad_log_reg(A, b, zold), 2);
    ztilde(2:end) = shrinkage( ztilde(2:end), (m*N)*mu/(rho*N) );
    
    z = ztilde*ones(1,N);
    
            
    % xi-update     
    xold = x;
    
    for i = 1:N        
        
        if 1
            % serial x-update
            %x = solv_logreg_consensus(K, m, tol, v, l, t)
            Ai      = A(1+(i-1)*m:i*m,:);
            bi      = b(1+(i-1)*m:i*m);            
                       
            grad_at_xoldi   = grad_log_reg(Ai, bi, xold(:,i)); % == g = C'*(exp(C*x)./(1 + exp(C*x)));
            
            x(:,i)          = z(:,i) - step_size__x_m.*grad_at_xoldi - y(:,i)/rho;
            
        elseif 0
            % serial x-update with proximity (generalized)
            %x = solv_logreg_consensus(K, m, tol, v, l, t)
            Ai      = A(1+(i-1)*m:i*m,:);
            bi      = b(1+(i-1)*m:i*m);            
                       
            grad_at_xoldi   = grad_log_reg(Ai, bi, xold(:,i)); % == g = C'*(exp(C*x)./(1 + exp(C*x)));
            
            alpha_          = mean([abs(step_size__x_m), abs(step_size__z)] )* 1.001; % need to be adaptive to make it work
            x(:,i)          = (1/1+alpha_)*(alpha_*x(:,i) + z(:,i) - step_size__x_m.*grad_at_xoldi - y(:,i)/rho);
            
        end
        
        bfgs_iters(i) = i;
    end    
    x_hat         = alpha*x + (1-alpha)*zold;
    
    % u-update
    y = y + gamma*rho*(x_hat - z);

    
    
    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, b, mu, x, z(:,1));
    
    history.r_norm(k)  = norm(x - z, 'fro');
    history.s_norm(k)  = norm(rho*(z - zold),'fro');

    history.LBFGS_iters(:,k) = bfgs_iters;
        
    history.eps_pri(k) = sqrt(p*N)*ABSTOL + RELTOL*max(norm(x,'fro'), norm(z,'fro'));
    history.eps_dual(k)= sqrt(p*N)*ABSTOL + RELTOL*norm(rho*y,'fro');
 
    if ~QUIET
        fprintf('%3d\t\t%10.4f\t\t%10.4f\t%10.4f\t\t%10.4f\t\t%10.2f\n', k, ...
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

function [grad] = grad_log_reg(A, b, x)
    %m   = size(A,1);
    %obj  = (sum(log(1 + exp(-A*x(2:end,:) -b*x(1,:))))) + rho*0.5*norm(x - z + u,2)^2;
    % gradient
    C    = [b, A];
    C_T  = C.';
    t    = ones(size(C, 1), 1);
    p    = exp( t .* (C*x) );
    grad = -C_T * (t ./ (1 + p) ); % + rho*(x - z + u);
end


function z = shrinkage(a, kappa)
    z = max(0, a-kappa) - max(0, -a-kappa);
end


