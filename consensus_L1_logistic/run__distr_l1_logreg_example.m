% Distributed L1 regularized logistic regression that compares the
% performance of TOP-ADMM against ADMM.

% BASELINE SOURCE CODE REFERENCE: S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein, “Distributed
% optimization and statistical learning via the alternating direction method
% of multipliers,” Foundations and Trends® in Machine Learning, vol. 3,
% no. 1, pp. 1–122, 2011.
% https://web.stanford.edu/~boyd/papers/admm/logreg-l1/distr_l1_logreg_example.html

clear; 
%% INPUT: Select the problem formulation to be solved by TOP-ADMM
select__top_admm_problem_reformulation = {'a'; 'b'; 'c'; 'd'; 'e'}; %{'a'; 'b'; 'c'; 'd'; 'e'}; % 'a'; 'b'; 'c'; 'd'; 'e' 

%% Generate problem data
rng('default');
rng(0);

logisticfun = @(x) 1./(1+exp(-x));

addpath(genpath(pwd));

n = 100; % feature length
p = 200; % dimension of matrix Ai is n x p 
M = 100; % the number of examples in the matrix A

w = sprandn(n, 1, 100/n);      % N(0,1), 10% sparse
v = randn(1);                  % random intercept

X0     = sprandn(p*M, n, 10/n);  % data / observations
b_true = sign(X0*w + v);
y      = (X0*w + v) > 0;

% noise is function of problem size use 0.1 for large problem
b0 = sign(X0*w + v + sqrt(0.1)*randn(p*M, 1)); % labels with noise

% packs all observations in to an p*M x n matrix
A0 = spdiags(b0, 0, p*M, p*M) * X0;

ratio = sum(b0 == 1)/(p*M);
mu    = (0.001)*1/(p*M) * norm((1-ratio)*sum(A0(b0==1,:),1) + ratio*sum(A0(b0==-1,:),1), 'inf');

x_true = [v; w];

legend_list = {};
%% Parameters for TOP-ADMM/ADMM

% some predefined parameters
rho          = 1;
gamma        = 1; % can help in accelerating the convergence
alpha_relax  = 1; % can help in accelerating the convergence


%% RUN CLASSICAL ADMM (that utilizes QUASI-NEWTON METHOD internally)

[x_admm, history_admm] = distr_l1_logreg(A0, b0, mu, M, rho, alpha_relax);    

K_admm                 = length(history_admm.objval);

%% Post processing
X               = X0;
b_admm          = (X*x_admm(2:end) + x_admm(1));

% compute the output (class assignment) of the model for each data point
modelfit_true           = logisticfun(b_true) > 0.5;
modelfit_admm           = logisticfun(b_admm) > 0.5;

% calculate percent correct (percentage of data points% that are correctly classified by the model)
pctcorrec_true          = sum(modelfit_true==y) / length(y) * 100;
pctcorrec_admm          = sum(modelfit_admm==y) / length(y) * 100;

legend_list             = [legend_list; sprintf('ADMM (%1.2f%% training error; avg LBFGS iters %1.2f)', (100-pctcorrec_admm), mean(mean(history_admm.LBFGS_iters,1)))];
%% Plot ADMM results
h = figure(1); clf
semilogy(cumsum(mean(history_admm.LBFGS_iters,1)), history_admm.objval, 'MarkerSize', 10, 'LineWidth', 2); hold all; 
legend(legend_list);
ylabel('$\sum_m h_m(x_m^k) + g(z^k)$'); xlabel('iter ($k$)');

figure(2); clf;
semilogy(cumsum(mean(history_admm.LBFGS_iters,1)), max(1e-8, history_admm.r_norm),  'LineWidth', 2); hold all; 
ylabel('primal residual: ||r||_2');
legend(legend_list);

figure(3); clf;
semilogy(cumsum(mean(history_admm.LBFGS_iters,1)), max(1e-8, history_admm.s_norm), 'LineWidth', 2); hold all;
ylabel('dual residual: ||\Delta x||_2'); xlabel('iter (k)');
legend(legend_list);

%% RUN TOP-ADMM 

for ii = 1:numel(select__top_admm_problem_reformulation)

if ii==1; clear_figure_flag = true; else; clear_figure_flag = false; end
    
% According to problem formulation select some appropriate step-sizes
switch lower(select__top_admm_problem_reformulation{ii})
    case 'a'
        step_size__x = 0;
        step_size__z = 0.0025;
    case 'b'
        step_size__x = 0.035;
        step_size__z = 0.0;
    case 'c'
        step_size__x = 0.025; 
        step_size__z = 0.0025; 
    case 'd'
        step_size__x = -0.00255; %-0.0005; 
        step_size__z = 0.00255; %0.0025; 
    case 'e'
        step_size__x = 0.075; 
        step_size__z = -0.0001; 
    otherwise
        error('Unknown problem formulation');
end

% RUN CORE TOP-ADMM

[x_topadmm, history_topadmm] = distr_l1_logreg_topadmm(A0, b0, mu, M, rho, gamma, alpha_relax, step_size__x, step_size__z); % it is a cyclic and updates shuffled version to analyze the convergence
%[x_topadmm, history_topadmm] = distr_l1_logreg_topadmm__cyclic_version(A0, b0, mu, M, rho, gamma, alpha_relax, step_size__x, step_size__z); % it is an ordered version


%% POST-PROCESSING

b_topadmm               = (X*x_topadmm(2:end) + x_topadmm(1));
modelfit_topadmm        = logisticfun(b_topadmm) > 0.5;
pctcorrec_topadmm       = sum(modelfit_topadmm==y) / length(y) * 100;


%% Reporting

K_topadmm   = length(history_topadmm.objval);

legend_list = [legend_list; sprintf('TOP-ADMM(%s): %1.2f%% training error; \\tau=%1.5f, \\vartheta=%1.5f', ...
                lower(select__top_admm_problem_reformulation{ii}), (100-pctcorrec_topadmm), step_size__z, step_size__x)];


figure(1); 
semilogy(1:K_topadmm, history_topadmm.objval, 'MarkerSize', 10, 'LineWidth', 2); 
legend(legend_list);
ylabel('$\sum_m h_m(x_m^k) + g(z^k)$'); xlabel('iter ($k$)');

figure(2); 
semilogy(1:K_topadmm, max(1e-8, history_topadmm.r_norm),  'LineWidth', 2);  
ylabel('primal residual: ||r||_2');
legend(legend_list);
ylim([10e-5, 10e2]);


figure(3); 
semilogy(1:K_topadmm, max(1e-8, history_topadmm.s_norm), 'LineWidth', 2);  
ylabel('dual residual: ||\Delta x||_2'); xlabel('iter (k)');
legend(legend_list);
ylim([10e-5, 10e2]);

end


