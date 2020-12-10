clear variables;

%% INPUTs

n                       = 1000; % number of training data samples for one label
svm_optimization_scheme = {'cvx_based'; 'smo'; 'top_admm'}; %'cvx_based'; 'top_admm'; 'smo';
nrof_topaddm_iter       = 1000;
rho                     = 1;
eta                     = 1; %1.015;
sigma                   = 1;
C                       = 1; % for the box constraint (high side)

% Select one of the TOP-ADMM step-sizes manually corresponding to the equivalent
% methods
select__top_admm_problem_reformulation = {'a'; 'b'; 'c'; 'd'; 'e'}; 


%% Derived parameters and dataset generation
rng(1); % For reproducibility
 
r     = sqrt(rand(n,1)); % Radius
t     = 2*pi*rand(n,1);  % Angle
data1 = [r.*cos(t), r.*sin(t)]; % Points

r2    = sqrt(3*rand(n,1)+1); % Radius
t2    = 2*pi*rand(n,1);      % Angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points

data  = [data1; data2];

% make a vector of classifications.
dataClass        = ones(2*n,1);
dataClass(1:n)   = -1;

% randomize the data and labels
p                = randperm(length(dataClass));
data             = data(p, :);
dataClass        = dataClass(p);

% Train with 50% of data.
train_fraction = .5;
nTrain         = floor(train_fraction *length(p));

%# split into train/test datasets
trainData    = data(1:nTrain, :);%(1:150,:);
testData     = data(nTrain+1:end, :);%(151:270,:);
trainClass   = dataClass(1:nTrain, :);%(1:150,:);
testClass    = dataClass(nTrain+1:end, :);%(151:270,:);
numTrain     = size(trainData,1);
numTest      = size(testData,1);

% Create a kernel matrix
rbfKernel = @(X,Y) exp(-sigma .* pdist2(X,Y,'euclidean').^2);

%# compute kernel matrices between every pairs of (train,train) and
%# (test,train) instances and include sample serial number as first column
% K_     = [ (1:numTrain)' , rbfKernel(trainData,trainData) ];
% KK_    = [ (1:numTest)'  , rbfKernel(testData,trainData)  ];
K      = rbfKernel(trainData,trainData) ; 
KK     = rbfKernel(testData,trainData);


%%
figure(1); clf;
figure(2); clf;

legend_str_vec = {};

for ii = 1:length(svm_optimization_scheme)
    clear objective_val legend_str;
    switch lower(svm_optimization_scheme{ii})
        case {'matlab_toolbox'; 'smo'}
            %Train the SVM Classifier
            cl      = fitcsvm(trainData,trainClass,'KernelFunction','rbf',...
                'BoxConstraint',C,'ClassNames',[-1,1], 'verbose', 2);
            [yt]        = predict(cl,trainData);
            MyAcc_train = (1 - nnz(trainClass - sign(yt))/length(trainClass));
            
            % Test data
            [yp]        = predict(cl,testData);
            MyAcc_test  = (1 - nnz(testClass - sign(yp))/length(testClass));
            
            objective_val{1}   = cl.ConvergenceInfo.History.Objective; %cl.ConvergenceInfo.Objective * ones(1, nrof_topaddm_iter);
            test_accuracy_val  = MyAcc_test * ones(1, nrof_topaddm_iter);
            
            legend_str{1}      = sprintf('SMO: [%2.2f%%; %2.2f%%]', MyAcc_train*100, MyAcc_test*100);
            
            % Predict scores over the grid
            d = 0.02;
            [x1Grid,x2Grid] = meshgrid(min(trainData(:,1)):d:max(trainData(:,1)),...
                min(trainData(:,2)):d:max(trainData(:,2)));
            xGrid = [x1Grid(:),x2Grid(:)];
            [~, scores]  = predict(cl,xGrid);
            % Plot the data and the decision boundary
            figure(2);
            h(1:2) = gscatter(trainData(:,1),trainData(:,2),trainClass,'rb','.');
            hold on
            ezpolar(@(x)1);
            h(3) = plot(trainData(cl.IsSupportVector,1),trainData(cl.IsSupportVector,2),'ko');
            contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
            legend(h,{'-1','+1','Support Vectors'});
            axis equal
            hold off
            
            
        case 'cvx_based'
            y     = trainClass;
            len_y = length(y);
            %Train the SVM Classifier
            cvx_begin  %quiet
            variable x(len_y);
            minimize( 0.5* quad_form((x .* y), K) - sum(x)  );
            subject to
            y'*x == 0;
            0 <= x <= C;
            cvx_end
            MyAcc_train = kernel_svm_classification_test(x, C, K, K, trainClass, trainClass);
            
            % Test data
            MyAcc_test         = kernel_svm_classification_test(x, C, K, KK, trainClass, testClass);
            
            objective_val{1}   = cvx_optval * ones(1, nrof_topaddm_iter);
            test_accuracy_val  = MyAcc_test * ones(1, nrof_topaddm_iter);
            
            legend_str{1}      = sprintf('CVX: [%2.2f%%; %2.2f%%]', MyAcc_train*100, MyAcc_test*100);
            
        case 'top_admm'
            
            for jj = 1:numel(select__top_admm_problem_reformulation)
                
                switch lower(select__top_admm_problem_reformulation{jj})
                    case 'a'
                        tau           = 0.0175;
                        vartheta      = 0.0;
                    case 'b'
                        tau           = 0;
                        vartheta      = 0.0125;
                    case 'c'
                        tau           = 0.0175;
                        vartheta      = 0.0175;
                    case 'd'
                        tau           =  0.0175;
                        vartheta      = -0.0075;
                    case 'e'
                        tau           = -0.001;
                        vartheta      = 0.01;
                    otherwise
                        error('Unknown problem formulation');
                end
                
                if 0 % if you increase C then increase stepsize appropriately
                    tau      = tau*1.025;
                    vartheta = vartheta*1.0;
                elseif 1
                    tau      = tau/4.99;
                    vartheta = vartheta/4.95;
                end
                
                x  = zeros(length(trainClass),1);
                y  = x;
                z  = x;
                
                K_bar   = diag(trainClass) * K * diag(trainClass);
                grad_h  = @(x) K_bar * x - 1;
                obj_fun = @(x) 0.5*x'*K_bar*x - sum(x);
                
                objective_val{jj} = zeros(1, nrof_topaddm_iter);
                
                for kk = 1:nrof_topaddm_iter
                    
                    % z update
                    z_ = (x - tau * grad_h(z) + y/rho);
                    z  = project_hyperplane_constraint(z_, trainClass, 0);
                    
                    % x update
                    x_ = z - vartheta *  grad_h(x) - y/rho;
                    x  = project_box_constraint(x_, 0, C);
                    
                    % y update
                    y  = y + eta*rho*(x - z);
                    
                    % objective value at the given iteration
                    objective_val{jj}(kk) = obj_fun(z);
                    
                    % Training accuracy
                    MyAcc_train       = kernel_svm_classification_test(x, C, K, K, trainClass, trainClass);
                    % Test data
                    MyAcc_test        = kernel_svm_classification_test(x, C, K, KK, trainClass, testClass);
                    test_accuracy_val(kk)  = MyAcc_test;
                end
                legend_str{jj}    = sprintf('TOP-ADMM (\\tau=%1.4f, \\vartheta=%1.4f): [%2.2f%%; %2.2f%%]', tau, vartheta, MyAcc_train*100, MyAcc_test*100);
                
            end
        otherwise
            error('Unknown method');
            
    end

for jj = 1:length(legend_str)
    legend_str_vec = [legend_str_vec; legend_str{jj}];
end
%% Plot: Objective versus Iterations
figure(1); 
for jj = 1:length(objective_val)
semilogy(abs(objective_val{jj}), 'linewidth', 2.5);
end
if ii==1
    hold all;
end
ylabel('Objective');
xlabel('Iterations');
legend(legend_str_vec, 'interpreter', 'tex')

end

%%


%%
function K = km_kernel(X1,X2,ktype,kpar)

switch ktype
	case 'gauss'	% Gaussian kernel
		sgm = kpar;	% kernel width
		
		dim1 = size(X1,1);
		dim2 = size(X2,1);
		
		norms1 = sum(X1.^2,2);
		norms2 = sum(X2.^2,2);
		
		mat1 = repmat(norms1,1,dim2);
		mat2 = repmat(norms2',dim1,1);
		
		distmat = mat1 + mat2 - 2*X1*X2';	% full distance matrix
		K = exp(-distmat/(2*sgm^2));
		
	case 'gauss-diag'	% only diagonal of Gaussian kernel
		sgm = kpar;	% kernel width
		K = exp(-sum((X1-X2).^2,2)/(2*sgm^2));
		
	case 'poly'	% polynomial kernel
		p = kpar(1);	% polynome order
		c = kpar(2);	% additive constant
		
		K = (X1*X2' + c).^p;
		
	case 'linear' % linear kernel
		K = X1*X2';
		
	otherwise	% default case
		error ('unknown kernel type')
end

end



function p = project_hyperplane_constraint(z, y, a)

% compute the projection: y' * z = eta
mu = ( a - sum(y.*z) ) ./ sum(y.^2);
p  = z + bsxfun(@times, mu, y);

end

 function p = project_box_constraint(x, low, high)
 
 % compute the projection: low <= x <= high
 p = min(x, high);
 p = max(p, low);

 end