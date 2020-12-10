function MyAcc = kernel_svm_classification_test(mySol, C, K, KK, trainClass, testClass)

%       Classification accuracy test
nzMySol = find(0 < mySol <= C);
w_x     =   K * (mySol .* trainClass);
b       = (1/nnz(nzMySol)) * sum (w_x(nzMySol) - trainClass(nzMySol));
predict = (KK * (mySol.* trainClass) - b);
MyAcc   = (1 - nnz(testClass - sign(predict))/length(testClass));