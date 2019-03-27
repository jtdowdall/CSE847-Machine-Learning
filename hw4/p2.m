data = load('ad_data.mat');
par_list  = [1e-8, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
num_features = zeros(size(par,2),1);
auc_list = zeros(size(par,2),1);
for i = 1:size(par_list,2)
    par = par_list(i);
    [w, c] = logistic_l1_train(data.X_train, data.y_train, par);
    num_features(i) = nnz(w);
    preds = data.X_test * w + c;
    [X,Y,T,AUC] = perfcurve(data.y_test,preds,1);
    auc_list(i) = AUC;
end
figure()
plot(par_list, num_features)
xlabel('Regularization parameter')
ylabel('Number of features')
title('Sparsity of lasso regression based on regularization parameter')

figure()
plot(par_list, auc_list)
xlabel('Regularization parameter')
ylabel('Area under ROC')
title('ROC performance vs regularization parameter in lasso regression')


function[w, c] = logistic_l1_train(data, labels, par)
% OUTPUT w is equivalent to the first d dimension of weights in logistic train
% c is the bias term, equivalent to the last dimension in weights in logistic train.
% Specify the options (use without modification).
opts.rFlag = 1; % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4; % termination options.
opts.maxIter = 5000; % maximum iterations.
[w, c] = LogisticR(data, labels, par, opts);
end