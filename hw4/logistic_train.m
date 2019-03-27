function [weights] = logistic_train(data, labels, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
% data = n * (d+1) matrix withn samples and d features, where
% column d+1 is all ones (corresponding to the intercept term)
% labels = n * 1 vector of class labels (taking values 0 or 1)
% epsilon = optional argument specifying the convergence
% criterion - if the change in the absolute difference in
% predictions, from one iteration to the next, averaged across
% input features, is less than epsilon, then halt
% (if unspecified, use a default value of 1e-5)
% maxiter = optional argument that specifies the maximum number of
% iterations to execute (useful when debugging in case your
% code is not converging correctly!)
% (if unspecified can be set to 1000)
%
% OUTPUT:
% weights = (d+1) * 1 vector of weights where the weights correspond to
% the columns of "data"
%
labels(labels==-1.0)=0.0;
if ~exist('epsilon','var'), epsilon=1e-5; end
if ~exist('maxiter','var'), maxiter=1000; end
d = size(data, 2);
N = size(data, 1);
weights = zeros(d,1);
old_t = ones(N,1)*2;
for i = 1:maxiter
    t = data*weights;
    %mean(abs(t-old_t))
    if mean(abs(t-old_t)) < epsilon, break; end;
    old_t = t;
    b = 1 ./ (1 + exp(-t)) - labels;
    gradient = 1 / N * data' * b;
    weights = weights - gradient;
end

