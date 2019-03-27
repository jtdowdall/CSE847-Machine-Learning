data = load('data.txt');
labels = load('labels.txt');
% train split
train_data =  data(1:2000,:);
train_labels = labels(1:2000,:);
% test split
test_data = data(2001:4601,:);
test_labels = labels(2001:4601,:);
%logistic regression

n_list = [200 500 800 1000 1500 2000];
acc_list = zeros(6,1);
for i = 1:6
    n = n_list(i);
    train_data =  data(1:n,:);
    train_labels = labels(1:n,:);
    weights = logistic_train(train_data, train_labels);
    preds = 1 ./ (1 + exp(-(test_data*weights)));
    preds(preds >= 0.5) = 1;
    preds(preds < 0.5) = 0;
    acc = sum(preds==test_labels) / size(test_data,1);
    acc_list(i) = acc;
end
acc_list
plot(n_list, acc_list)
xlabel('Size of training data')
ylabel('Accuracy')
title('Logistic regression validation accuracy vs training size')