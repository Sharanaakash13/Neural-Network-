%% 
% Big Data and small Data assignment 1
% Question 2
% Neural Network

%% Preparing workspace
clc; clear; close all;

%% Loading dataset
% open .csv file
fprintf('Loading and Visualizing Data ...\n')
table = readtable('breast_cancer_HTE1.csv','VariableNamingRule','preserve');
summary(table)
% Creating dataset matrix and replacing strings with labels
data = table{:,3:32};                            % numerical features
[data(:,31),label] = grp2idx(table{:,2});        % categorical 

%%  Checking and removing missing values
fprintf('Checking for missing values...\n\n')
[data,TF] = rmmissing(data);
ind = find(TF);         % Index of the outliers
s = size(ind,1);        % No. of outliers
if s == 0
    fprintf('No missing  values found\n')
else
    fprintf('Number of missing values: %d\n',s)
end

[m,n] = size(data);     % m = height of dataset; n = width of dataset

% Assigning values to X and y
X = data(:,1:30);
y = data(:,n);

%% Visualization
fprintf('Visualizing the data.....\n 1. Bar Chart\n 2. Box plot\n 3.Correlation Plot\n')
% =================== Bar chart ====================
count = (countcats(categorical(table{:,2})))';
labels_name ={'B'; 'M'};
sgtitle('Benign vs Malignant')
b = barh(count,'FaceColor','flat');
b.CData(2,:)=[0.8500 0.3250 0.0980];
set(gca,'yticklabel',labels_name)
fprintf('Number of Benign : %d\n', count(:,1))
fprintf('Number of Malignant : %d\n', count(:,2))

% =================== Box Plot ====================
figure('Name','Visualization','Numbertitle','off')
boxplot(X)
j=(3:12);
xlabel('Input Variables')

% % =================== Correlation ====================
figure('Name','Correlation')
R = corrplot(X(:,1:10),'testR','on');
display(R)

%% Feature Extraction 
% % Due to Multicollinearity of variables
fprintf('Removing some features due to multicolinearity...\n')
X_new = X(:,[1 2 5 6 9 10 11 12 13 15 16 19 20 21 22 25 26 29 30]);
%X_new = X(:,[1 2 5 8 9 10 11 12 15 17 18 19 20 21 22 25 26 28 29 30]);

%% Normalizing 
fprintf('Normalizing the features ...\n')
[X_norm, mu, sigma] = normalize(X_new);

%%  Partitioning the dataset
fprintf('Randomizing and dividing the dataset into\n') 
fprintf('1.Training set - 60 percentage\n 2.Cross Validation set - 20 percentage\n 3.Test set - 20 percentage\n\n')

rng(2) % for reproducibility of random number

% Dividerand : Randomize and divide the data into train 70%, test30% 
[idx_train,~,idx_test] = dividerand(height(X_norm),0.7,0,0.3);
train = idx_train'; % random index for train
test = idx_test';   % random index for test

% Training dataset as 70 percent
X_train = X_norm(train,:);
Y_train = y(train,:);

% Test dataset as 30 percent 
X_test = X_norm(test,:);
Y_test = y(test,:);

%% Initializing size of Neural Network
fprintf('Initializing the size of the NN layers...\n')
input_layer_size = size(X_train,2);           % number of units in input layer
hidden_layer_size =1;                         % number of units in hidden layer 
output_layer_size =size(label,1);             % number of units in output layer 
epochs = 100;                                 % number of epochs

%% Initializing weights
% Initializing the weighting parameter 
fprintf('Initializing and randomizing weights ...\n')
% Initial Theta 1
init_theta1 = ini_weight(hidden_layer_size,input_layer_size);
% Initial Theta 2
init_theta2 = ini_weight(output_layer_size,hidden_layer_size);

fprintf('Initializing weighting parameters as a vector ...\n')
% Converting into single vector
nn_params = [init_theta1(:); init_theta2(:)];

%% Training NN  and compute cost (Train set)

fprintf('\nTraining Neural Network... \n')

% Setting number of iteration
options = optimset('MaxIter', epochs);

% Regularization parameter lambda
lambda = 0;

% Create "short hand" for the cost function to be minimized
costFunction_train = @(p) nn_CostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   output_layer_size, X_train, Y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[train_nn_params, train_cost] = fmincg(costFunction_train, nn_params, options); 

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(train_nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(train_nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 output_layer_size, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Predicting training set
fprintf('Predicting training set accuracy...')
pred_val = predict(Theta1, Theta2, X_train);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_val == Y_train)) * 100);

%% Predicting test set

pred_val = predict(Theta1, Theta2, X_test);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_val == Y_test)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% To improve the predictiction performance 
%% Repartitioning
rng(2) % for reproducibility of random number

% Dividerand : Randomize and divide the data into train, val, test 
[idx_train,idx_val,idx_test] = dividerand(height(X_norm),0.6,0.2,0.2);
train = idx_train'; % random index for train
val = idx_val';
test = idx_test';   % random index for test

% Train dataset as 60 percent
X_train = X_norm(train,:);
Y_train = y(train,:);

% Validation set dataset as 20 percent
X_val = X_norm(val,:);
Y_val = y(val,:);

% Test dataset as 20 percent 
X_test = X_norm(test,:);
Y_test = y(test,:);
%% Training using varying lambda for validation set
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', epochs);

% initializing matrices
lambda_new = (0:0.01:1);
every_weight= zeros(input_layer_size+5,length(lambda_new));
cost_train = zeros(length(lambda_new),1);
cost_val = zeros(length(lambda_new),1);

% Training NN
for i =1:length(lambda_new)
fprintf('\n For Lambda: %f\n', lambda_new(i));

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nn_CostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   output_layer_size, X_train,Y_train, lambda_new(i));

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params_lamb, cost] = fmincg(costFunction, nn_params, options);
every_weight(:,i)=nn_params_lamb;
             
% ============ Computing cost for the training set ===============             
fprintf('\nComputing cost of training set ...\n')

% Weight regularization parameter (we set this to 0 here).
J_train = nn_CostFunction_withot_bp(nn_params_lamb, input_layer_size, hidden_layer_size, ...
                   output_layer_size, X_train, Y_train);
cost_train(i,:) = J_train;      % storing the cost value       

% ============ Computing cost for the training set ===============             
fprintf('\nComputing cost of cross validation set ...\n')

% Weight regularization parameter (we set this to 0 here).
J_val = nn_CostFunction_withot_bp(nn_params_lamb, input_layer_size, hidden_layer_size, ...
                   output_layer_size, X_val, Y_val);
cost_val(i,:) = J_val;      % storing the cost value       
end

% Plotting lambda vs cost function 
fprintf('\nPlotting Lambda vs Cost function...\n')
figure('Name','Choosing the regularization parameter','Numbertitle','off')
title('Cost based on regularization')
scatter(lambda_new,cost_train,'b+')
hold on;
scatter(lambda_new,cost_val,'r+')
ylim auto
xlabel('Lambda')
ylabel('Error')
grid on;hold off;
legend('J train','J cv')

% Choosing lambda with lowest CV cost function
[~,I]= min(cost_val);
nn_params_lamb_opt = every_weight(:,I);
lambda_opt = lambda_new(I);
fprintf('The lambda is choosen as: %f\n\n',lambda_opt)

% Obtain Theta1 and Theta2 back from nn_params
Theta1_opt = reshape(nn_params_lamb_opt(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2_opt = reshape(nn_params_lamb_opt((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 output_layer_size, (hidden_layer_size + 1));


%% Predicting training set (regularization added)

pred_train = predict(Theta1_opt, Theta2_opt, X_train);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_train== Y_train)) * 100);

%% Predicting cross validation set (regularization added)

pred_val = predict(Theta1_opt, Theta2_opt, X_val);

fprintf('\nCross Validation Set Accuracy: %f\n', mean(double(pred_val == Y_val)) * 100);

%% Predicting test set (regularization added)

pred_test = predict(Theta1_opt, Theta2_opt, X_test);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == Y_test)) * 100);

             
%% Accuracy of training data

%Prediction accuracy
% prediction = predict(Theta1,Theta2,X); %Contains prediction of y for all m
% Accuracy_train = mean(double(prediction == y))*100;
% Accuracy_train_Change(ctr,1) = Accuracy_train(end);
% fprintf('\n Training Accuracy: %f\n',Accuracy_train);
% 
%% Accuracy of Validation Data
% validationData = validationData(randperm(size(validationData,1)),:); %Random shuffling
% X_validation = validationData(:,1:end-1);
% y_validation = validationData(:,end);
% 
% %Prediction accuracy
% prediction = predict(Theta1,Theta2,X_validation); %Contains prediction of y for all m
% Accuracy_validation = mean(double(prediction == y_validation))*100;
% Accuracy_validation_Change(ctr,1) = Accuracy_validation(end);
% 
% fprintf('\n Validation Accuracy: %f\n',Accuracy_validation);
% %% Accuracy of testing data
% testData = testData(randperm(size(testData,1)),:); %Random shuffling
% X_test = testData(:,1:end-1);
% y_test = testData(:,end);
% 
% %Prediction accuracy
% prediction = predict(Theta1,Theta2,X_test); %Contains prediction of y for all m
% Accuracy_test = mean(double(prediction == y_test))*100;
% Accuracy_test_Change(ctr,1) = Accuracy_test(end);
% 
% fprintf('\n Test Accuracy: %f\n',Accuracy_test);