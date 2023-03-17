function J  = nn_CostFunction_withot_bp(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;

%% Part 1 implementation

a1 = [ones(m, 1) X];

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1), 1) a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);
hThetaX = a3;

yVec = zeros(m,num_labels);

for i = 1:m
    yVec(i,y(i)) = 1;
end

% for i = 1:m
%     
%     term1 = -yVec(i,:) .* log(hThetaX(i,:));
%     term2 = (ones(1,num_labels) - yVec(i,:)) .* log(ones(1,num_labels) - hThetaX(i,:));
%     J = J + sum(term1 - term2);
%     
% end
% 
% J = J / m;

J = 1/m * sum(sum(-1 * yVec .* log(hThetaX)-(1-yVec) .* log(1-hThetaX)));

end