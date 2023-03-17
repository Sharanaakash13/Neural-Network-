function theta = ini_weight(layer_out,layer_in)
    % Initializing the weights to small epsilon value 
    ini_epsilon = 0.12;  
    % Randam Initialization : Symmetry Breaking
    theta = rand(layer_out, 1+layer_in)*(2*ini_epsilon)-ini_epsilon;
end