function [X_norm, mu, sigma] = normalize(x)
    n = size(x,2);      % No. of variables
    mu = ones(1,size(x,2)); 
    sigma = ones(1,size(x,2));
    X_norm = x;
    for i = 1:n         
        mu(i) = mean(x(:,i));    % average value of x
        sigma(i) = std(x(:,i));  % standard deviation (sigma)
        X_norm(:,i) = (x(:,i) - mu(i)) / sigma(i); % normalize x
    end    
end
