function [v, pi] = valueIteration(model, maxit)

% initialize the value function
v = zeros(model.stateCount, 1);
v1 = zeros(model.stateCount, 1);
    pi = ones(model.stateCount, 1);
epsilon = 1.0000e-22;

for i = 1:maxit,
    % initialize the policy and the new value function
    policy = ones(model.stateCount, 1);
    v_ = zeros(model.stateCount, 1);

    % perform the Bellman update for each state
    for s = 1:model.stateCount,
        % COMPUTE THE VALUE FUNCTION AND POLICY
        % YOU CAN ALSO COMPUTE THE POLICY ONLY AT THE END
        P = reshape(model.P(s,:,:), model.stateCount, 4);          
        [v_(s,:), action] = max(model.R(s,:) + (model.gamma * P' * v)');
        policy(s,:) = action; 
    end
    
    
    v1 = v;
    v = v_;
    pi = policy;
    
    % exit early
    if v - v1 <= epsilon
        % CHANGE THE IF-STATEMENT
        fprintf('Function converged after %d episode\n',i);
        break;
    end      
end

