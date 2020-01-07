function [v, pi] = policyIteration(model, maxit)

% initialize the value function
v = zeros(model.stateCount, 1);
epsilon = 0.001;

for i = 1:maxit,
    % initialize the policy and the new value function
    pi = ones(model.stateCount, 1);
    v_ = zeros(model.stateCount, 1);
    
    for s = 1:model.stateCount,
        TranProb = reshape(model.P(s,:,:), model.stateCount, 4);
        [~, action] = max(model.R(s,:) + (model.gamma * TranProb' * v)');
        v_(s,:) = model.R(s, action) + (model.gamma * TranProb(:, action)' * v)' ; 
        pi(s,:) = action;
    end
    
    v1 = v; 
    v = v_; 
       
    if v - v1 <= epsilon
        fprintf('Function converged after %d episode\n',i);
        break;
    end   

end

