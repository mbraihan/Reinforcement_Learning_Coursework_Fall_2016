function [v, pi, c_Rew] = qLearning(model, maxit, maxeps, epsilon, alpha)

% initialize the value function
Q = zeros(model.stateCount, 4);
pi = ones(model.stateCount, 1);

policy = ones(model.stateCount, 1);

c_Rew = zeros(length(maxeps), 1);

for i = 1:maxeps,
    % every time we reset the episode, start at the given startState
    s = model.startState;
%   a = 1;
    a = epsilon_greedy_policy(Q(s,:), epsilon);
    Rew = 0;

    for j = 1:maxit,
        % PICK AN ACTION
%         a = 1;
        p = 0;
        r = rand;

        for s_ = 1:model.stateCount,
            p = p + model.P(s, s_, a);
            if r <= p,
                break;
            end
        end

        % s_ should now be the next sampled state.
        % IMPLEMENT THE UPDATE RULE FOR Q HERE.
       
        R = model.R(s,a);
        
        Rew = Rew + model.gamma * R;
        
        a_ = epsilon_greedy_policy(Q(s,:), epsilon);
                
        Q(s,a) = Q(s,a) + alpha * ( R + model.gamma * max(Q(s_, :)) - Q(s,a) );    
        s = s_;
        a = a_;
               
        [~, idx] = max(Q(s,:));
        policy(s,:) = idx;
        Q1 = Q(:, idx);
                      
        % SHOULD WE BREAK OUT OF THE LOOP?              
         if s == model.goalState
             if R == model.R(model.goalState, a)
                break;
             end
         break;
         end

    end
    c_Rew(i) = Rew;
end

% REPLACE THESE
pi = policy;
v = Q1;

end


