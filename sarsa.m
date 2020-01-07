function [v, pi, c_Rew] = sarsa(model, maxit, maxeps, epsilon, alpha)

% initialize the value function
Q = zeros(model.stateCount, 4);  
pi = ones(model.stateCount, 1);  

policy = ones(model.stateCount, 1);

c_Rew = zeros(length(maxeps), 1);

for i = 1:maxeps,
    %every time we reset the episode, start at the given startState   
    s = model.startState;
    
    %a = 1;
    a = epsilon_greedy_policy(Q(s,:), epsilon);
    
    Rwd = 0;
   
    for j = 1:maxit,            
        % PICK AN ACTION
        %%a = 1;
        
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
        
        reward = model.R(s,a); 
        
        Rwd = Rwd + reward;
        
        a_ = epsilon_greedy_policy(Q(s_,:), epsilon);
        
        Q(s,a) = Q(s,a) + alpha * [reward + model.gamma * Q(s_, a_) - Q(s,a)];
        
        s = s_;
        a = a_;

        [~, idx] = max(Q(s,:));
        policy(s) = idx;
        q = Q(:, idx);
        
         % SHOULD WE BREAK OUT OF THE LOOP?
         if s == model.goalState
%              if R == model.R(model.goalState, a)
%                 break;
%              end
            break;
         end
            
             
    end   
    c_Rew (i)  = Rwd;
end

pi = policy;
v = q;

c_Rew = c_Rew;

end


