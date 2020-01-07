function action = epsilon_greedy_policy(Q, epsilon)
    Total_a = [1 2 3 4];
%     e = 0.01;
    e_prob = rand();
    if e_prob < (1 - epsilon)
        [~, action] = max(Q);        
    else
        action = Total_a(randi(length(Total_a)));       
    end    
    
end

    

