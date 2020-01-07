clear all
clc

cliffworld;
episodes = 500;
iterInAllEpi = 500;
iterations = 150;


epsilon = 0.6;
alpha = [0.1, 0.2, 0.3, 0.4, 0.5 ];


Sarsa_A = zeros(episodes, size(alpha,2));
QLearning_A = zeros(episodes, size(alpha,2));

Cum_Sa = zeros(episodes, iterations);
Cum_Ql = zeros(episodes, iterations);


for i = 1:size(epsilon, 2)
    
    for j = 1:size(alpha,2)

        
        for iter = 1:iterations,
            cliffworld;
            [~, ~, Cum_Sa(:,iter)] = sarsa(model, iterInAllEpi, episodes, epsilon(i), alpha(j));
            [~, ~, Cum_Ql(:,iter)] = qLearning(model, iterInAllEpi, episodes, epsilon(i), alpha(j));
        end
        
        
        N_Cum_Sarsa = zeros(episodes,1);
        N_Cum_Q = zeros(episodes,1);
        
        
        
        for k = 1:episodes,
            N_Cum_Sarsa(k) = mean(Cum_Sa(k,:));
            N_Cum_Q(k) = mean(Cum_Ql(k,:));
        end
        
        
        
        Sarsa_A(: , j) = N_Cum_Sarsa';
        QLearning_A (:, j) = N_Cum_Q';
    
    end
    

    
end


figure(1)
plot([1:episodes], Sarsa_A(:,1), 'k' ,[1:episodes], Sarsa_A(:,2), 'r', [1:episodes], Sarsa_A(:,3), 'b', [1:episodes], Sarsa_A(:,4), 'm', [1:episodes], Sarsa_A(:,5), 'c')
legend('Alpha = 0.1', 'Alpha = 0.2', 'Alpha = 0.3', 'Alpha= 0.4', 'Alpha = 0.5' )
xlabel('Number of Episodes','fontweight','bold','fontsize',12)
ylabel('Cumulative Reward','fontweight','bold','fontsize',12)
title('SARSA with Different Step Sizes on Cliffworld, Epsilon = 0.6','fontweight','bold','fontsize',12);
axis([0 500 -2500 0])

figure(2)
plot([1:episodes], QLearning_A(:,1), 'k' ,[1:episodes], QLearning_A(:,2), 'r', [1:episodes], QLearning_A(:,3), 'b', [1:episodes], QLearning_A(:,4), 'm', [1:episodes], QLearning_A(:,5), 'c')
legend('Alpha = 0.1', 'Alpha = 0.2', 'Alpha = 0.3', 'Alpha= 0.4', 'Alpha = 0.5' )
xlabel('Number of Episodes','fontweight','bold','fontsize',12)
ylabel('Cumulative Reward','fontweight','bold','fontsize',12)
title('Q-Learning with Different Step Sizes on Cliffworld, Epsilon = 0.6','fontweight','bold','fontsize',12);
axis([0 500 -4500 0])

