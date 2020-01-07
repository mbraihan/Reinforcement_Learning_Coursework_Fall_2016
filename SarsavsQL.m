clear all
clc

cliffworld;
episodes = 500;
iterInAllEpi = 500;
iterations = 150;
% epsilon = 0.2;
% alpha = 0.2;


Cum_Sarsa = zeros(episodes, iterations);
Cum_Q = zeros(episodes, iterations);


for iter = 1:iterations,
    [~, ~, Cum_Sarsa(:,iter)] = sarsa1(model, iterInAllEpi, episodes);
    [~, ~, Cum_Q(:,iter)] = qLearning(model, iterInAllEpi, episodes);
end

N_Cum_Sarsa = zeros(episodes,1);
N_Cum_q = zeros(episodes,1);
for i = 1:episodes,
    N_Cum_Sarsa(i) = mean(Cum_Sarsa(i,:));
    N_Cum_q(i) = mean(Cum_Q(i,:));
end

plot([1:episodes], N_Cum_Sarsa')
hold on
plot([1:episodes], N_Cum_q','r')



figure(1)
plot([1:episodes], N_Cum_Sarsa', 'k' ,[1:episodes], N_Cum_q', 'r')
legend('Sarsa', 'Q-Learning')
xlabel('Number of Episodes','fontweight','bold','fontsize',12)
ylabel('Cumulative Reward','fontweight','bold','fontsize',12)
title('Q-Learning and SARSA on Cliff World MDP','fontweight','bold','fontsize',12);


