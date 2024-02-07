clear
clc
close all

% Parameters
Ph = 0.4;
gamma = 1;
max_coin = 100;
Reward = 1;


% State space
states = 0:max_coin;

% Actions
actions = 1:max_coin-1;

% Initialize value function and policy and reward
V = zeros(size(states));
V(1) = 0;
V(end) = 0;
policy = zeros(size(states));
best_action = zeros(size(states));
reward = zeros(size(states));
reward(max_coin+1) = Reward; % win
reward(1) = 0; % lose


% Value Iteration
tolerance = 1e-8;
max_iter = 1000;

iter = 0;
while true
    iter = iter + 1;
    
    % policy evaluation
    V_old = V;
    action_values = zeros(size(actions));
    best_action = zeros(size(states));
    for i = states % state
        if i == 0 || i == max_coin % win or loose
            continue
        end
        
        for a = i:-1:1 % available actions
            expected_return = 0;
            % win with probabilty of Ph
            state_win = min(i + a, max_coin);
            expected_return = expected_return + Ph * (reward(state_win+1) + gamma * V(state_win+1));
            % lose with probability of (1-Ph)
            state_lose = max(i - a, 0);
            expected_return = expected_return + (1-Ph) * (reward(state_lose+1) + gamma * V(state_lose+1));
            action_values(a) = expected_return;
        end
        [max_value, max_index] = max(action_values);
        V(i + 1) = max_value;
        best_action(i + 1) = actions(max_index);
    end

    % policy improvement
    policy = best_action;
    delta = max(abs(V - V_old));
    if delta < tolerance || iter == max_iter % exit the loop
        break
    end
end

plot(0:max_coin, policy);
xlabel('Capital');
ylabel('Stake');
title('Optimal Policy for the Gambler''s Problem with Unbalanced Coin');
xticks(0:5:max_coin);

figure
plot(V)


