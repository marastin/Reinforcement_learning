% Blackjack Problem using Monte Carlo ES (Exploring Starts)

clear
clc
close all
max_iter = 1000000;

states = zeros([10, 10, 2]); % Player Sum (11-21) - Dealer Showing (A-10) - Usable Ace (false:1 - true:2)
actions = [1, 2]; % Hit - Stick
Deck = [1:10, 10, 10, 10]; % numbers - jack - queen - king
Bust = 21;
Dealer_limit = 17;
gamma = 0.9; % discount factor

states_actions = zeros([size(states), numel(actions)]);
states_actions_visit_count = zeros([size(states), numel(actions)]);
% states_actions_average = zeros([size(states), numel(actions)]);
Q = zeros([size(states), numel(actions)]);
Policy = ones(size(states));
Returns = zeros(size(states_actions));
s_size = size(states_actions);
a_size = numel(actions);

iter = 0;
while iter < max_iter
    state = gen_random_state(s_size); % 11:21
    action = gen_random_action(a_size); % 1:2

    % Player Turn
    player_turn = true;
    player_bust = false;
    player_natural = false;
    player_usable_ace = (state(3) == 2);
    visited_state_action = {[state, action]};

    while player_turn
        if action == 1 % Hit
            new_card = get_new_card(Deck);
            state(1) = state(1) + new_card;
            if state(1) > Bust
                if player_usable_ace
                    state(1) = state(1) - 10;
                    state(3) = 1;
                    player_usable_ace = false;
                else
                    player_bust = true;
                    player_turn = false;
                end
            elseif state(1) == Bust
                player_natural = true;
                player_turn = false;
            end
        elseif action == 2 % Stick
            player_turn = false;
        end
        if (player_bust == false) && (player_natural == false)
            visited_state_action{end+1} = [state, action]; % Change in size happens a few time until the end of episode
        end
        if player_turn
            action = get_policy(state, Q);
        end
    end

    % Dealer Turn
    dealer_turn = true;
    dealer_bust = false;
    dealer_sum = state(2); % The shown card add to sum. later the other cards will be added
    dealer_usable_ace = (state(2) == 1); % Dealer shown card is Ace so that it can be considered as usable
    while dealer_turn
        new_card = get_new_card(Deck);
        dealer_sum = dealer_sum + new_card; % Other cards added to sum here
        if dealer_sum >= Dealer_limit % Dealer Stick
            dealer_turn = false;
        end
        if dealer_sum > Bust % Dealer Bust
            if dealer_usable_ace
                dealer_sum = dealer_sum - 10;
                dealer_usable_ace = false;
            else
                dealer_bust = true;
                dealer_turn = false;
            end
        end
    end

    % Result
    if player_bust % Dealer wins
        reward = -1;
    elseif dealer_bust % Player wins
        reward = 1;
    elseif state(1) > dealer_sum % Player wins
        reward = 1;
    elseif state(1) < dealer_sum % Dealer wins
        reward = -1;
    elseif state(1) == dealer_sum % Draw
        reward = 0;
    end
    
    G = reward;
    for step = numel(visited_state_action):-1:1
        current_visit = visited_state_action{step};
        linear_idx = sub2ind(size(states_actions), current_visit(1) - 10, current_visit(2), current_visit(3), current_visit(4));
        old_ave = Q(linear_idx);
        n = states_actions_visit_count(linear_idx);
        new_ave = (n*old_ave + G)/(n+1);
        G = gamma*G;
        Q(linear_idx) = new_ave;
        states_actions_visit_count(linear_idx) = states_actions_visit_count(linear_idx) + 1;
    end

    iter = iter + 1;
end


[V, greedy_actions] = max(Q, [], 4);

% Plotting
figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]); % Maximize the figure
defaultColormap = colormap;

subplot(2,2,1)
imagesc(greedy_actions(:, :, 1));
title('Action - No Usable Ace')
xlabel('Dealer Shown')
ylabel('Player Sum')
xticks(1:10);
yticks(1:10);
yticklabels(11:20);
axis equal;
colormap(gca, [1 1 0; 0 0 1]) % Yellow for values <= 1.5, Blue for values > 1.5
c = colorbar;
c.Ticks = [1 2]; % Set tick locations
c.TickLabels = {'Hit', 'Stick'}; % Set tick labels
c.Limits = [0.5 2.5];



subplot(2,2,2)
imagesc(greedy_actions(:, :, 2));
title('Action - Usable Ace')
xlabel('Dealer Shown')
ylabel('Player Sum')
xticks(1:10);
yticks(1:10);
yticklabels(11:20);
axis equal;
colormap(gca, [1 1 0; 0 0 1]) % Yellow for values <= 1.5, Blue for values > 1.5
c = colorbar;
c.Ticks = [1 2]; % Set tick locations
c.TickLabels = {'Hit', 'Stick'}; % Set tick labels
c.Limits = [0.5 2.5];

subplot(2,2,3)
[X, Y] = meshgrid(1:10, 1:10);
surf(X, Y, V(:, :, 1));
title('No Usable Ace')
xlabel('Dealer Shown')
ylabel('Player Sum')
zlabel('State Value')
xticks(1:10);
yticks(1:10);
yticklabels(11:20);

subplot(2,2,4)
[X, Y] = meshgrid(1:10, 1:10);
surf(X, Y, V(:, :, 2));
title('Usable Ace')
xlabel('Dealer Shown')
ylabel('Player Sum')
zlabel('State Value')
xticks(1:10);
yticks(1:10);
yticklabels(11:20);


function state = gen_random_state(states_size)
    state = [randi([1, states_size(1)]) + 10, randi([1, states_size(2)]), randi([1, states_size(3)])];
end

function action = gen_random_action(actions_size)
    action = randi([1, actions_size]);
end

function action = get_policy(state, Q)
    if Q(state(1) - 10, state(2), state(3), 1) > Q(state(1) - 10, state(2), state(3), 2)
        action = 1;
    elseif Q(state(1) - 10, state(2), state(3), 1) < Q(state(1) - 10, state(2), state(3), 2)
        action = 2;
    else
        state_action_size = size(Q);
        action = gen_random_action(state_action_size(end));
    end
end

function new_card = get_new_card(Deck)
    new_card = Deck(randi([1,numel(Deck)]));
end


