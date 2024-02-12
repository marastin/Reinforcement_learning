% Blackjack Problem using Monte Carlo ES (Exploring Starts)

clear
clc
close all

states = zeros([10, 10, 2]); % Player Sum (11-21) - Dealer Showing (A-10) - Usable Ace (Yes-No)
actions = [1, 2]; % Hit - Stick
states_actions = zeros([size(states), numel(actions)]);
s_size = size(states_actions);
a_size = numel(actions);
Deck = [1:10, 10, 10, 10]; % numbers - jack - queen - king
Bust = 21;
Dealer_limit = 17;
Q = zeros(size(states));
Policy = ones(size(states));
Returns = zeros(size(states_actions));

ind = 0;
while ind < 10
    state = gen_random_state(s_size);
    player_bust = false;
    dealer_bust = false;
    visited_state_action = {};
    
    % Player Turn
    player_turn = true;
    while player_turn
        action = gen_random_action(a_size);
        visited_state_action{end+1} = [state, action];
        if action == 1 % Hit
            new_card = get_new_card(Deck);
            state(1) = state(1) + new_card;
            if state(1) > Bust
                player_turn = false;
                player_bust = true;
            end
        elseif action == 2 % Stick
            player_turn = false;
        end
    end

    % Dealer Turn
    dealer_turn = true;
    dealer_sum = state(2);
    while dealer_turn
        new_card = get_new_card(Deck);
        dealer_sum = dealer_sum + new_card;
        if dealer_sum > Dealer_limit % Dealer Stick
            dealer_turn = false;
        end
        if dealer_sum > Bust % Dealer Bust
            dealer_bust = true;
            dealer_turn = false;
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


    ind = ind + 1;
end


function state = gen_random_state(states_size)
    state = [randi([1, states_size(1)]), randi([1, states_size(2)]), randi([1, states_size(3)])];
end

function action = gen_random_action(actions_size)
    action = randi([1, actions_size]);
end

function new_card = get_new_card(Deck)
    new_card = Deck(randi([1,numel(Deck)]));
end
