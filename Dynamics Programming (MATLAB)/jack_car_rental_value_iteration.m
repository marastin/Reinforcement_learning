% Value iteration - Jack's car rental
clear
clc
close all

% Parameters
lambda_request_location1 = 3;
lambda_request_location2 = 4;
lambda_return_location1 = 3;
lambda_return_location2 = 2;
gamma = 0.9;
max_cars = 20;
max_moves = 5;
rental_reward = 10;
move_cost = 2;

% Precalculation
poisspdf_request_location1 = poisspdf(0:max_cars, lambda_request_location1);
poisspdf_request_location2 = poisspdf(0:max_cars, lambda_request_location2);
poisspdf_return_location1 = poisspdf(0:max_cars, lambda_return_location1);
poisspdf_return_location2 = poisspdf(0:max_cars, lambda_return_location2);


% State space
states = zeros(max_cars + 1, max_cars + 1);

% Actions (net cars moved)
actions = -max_moves:max_moves;

% Initialize value function and policy
V = zeros(max_cars + 1, max_cars + 1);
policy = zeros(max_cars + 1, max_cars + 1);

% Value Iteration
tolerance = 1e-2;
iter = 0;
while true
    iter = iter + 1;
    disp(['iter: ' num2str(iter)])
    delta = 0;
    for i = 0:max_cars
        for j = 0:max_cars
            v = V(i + 1, j + 1);
            action_values = zeros(size(actions));
            
            for a = 1:length(actions)
                action = actions(a);
                next_state1 = min(i - action, max_cars);
                next_state2 = min(j + action, max_cars);
                
                % Calculate expected returns
                expected_return = 0;
                for request_location1 = 0:max_cars
                    for request_location2 = 0:max_cars
                        for return_location1 = 0:max_cars
                            for return_location2 = 0:max_cars
                                prob = poisspdf_request_location1(request_location1 + 1) * ...
                                    poisspdf_request_location2(request_location2 + 1) * ...
                                    poisspdf_return_location1(return_location1 + 1) * ...
                                    poisspdf_return_location2(return_location2 + 1);
                                
                                rented1 = min(request_location1, next_state1);
                                rented2 = min(request_location2, next_state2);
                                reward = (rented1 + rented2) * rental_reward;
                                
                                next_state1_tmp = min(next_state1 - rented1 + return_location1, max_cars);
                                next_state2_tmp = min(next_state2 - rented2 + return_location2, max_cars);
                                
                                expected_return = expected_return + prob * (reward + gamma * V(next_state1_tmp + 1, next_state2_tmp + 1));
                            end
                        end
                    end
                end
                
                action_values(a) = expected_return - move_cost * abs(action);
            end
            
            [max_value, max_index] = max(action_values);
            
            % Update value function and policy
            V(i + 1, j + 1) = max_value;
            policy(i + 1, j + 1) = actions(max_index);
            
            delta = max(delta, abs(v - V(i + 1, j + 1)));
        end
    end
    disp(['delta = ' num2str(delta)])
    if delta < tolerance
        break;
    end
    disp(['Policy (iter ' num2str(iter) ':']);
    disp(policy);
end

disp('Optimal Policy (Net Cars Moved):');
disp(policy);
C = contour(policy);
clabel(C, -5:5);
