% Grid world 4x4
% Policy improvement

clear
clc
close all


n_row = 4;
n_col = 4;
n_iter = 10; % number of total policy improvement  

V = zeros(n_row, n_col);

R = -1; % for every action
gamma = 0.9;

actions = [0 -1; 0 1; -1, 0; 1, 0];   % Left, Right, Up, Down
actions_rows = [1, 2, 3, 4];

initial_policy = randi(length(actions_rows), n_row, n_col);
initial_policy(1,1) = 0;
initial_policy(n_row, n_col) = 0;
policy = initial_policy;

disp('Initial Policy: ')
disp(policy)

for it = 1:n_iter

    delta = 0;
    theta = 0.001;
    
    % policy evaluation
    while 1
        for i = 1:n_row
            for j = 1:n_col
                G = 0; % expected return
                state = [i, j];
                if (i == 1 && j == 1) || (i == n_row && j == n_col)
                    continue
                end

                action = actions(policy(i,j),:);
                next_state = state + action;
                r = R;
                if all(next_state >= 1) && all(next_state <= [n_row, n_col])
                    G = G + 1*(r + gamma*V(next_state(1), next_state(2)));
                else
                    G = G + 1*(r + gamma*V(i,j));
                end
                delta = max(0, abs(V(i,j) - G));
                V(i,j) = G;
            end
        end

        if delta < theta
            break
        end
    end
    
    old_policy = policy;
    for i = 1:n_row
        for j = 1:n_col
            state = [i, j];
            if (i == 1 && j == 1) || (i == n_row && j == n_col)
                continue
            end
            q = zeros(4,1);
            for a = 1:size(actions)
                action = actions(a, :);
                next_state = state + action;
                
                if all(next_state >= 1) && all(next_state <= [n_row, n_col])
                    q(a) = -1 + gamma*V(next_state(1), next_state(2));
                else
                    q(a) = -1 + gamma*V(i, j);
                end
            end
            maxValue = max(q);
            policy(i,j) = find(q == maxValue, 1);
        end
    end

    disp(['State Values after iteration ' num2str(it) ' : ']);
    disp(V)
    disp(['Policy after iteration ' num2str(it) ' : ']);
    disp(policy)
    
    if all(old_policy == policy)
        break
    end
    
end


disp('Stable Policy:');
disp(policy)
disp('1-Left, 2-Right, 3-Up, 4-Down')