% Grid world 4x4
% Example 4.1 Book

clear
clc
close all


n_row = 4;
n_col = 4;
n_iter = 1000;

V = zeros(n_row, n_col);

R = -1; % for every action
gamma = 1;

actions = [0 -1; 0 1; -1, 0; 1, 0];   % Left, Right, Up, Down


for it = 1:n_iter
    for i = 1:n_row
        for j = 1:n_col
            G = 0; % expected return
            state = [i, j];
            if (i == 1 && j == 1) || (i == n_row && j == n_col)
                continue
            end
    
            for a = 1:size(actions,1)
                action = actions(a, :);
                next_state = state + action;
                
                r = R;

                if all(next_state >= 1) && all(next_state <= [n_row, n_col])
                    G = G + 0.25*(r + gamma*V(next_state(1), next_state(2)));
                else
                    G = G + 0.25*(r + gamma*V(i,j));
                end
    
            end
            V(i,j) = G;
        end
    end
    
end
disp('State Values after Iterative Policy Evaluation:');
disp(V)

