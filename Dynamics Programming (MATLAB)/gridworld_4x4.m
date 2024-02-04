% Grid world 4x4
% Example 4.1 Book

clear
clc
close all


n_row = 4;
n_col = 4;
n_iter = 1000;

V = zeros(n_row, n_col);


% nu(1,1) = 0;
% nu(n_row,n_col) = 0;

disp(V)

R = -1;
gamma = 1;

actions = [0 -1; 0 1; -1, 0; 1, 0];


for it = 1:n_iter
    V_new = zeros(n_row, n_col);
    for i = 1:n_row
        for j = 1:n_col
            state = [i, j];
            if (i == 1 && j == 1) || (i == n_row && j == n_col)
                continue
            end
    
            for a = 1:size(actions,1)
                action = actions(a, :);
                next_state = state + action;
                r = R;
                if all(next_state >= 1) && all(next_state <= [n_row, n_col])
                    V_new(i,j) = V_new(i,j) + 0.25*(r + gamma*V(next_state(1),next_state(2)));
                else
                    V_new(i,j) = V_new(i,j) + 0.25*(r + gamma*V(i,j));
                end
    
    
            end
        end
    end
    V = V_new;
end
disp(it)
disp(V)

