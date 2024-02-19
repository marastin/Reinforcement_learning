clear
clc
close all


%% Track design
Track = zeros([32, 17]);
Track(1,4:9) = 2; % start line
Track(2:3,4:9) = 1;
Track(4:10,3:9) = 1;
Track(11:18,2:9) = 1;
Track(19:25,1:9) = 1;
Track(26,1:10) = 1;
Track(27:28,1:17) = 1;
Track(29,2:17) = 1;
Track(30:31,3:17) = 1;
Track(32,4:17) = 1;
Track(27:32,17) = 3; % finish line

% Globals
Startline = sub2ind(size(Track), find(Track == 2));
Track_Size = size(Track);

% % Track Plotting
% figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]); % Maximize the figure
% Track_turned = flipud(Track);
% imagesc(Track);
% title('Track')
% xlabel('Horizontal')
% ylabel('Vertical')
% xticks(1:17);
% yticks(1:32);
% axis equal;
% colormap(gca, [1 1 0; 0 0 1; 1 0 0; 0 1 0])
% c = colorbar;
% c.Ticks = [0, 1, 2, 3]; % Set Color tick locations
% c.Limits = [0, 3]; % Color limits

n_rows = size(Track, 1);
n_cols = size(Track, 2);

%% Initialization
all_actions = {[1, -1], [1, 0], [1, 1], [0, -1], [0, 0], [0, 1], [-1, -1], [-1, 0], [-1, 1]}; % [dVr, dVc] 
n_actions = numel(all_actions);
vels = 0:5;
n_vels = numel(vels);
n_episodes = 100;
epsilon = 0.8;
Q = rand(n_rows, n_cols, n_vels, n_vels, n_actions);
C = zeros(n_rows, n_cols, n_vels, n_vels, n_actions);
[~, Policy] = max(Q, [], 5);

%% Main
for i = 1:n_episodes
    disp(i)
    state = move_to_start(Startline, Track_Size);
    is_finished = false;
    while ~is_finished

        if rand() > epsilon
            b = Policy(state(1), state(2), state(3), state(4));
        else
            posible_actions = get_possible_actions(state, all_actions);
            indices_of_posible_actions = find(posible_actions == 1);
            b = datasample(indices_of_posible_actions, 1);
        end
        action = all_actions{b};
        state = update_state(state, action, Track, Startline, Track_Size);        
        
        if check_finished(state, Track)
            is_finished = true;
            break
        end
        if ~check_in_track(state, Track)
            state = move_to_start(Startline, Track_Size);
        end
    end
end


%% Functions
function possible_actions_indices = get_possible_actions(state, all_actions)
    vc = state(4);
    vr = state(3);
    possible_actions_indices = ones(size(all_actions));
    for i = 1:numel(all_actions)
        action = all_actions{i};
        if any(([vr, vc] + action) > 5)
            possible_actions_indices(i) = 0;
            continue
        end
        
        if any(([vr, vc] + action) < 1)
            possible_actions_indices(i) = 0;
            continue
        end

        if any(([vr, vc] + action) == [1, 1])
            possible_actions_indices(i) = 0;
            continue
        end
    end
end

function new_state = update_state(state, action, Track, Startline, Track_Size) % state: r, c, vr, vc
    r = state(1);
    c = state(2);
    vr = state(3) - 1;
    vc = state(4) - 1;
    new_state = [r+vr, c+vc, vr+action(1)+1, vc+action(2)+1];

    if ~check_in_track(new_state, Track)
        new_state = move_to_start(Startline, Track_Size);
    end
end

function new_state = move_to_start(Startline, Track_Size)
    ind = Startline(randi(numel(Startline)));
    [x, y] = ind2sub(Track_Size, ind);
    new_state = [x, y, 1, 1]; % velocity must set to zeros 

end

function in_track = check_in_track(state, Track)
    if Track(state(1), state(2)) > 0
        in_track = true;
    else
        in_track = false;
    end
end

function finished = check_finished(state, Track)
    if Track(state(1), state(2)) == 3
        finished = true;
    else
        finished = false;
    end
end


