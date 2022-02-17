
function done = get_data_multi_user(fc, pos_log, name, ENGINE, scenarios)
    if ENGINE == "octave"
        mkdir 'Data_sets/tmp';
    else
        rng shuffle;
        mkdir("Data_sets/tmp");
    end
    
    load(strcat("Data_sets/", pos_log));
    pos_bs = cell2mat(pos_log(1));
    pos_log = pos_log(2:end);
    
    % Get the length of the different matrices/vectors
    sb = size(pos_bs);          % Number of base stations
    sp = length(pos_log);       % Number of users
    sc = size(scenarios);
     
    l = qd_layout;                                          % Create new QuaDRIGa layout
    l.simpar.center_frequency = fc;                         % Set center frequency to 2 GHz
    l.simpar.use_absolute_delays = 1;                       % Enables true LOS delay
    l.simpar.show_progress_bars = 1;                        % Disable progress bars
    
    % Create the base stations
    l.no_tx = sb(2);
    for i = 1:sb(2)
        l.tx_position(:, i) = [pos_bs(1, i), pos_bs(2, i), 10]';     
    end
    
    % Create the users
    
    l.rx_track = qd_track();
    l.no_rx = sp;
    for i = 1:sp
        pos_log_mat = cell2mat(pos_log(i));       % pos_log is currently given as a cell and therefor needs to be converted
        l.rx_track(1, i).scenario = strtrim(scenarios(1, :));             % Set propagation scenario
    
        l.rx_track(1, i).positions=squeeze(pos_log_mat);   % Set start position and MT height
        l.rx_track(1, i).calc_orientation();
    
        l.rx_track(1, i).no_segments = l.rx_track(i).no_snapshots;       % Use spatial consisteny for mobility
                      
    end
    
    [c, b] = l.get_channels();                      % Generate channel coefficients
    c = merge( c, [], 0 );                          % Combine output channels
    
    sch = length(c);  % Get the number of created channels
    for i = 1:sch
        c(i).individual_delays = 0;                        % Remove per-antenna delays
    end
    
    % Number of snapshots
    N = c(1).no_snap;
    
    for i = 1:sp
        c_coeff = [];
        for j = 1:sch
            if mod(j - (i-1), sp) == 1
                c_coeff = cat(3, c_coeff, squeeze(c(j).coeff(1,1,:,:))');
            end
        end
    
    
        % Get the wanted dim: episodes, steps, multipaths
        tmp_b_AoA = cat(3, b(:).AoA);
        tmp_b_AoD = cat(3, b(:).AoD);
        b_AoA = permute(tmp_b_AoA(N*(i-1)+1:N*i, :, :), [3, 1, 2]);
        b_AoD = permute(tmp_b_AoD(N*(i-1)+1:N*i, :, :), [3, 1, 2]);
        c_coeff = permute(c_coeff, [3, 1, 2]);
    
        output = cell(4,1);
        output{1} = b_AoA;
        output{2} = b_AoD;
        output{3} = c_coeff;
        output{4} = l.rx_track(1, i).orientation;
    
        if ENGINE == "octave"
            save("-7", strcat("Data_sets/tmp/",name,"_",mat2str(i),".mat"), 'output');
        else
            save("Data_sets/tmp/"+name+"_"+string(i), 'output');
        end
    end
    
    AoA_cell = cell(sp, 1);
    AoD_cell = cell(sp, 1);
    Coeff_cell = cell(sp, 1);
    Ori_cell = cell(sp, 1);
    
    for episode = 1:sp
        load(strcat("Data_sets/tmp/",name,"_",mat2str(episode)));
    
        AoA_cell{episode} = output{1};
        AoD_cell{episode} = output{2};
        Coeff_cell{episode} = output{3};
        Ori_cell{episode} = output{4};
    end
    
    output = cell(4,1);
    output{1} = AoA_cell;
    output{2} = AoD_cell;
    output{3} = Coeff_cell;
    output{4} = Ori_cell;
    
    if ENGINE == "octave"
        save("-7", strcat("Data_sets/",name,".mat"), 'output');
    else
        save("Data_sets/" + name, 'output');
    end
    % rmdir("Data_sets/tmp", 's')
    
    done = 1;
    
end