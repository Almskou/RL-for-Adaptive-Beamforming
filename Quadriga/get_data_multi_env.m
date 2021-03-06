
function done = get_data_multi_env(fc, pos_log, name, ENGINE, scenarios)
    if ENGINE == "octave"
        mkdir 'Data_sets/tmp';
    else
        rng shuffle;
        mkdir("Data_sets/tmp");
    end
    
    load(strcat("Data_sets/", pos_log));
    pos_bs = cell2mat(pos_log(1));
    pos_log = pos_log(2:end);
    
    sb = size(pos_bs);
     
    l = qd_layout;                                          % Create new QuaDRIGa layout
    l.simpar.center_frequency = fc;                         % Set center frequency to 2 GHz
    l.simpar.use_absolute_delays = 1;                       % Enables true LOS delay
    l.simpar.show_progress_bars = 1;                        % Disable progress bars
    
    l.no_tx = sb(2);
    for i = 1:sb(2)
        l.tx_position(:, i) = [pos_bs(1, i), pos_bs(2, i), 10]';                            % Set BS posittions
    end

    l.rx_track = qd_track();
    
    sp = length(pos_log);  
    sc = size(scenarios);

    for episode = 1:sp
        pos_log_mat = cell2mat(pos_log(episode));       % pos_log is currently given as a cell and therefor needs to be converted
    
        sce_index = randi([1, sc(1)]);
        l.rx_track.scenario = strtrim(scenarios(sce_index, :));             % Set propagation scenario
        l.rx_track.positions=squeeze(pos_log_mat);   % Set start position and MT height

        
        l.rx_track.calc_orientation();

        l.rx_track.no_segments = l.rx_track.no_snapshots;       % Use spatial consisteny for mobility
                      
        [c, b] = l.get_channels();                      % Generate channel coefficients
        c = merge( c, [], 0 );                          % Combine output channels
        c_coeff = [];

        for i = 1:sb(2)
            c(i).individual_delays = 0;                        % Remove per-antenna delays
            c_coeff = cat(3, c_coeff, squeeze(c(i).coeff(1,1,:,:))');
        end

        % Get the wanted dim: episodes, steps, multipaths
        b_AoA = permute(cat(3, b(:).AoA), [3, 1, 2]);
        b_AoD = permute(cat(3, b(:).AoD), [3, 1, 2]);
        c_coeff = permute(c_coeff, [3, 1, 2]);

        output = cell(4,1);
        output{1} = b_AoA;
        output{2} = b_AoD;
        output{3} = c_coeff;
        output{4} = l.rx_track.orientation;
    
        if ENGINE == "octave"
            save("-7", strcat("Data_sets/tmp/",name,"_",mat2str(episode),".mat"), 'output');
        else
            save("Data_sets/tmp/"+name+"_"+string(episode), 'output');
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