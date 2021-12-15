
function done = get_data(fc, pos_log, name, ENGINE, scenarios)
    rng shuffle
    
    load("Data_sets/"+pos_log);
    
    l = qd_layout;                                          % Create new QuaDRIGa layout
    l.simpar.center_frequency = fc;                         % Set center frequency to 2 GHz
    l.simpar.use_absolute_delays = 1;                       % Enables true LOS delay
    l.simpar.show_progress_bars = 1;                        % Disable progress bars
    l.tx_position = [0, 0, 10]';                            % Set BS posittions
    
    l.rx_track = qd_track();

    sp = size(pos_log);  
    sc = size(scenarios);

    mkdir("Data_sets\tmp")

    chunksize = 20000;
    for episode = 1:sp(1)
        sce_index = randi([1, sc(1)]);
        for chunk = 1:ceil(sp(3)/chunksize)
            if chunk*chunksize>sp(3)
                end_idx = sp(3);
            else
                end_idx = chunk*chunksize;
            end
            l.rx_track.positions=squeeze(pos_log(episode, :, (chunk-1)*chunksize + 1:end_idx));   % Set start position and MT height

            l.rx_track.scenario = strtrim(scenarios(sce_index, :));             % Set propagation scenario
            l.rx_track.calc_orientation();
    
            l.rx_track.no_segments = l.rx_track.no_snapshots;       % Use spatial consisteny for mobility
    
            b = l.init_builder;                                     % Initializes channel builder
    
            b.gen_parameters(0);                               % Clears LSF SSF and SOS parameters
            b.gen_parameters(5);                            % Generates all missing parameters
        
            c = get_channels( b );                          % Generate channel coefficients
            c = merge( c, [], 0 );                          % Combine output channels
            c.individual_delays = 0;                        % Remove per-antenna delays

            output = cell(4,1);
            output{1} = b.AoA;
            output{2} = b.AoD;
            output{3} = squeeze(c.coeff(1,1,:,:))';
            output{4} = l.rx_track.orientation();
        
            if ENGINE == "octave"
                save("-7", "Data_sets/tmp/"+name+"_"+string(episode)+"_"+string(chunk), 'output');
            else
                save("Data_sets/tmp/"+name+"_"+string(episode)+"_"+string(chunk), 'output');
            end

        end 
    end

    AoA_cell = cell(sp(1),1);
    AoD_cell = cell(sp(1),1);
    Coeff_cell = cell(sp(1),1);
    Ori_cell = cell(sp(1), 1);

    for episode = 1:sp(1)
        AoA = [];
        AoD = [];
        Coeff = [];
        Ori = [];
        for chunk = 1:ceil(sp(3)/chunksize)
            load("Data_sets/tmp/"+name+"_"+string(episode)+"_"+string(chunk));
            AoA = [AoA; output{1}];
            AoD = [AoD; output{2}];
            Coeff = [Coeff; output{3}];
            Ori = [Ori, output{4}];
        end
        AoA_cell{episode} = AoA;
        AoD_cell{episode} = AoD;
        Coeff_cell{episode} = Coeff;
        Ori_cell{episode} = Ori;
    end

    output = cell(4,1);
    output{1} = AoA_cell;
    output{2} = AoD_cell;
    output{3} = Coeff_cell;
    output{4} = Ori_cell;

    if ENGINE == "octave"
        save("-7", "Data_sets/" + name, 'output');
    else
        save("Data_sets/" + name, 'output');
    end
    rmdir("Data_sets/tmp", 's')

    done = 1;
    
end