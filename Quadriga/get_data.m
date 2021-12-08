
function done = get_data(fc, pos_log, name, ENGINE, scenarios)
    load(pos_log);
    
    l = qd_layout;                                          % Create new QuaDRIGa layout
    l.simpar.center_frequency = fc;                         % Set center frequency to 2 GHz
    l.simpar.use_absolute_delays = 1;                       % Enables true LOS delay
    l.simpar.show_progress_bars = 1;                        % Disable progress bars
    l.tx_position = [0, 0, 10]';                            % Set BS posittions
    
    l.rx_track = qd_track();

    sp = size(pos_log);  
    sc = size(scenarios);

    for episode = 1:sp(1)
        for chunck = 1:ceil(sp(3)/20000)
            if chunck*20000>sp(3)
                end_idx = sp(3);
            else
                end_idx = chunck*20000;
            end
            l.rx_track.positions=squeeze(pos_log(episode, :, (chunck-1)*20000 + 1:end_idx));   % Set start position and MT height
            sce_index = randi([1, sc(1)]);
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
                save("-7", name+"_"+string(episode)+"_"+string(chunk), 'output');
            else
                save(name+"_"+string(episode)+"_"+string(chunk), 'output');
            end

        end 
    end

    done = 1;
    
end