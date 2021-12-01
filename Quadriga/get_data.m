
function done = get_data(fc, pos_log, name, ENGINE, scenarios)
    load(pos_log);
    l = qd_layout;                                          % Create new QuaDRIGa layout
    l.simpar.center_frequency = fc;                         % Set center frequency to 2 GHz
    l.simpar.use_absolute_delays = 1;                       % Enables true LOS delay
    l.simpar.show_progress_bars = 1;                        % Disable progress bars
    l.tx_position = [0, 0, 10]';                            % Set BS posittions
    
    l.rx_track = qd_track();

    sp = size(pos_log);
    AoA = cell(sp(1),1);
    AoD = cell(sp(1),1);
    Coeff = cell(sp(1),1);
    Ori = cell(sp(1), 1);
    
    sc = size(scenarios);

    for episode = 1:sp(1)
        l.rx_track.positions=squeeze(pos_log(episode, :, :));   % Set start position and MT height
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
        
        AoA{episode} = b.AoA;
        AoD{episode} = b.AoD;
        Coeff{episode} = squeeze(c.coeff(1,1,:,:))';
        Ori{episode} = l.rx_track.orientation();
        
    end
    
    output = cell(4,1);
    output{1} = AoA;
    output{2} = AoD;
    output{3} = Coeff;
    output{4} = Ori;

    
    if ENGINE == "octave"
        save("-7", name, 'output');
    else
        save(name, 'output');
    end

    done = 1;
    
end