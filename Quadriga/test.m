
fc = 28e9;
l = qd_layout;                                          % Create new QuaDRIGa layout
l.simpar.center_frequency = fc;                        % Set center frequency to 2 GHz
l.simpar.use_absolute_delays = 1;                       % Enables true LOS delay
l.simpar.show_progress_bars = 1;                        % Disable progress bars
l.tx_position = [0, 0, 10]';                            % Set BS posittions

l.rx_track = qd_track( 'circular' , 100, -pi/2);           % 50 m long track going north
l.rx_track.initial_position = [0 ; -15 ; 1.5];
%l.rx_track.initial_position = [20 ; 30 ; 1.5 ];         % Set start position and MT height
l.rx_track.interpolate_positions(5);                   % One channel sample every 10 cm
l.rx_track.scenario = '3GPP_38.901_UMi_LOS';           % Set propagation scenario

l.rx_track.no_segments = 1;       % Use spatial consisteny for mobility

b = l.init_builder;                                     % Initializes channel builder

b.gen_parameters;                               % Generate small-scale-fading parameters

c = get_channels( b );                          % Generate channel coefficients
c = merge( c, [], 0 );                          % Combine output channels
c.individual_delays = 0;                        % Remove per-antenna delays

output = cell(3,1);
output{1} = b.AoA;
output{2} = b.AoD;
output{3} = squeeze(c.coeff(1,1,:,:))';

l.visualize
