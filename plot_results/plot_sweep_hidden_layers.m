function plot_sweep_hidden_layers(sweep, filename, mode)
    % plot_sweep_hidden_layers(sweep, filename, mode)
    % If not incuded mode it will show encoder o decoder
    if nargin < 3
        mode = 'encoder_decoder';
    end

    % Size parameters for plot
    col_width = 3.5;
    fig_height = 2.35;
    fig = figure('Units','inches','Position',[1 1 col_width fig_height]);
    set(fig,'PaperUnits','inches');
    set(fig,'PaperSize',[col_width fig_height]);
    set(fig,'PaperPosition',[0 0 col_width fig_height]);
    set(fig,'PaperPositionMode','manual');

    % Font size
    custom_fontsize = 8;
    ENERGY_PJ_PER_MAC = 4.6;   % Horowitz, ISSCC 2014, 45nm float32 mult+add (pJ/MAC)
    COLOR_SYM  = [0.00, 0.45, 0.74];
    COLOR_ASYM = [0.85, 0.33, 0.10];

    sweep.encoder_energy_nJ = sweep.encoder_macs * ENERGY_PJ_PER_MAC / 1000;
    sweep.decoder_energy_nJ = sweep.decoder_macs * ENERGY_PJ_PER_MAC / 1000;

    % Get csv stats
    summary = groupsummary(sweep, {'hidden_layers', 'symmetric'}, {'mean', 'std'}, {'encoder_energy_nJ', 'decoder_energy_nJ', 'mse_mean'});

    switch mode
        case 'encoder_decoder'
            leftVar  = 'encoder_energy_nJ'; leftLabel  = 'Encoder energy (nJ)';
            rightVar = 'decoder_energy_nJ'; rightLabel = 'Decoder energy (nJ)';
            figTitle = 'Encoder and decoder energy vs hidden layers';
        case 'mse_encoder'
            leftVar  = 'mse_mean';          leftLabel  = 'MSE (\circC^2)';
            rightVar = 'encoder_energy_nJ'; rightLabel = 'Encoder energy (nJ)';
            figTitle = 'MSE and encoder energy vs hidden layers';
        otherwise
            error('plot_sweep_hidden_layers:badMode','mode must be ''encoder_decoder'' or ''mse_encoder'', no ''%s''', mode);
    end

    t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(t, figTitle, 'FontSize', custom_fontsize, 'FontWeight', 'normal');

    axLeft  = nexttile; hold(axLeft, 'on');
    axRight = nexttile; hold(axRight, 'on');

    leftMeanCol  = ['mean_' leftVar];  leftStdCol  = ['std_' leftVar];
    rightMeanCol = ['mean_' rightVar]; rightStdCol = ['std_' rightVar];

    legend_handles = gobjects(1, 2);
    legend_entries = {'Symmetric', 'Asymmetric'};
    labels = [true, false];
    colors = {COLOR_SYM, COLOR_ASYM};

    for i = 1:numel(labels)
        subset = summary(summary.symmetric == labels(i), :);
        subset = sortrows(subset, 'hidden_layers');

        h = errorbar(axLeft, subset.hidden_layers, subset.(leftMeanCol), subset.(leftStdCol), '-o', 'Color', colors{i}, 'MarkerFaceColor', colors{i}, 'LineWidth', 1.2, 'MarkerSize', 4);
        errorbar(axRight, subset.hidden_layers, subset.(rightMeanCol), subset.(rightStdCol), '-o', 'Color', colors{i}, 'MarkerFaceColor', colors{i}, 'LineWidth', 1.2, 'MarkerSize', 4);
        legend_handles(i) = h;
    end
    hold(axLeft, 'off'); hold(axRight, 'off');

    xlabel(axLeft, 'Number of hidden layers');  ylabel(axLeft, leftLabel);
    xlabel(axRight, 'Number of hidden layers'); ylabel(axRight, rightLabel);
    set(axLeft, 'FontSize', custom_fontsize);
    set(axRight, 'FontSize', custom_fontsize);

    lgd = legend(legend_handles, legend_entries, 'Orientation', 'horizontal','FontSize', custom_fontsize, 'Box', 'off', 'NumColumns', 2);
    lgd.Layout.Tile = 'south';
    lgd.ItemTokenSize = [8 6];

    full_file_name = strcat(filename, '.pdf');
    exportgraphics(fig, full_file_name, 'ContentType', 'vector');
end
