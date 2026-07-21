function plot_sweep_mae_ratio(sweep, file_name)
    % Size parameters for plot
    col_width = 3.5;
    fig_height = 2.35;
    fig = figure('Units','inches','Position',[1 1 col_width fig_height]);
    set(fig,'PaperUnits','inches');
    set(fig,'PaperSize',[col_width fig_height]);
    set(fig,'PaperPosition',[0 0 col_width fig_height]);
    set(fig,'PaperPositionMode','manual');

    %Font size
    custom_fontsize = 8;
    COLOR_SYM  = [0.00, 0.45, 0.74];
    COLOR_ASYM = [0.85, 0.33, 0.10];

    latentDims = sort(unique(sweep.latent_dim));
    
    % Tiled layout ratio left, MAE right
    t = tiledlayout(1, 2, 'TileSpacing','compact', 'Padding','compact');
    axRatio = nexttile; hold(axRatio, 'on');
    axMAE = nexttile; hold(axMAE, 'on');
    % Global title
    sgtitle(t, sprintf('Compression ratio and MAE vs latent dimension'), 'FontSize',custom_fontsize, 'FontWeight','normal');
    
    % Manage global legend
    legend_handles = gobjects(1, 2);
    legend_entries = {'Symmetric', 'Asymmetric'};
    labels = [true, false];
    colors = {COLOR_SYM, COLOR_ASYM};
    idx = 0;
    % For loop
    for i = 1:numel(labels)
        symVal = labels(i);

        ratio_mean_v = zeros(numel(latentDims), 1); ratio_ci_v = zeros(numel(latentDims), 1);
        mae_mean_v   = zeros(numel(latentDims), 1); mae_ci_v   = zeros(numel(latentDims), 1);
        
        for k = 1:numel(latentDims)
            mask = sweep.symmetric == symVal & sweep.latent_dim == latentDims(k);

            ratio_vals = sweep.ratio(mask);
            mae_vals   = sweep.mae_mean(mask);

            [ratio_mean_v(k), ratio_ci_v(k)] = ConfidenceInterval(ratio_vals);
            [mae_mean_v(k), mae_ci_v(k)]     = ConfidenceInterval(mae_vals);
        end
        if symVal
            lineSpec = '-o';
        else
            lineSpec = '--s';
        end

        h = errorbar(axRatio, latentDims, ratio_mean_v, ratio_ci_v, lineSpec, 'Color', colors{i}, 'MarkerFaceColor', colors{i}, 'LineWidth', 1.2, 'MarkerSize', 4);
        errorbar(axMAE, latentDims, mae_mean_v, mae_ci_v, lineSpec, 'Color', colors{i}, 'MarkerFaceColor', colors{i}, 'LineWidth', 1.2, 'MarkerSize', 4);
        legend_handles(i) = h;
    end
    hold(axRatio, 'off'); hold(axMAE, 'off');
    xlabel(axRatio, 'Latent dimension'); ylabel(axRatio, 'Compression ratio');
    xlabel(axMAE, 'Latent dimension'); ylabel(axMAE, 'MAE (°C)');
    set(axRatio, 'FontSize', custom_fontsize);
    set(axMAE, 'FontSize', custom_fontsize);
    % Leyenda global fuera, abajo
    lgd = legend(legend_handles, legend_entries, 'Orientation', 'horizontal', 'FontSize',custom_fontsize, 'Box','off', 'NumColumns',2);
    lgd.Layout.Tile = 'south';
    lgd.ItemTokenSize = [8 6];

    % Export file to PDF
    full_file_name = strcat(file_name, '.pdf');
    exportgraphics(fig,full_file_name,'ContentType','vector');
end