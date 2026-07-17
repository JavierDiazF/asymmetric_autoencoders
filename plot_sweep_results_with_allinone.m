
%% plot_sweep_results.m
% Lee results/sweep_a_results.csv y results/sweep_b_results.csv (generados
% por main.py) y genera las graficas de comparacion:
%
%   Figura 1: Sweep A -- MSE vs ratio de compresion (una subgrafica por input_dim)
%   Figura 2: Sweep A -- MAE vs ratio de compresion (una subgrafica por input_dim)
%   Figura 3: Sweep B -- energia estimada (encoder y decoder) vs num. de hidden layers
%   Figura 4: Sweep B -- MSE vs num. de hidden layers
%
% En todas las graficas se comparan modelos simetricos vs. asimetricos,
% agregando (media +/- desviacion tipica) sobre las 10 semillas entrenadas
% por configuracion.

clear; clc; close all;

RESULTS_DIR = 'results/azure_results';
ENERGY_PJ_PER_MAC = 4.6;   % Horowitz, ISSCC 2014, 45nm float32 mult+add (pJ/MAC)

COLOR_SYM  = [0.00, 0.45, 0.74];   % azul   -> simetrico
COLOR_ASYM = [0.85, 0.33, 0.10];   % naranja -> asimetrico

%% ------------------------------------------------------------------
%  Carga de datos
%% ------------------------------------------------------------------
sweepA = readtable(fullfile(RESULTS_DIR, 'sweep_a_results.csv'));
sweepB = readtable(fullfile(RESULTS_DIR, 'sweep_b_results.csv'));
sweepC = readtable(fullfile(RESULTS_DIR, 'sweep_c_results.csv'));

sweepA.symmetric = parseBoolColumn(sweepA.symmetric);
sweepB.symmetric = parseBoolColumn(sweepB.symmetric);
sweepC.symmetric = parseBoolColumn(sweepC.symmetric);

%% ------------------------------------------------------------------
%  Numero total de combinaciones (input_dim, latent_dim) en Sweep A
%% ------------------------------------------------------------------
combosA = unique(sweepA(:, {'input_dim', 'latent_dim'}));
numCombosA = height(combosA);
fprintf('Sweep A: %d combinaciones distintas de (input_dim, latent_dim)\n', numCombosA);

%% ------------------------------------------------------------------
%  Figura 1 y 2: Sweep A -- MSE y MAE vs ratio de compresion,
%  una subgrafica por input_dim (vista exploratoria)
%% ------------------------------------------------------------------
inputDims = unique(sweepA.input_dim);

plotSweepAMetric(sweepA, inputDims, 'mse_mean', 'MSE (°C^2)', 'Sweep A: MSE vs ratio de compresion', COLOR_SYM, COLOR_ASYM);
plotSweepAMetric(sweepA, inputDims, 'mae_mean', 'MAE (°C)', 'Sweep A: MAE vs ratio de compresion', COLOR_SYM, COLOR_ASYM);

%% ------------------------------------------------------------------
%  Figura 1b y 2b: Sweep A -- todas las configuraciones en una sola
%  grafica (para el paper): color = input_dim, linea/marcador = sim/asim
%% ------------------------------------------------------------------
plotSweepAAllInOneCI(sweepA, 'mse_mean', 'MSE (°C^2)', 'Sweep A: MSE vs ratio de compresion (todas las configuraciones)');
plotSweepAAllInOneCI(sweepA, 'mae_mean', 'MAE (°C)', 'Sweep A: MAE vs ratio de compresion (todas las configuraciones)');
plotSweepAAllInOne(sweepA, 'mse_mean', 'MSE (°C^2)', 'Sweep A: MSE vs ratio de compresion (todas las configuraciones)');
plotSweepAAllInOne(sweepA, 'mae_mean', 'MAE (°C)', 'Sweep A: MAE vs ratio de compresion (todas las configuraciones)');
%% ------------------------------------------------------------------
%  Figura 1b y 2b: Sweep C -- todas las configuraciones en una sola
%  grafica (para el paper): color = input_dim, linea/marcador = sim/asim
%% ------------------------------------------------------------------
plotSweepAAllInOneCI(sweepC, 'mse_mean', 'MSE (°C^2)', 'Sweep C: MSE vs ratio de compresion (todas las configuraciones)');
plotSweepAAllInOneCI(sweepC, 'mae_mean', 'MAE (°C)', 'Sweep C: MAE vs ratio de compresion (todas las configuraciones)');

%% ------------------------------------------------------------------
%  Figura 3: Sweep B -- energia (encoder y decoder por separado) vs
%  numero de hidden layers
%% ------------------------------------------------------------------
sweepB.encoder_energy_nJ = sweepB.encoder_macs * ENERGY_PJ_PER_MAC / 1000;
sweepB.decoder_energy_nJ = sweepB.decoder_macs * ENERGY_PJ_PER_MAC / 1000;

figure('Name', 'Sweep B - Energia vs hidden layers', 'Color', 'w');
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
plotBySymmetric(sweepB, 'hidden_layers', 'encoder_energy_nJ', COLOR_SYM, COLOR_ASYM);
xlabel('Numero de hidden layers');
ylabel('Energia estimada del encoder (nJ)');
title('Encoder (lado del sensor)');
legend('Simetrico', 'Asimetrico', 'Location', 'best');
grid on;

nexttile;
plotBySymmetric(sweepB, 'hidden_layers', 'decoder_energy_nJ', COLOR_SYM, COLOR_ASYM);
xlabel('Numero de hidden layers');
ylabel('Energia estimada del decoder (nJ)');
title('Decoder (lado del receptor)');
legend('Simetrico', 'Asimetrico', 'Location', 'best');
grid on;

sgtitle('Sweep B: energia estimada (MACs x 4.6 pJ/MAC) vs profundidad');

%% ------------------------------------------------------------------
%  Figura 4: Sweep B -- MSE vs numero de hidden layers
%% ------------------------------------------------------------------
figure('Name', 'Sweep B - MSE vs hidden layers', 'Color', 'w');
plotBySymmetric(sweepB, 'hidden_layers', 'mse_mean', COLOR_SYM, COLOR_ASYM);
xlabel('Numero de hidden layers');
ylabel('MSE (°C^2)');
title('Sweep B: MSE vs profundidad');
legend('Simetrico', 'Asimetrico', 'Location', 'best');
grid on;


%% ==================================================================
%  Funciones locales
%% ==================================================================

function boolCol = parseBoolColumn(col)
    % pandas.to_csv escribe los booleanos de Python como texto "True"/"False",
    % no como 0/1 ni "true"/"false" en minuscula -- readtable puede
    % importar esa columna como cell de char, string o categorical segun
    % la version, nunca como logical automaticamente. Se normaliza aqui.
    if islogical(col)
        boolCol = col;
    else
        boolCol = string(col) == "True";
    end
end

function plotSweepAMetric(sweepA, inputDims, valueVar, yLabelStr, figTitle, colorSym, colorAsym)
    figure('Name', figTitle, 'Color', 'w');
    tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    for k = 1:numel(inputDims)
        nexttile;
        thisDim = sweepA(sweepA.input_dim == inputDims(k), :);

        summaryT = groupsummary(thisDim, {'latent_dim', 'symmetric'}, {'mean', 'std'}, valueVar);
        summaryT.ratio = inputDims(k) ./ (summaryT.latent_dim + 3);

        plotGroupedErrorbar(summaryT, 'ratio', valueVar, colorSym, colorAsym);

        xlabel('Ratio de compresion (input\_dim / (latent\_dim+3))');
        ylabel(yLabelStr);
        title(sprintf('input\\_dim = %d', inputDims(k)));
        legend('Simetrico', 'Asimetrico', 'Location', 'best');
        grid on;
    end
    sgtitle(figTitle);
end


function plotSweepAAllInOne(sweepA, valueVar, yLabelStr, figTitle)
    % Una sola grafica con TODAS las combinaciones (input_dim, latent_dim,
    % simetrico/asimetrico): color = input_dim, estilo de linea/marcador =
    % simetrico vs. asimetrico. Pensada para figura de paper (compacta,
    % ratio en escala log ya que abarca un rango amplio).
    %
    % NOTA: version antigua, usa la desviacion tipica de groupsummary como
    % barra de error. Sustituida por plotSweepAAllInOneCI (usa
    % ConfidenceInterval sobre el vector crudo de semillas). Se deja aqui
    % comentada como referencia.
    inputDims = unique(sweepA.input_dim);
    dimColors = lines(numel(inputDims));   % paleta con un color distinto por input_dim

    summaryT = groupsummary(sweepA, {'input_dim', 'latent_dim', 'symmetric'}, {'mean', 'std'}, valueVar);
    summaryT.ratio = summaryT.input_dim ./ (summaryT.latent_dim + 3);
    meanCol = ['mean_' valueVar];
    stdCol  = ['std_' valueVar];

    figure('Name', figTitle, 'Color', 'w');
    hold on;
    legendEntries = {};
    for d = 1:numel(inputDims)
        for symVal = [true, false]
            subset = summaryT(summaryT.input_dim == inputDims(d) & summaryT.symmetric == symVal, :);
            if isempty(subset)
                continue;
            end
            subset = sortrows(subset, 'ratio');

            if symVal
                lineSpec = '-o';
                labelSuffix = 'Sim.';
            else
                lineSpec = '--s';
                labelSuffix = 'Asim.';
            end

            errorbar(subset.ratio, subset.(meanCol), subset.(stdCol), lineSpec, ...
                'Color', dimColors(d, :), 'MarkerFaceColor', dimColors(d, :), ...
                'LineWidth', 1.5, 'MarkerSize', 5);
            legendEntries{end + 1} = sprintf('input\\_dim=%d, %s', inputDims(d), labelSuffix); %#ok<AGROW>
        end
    end
    hold off;

    set(gca, 'XScale', 'log');
    xlabel('Ratio de compresion (escala log)');
    ylabel(yLabelStr);
    title(figTitle);
    legend(legendEntries, 'Location', 'bestoutside');
    grid on;
end


function plotSweepAAllInOneCI(sweepA, valueVar, yLabelStr, figTitle)
    % Igual que plotSweepAAllInOne, pero la barra de error es el intervalo
    % de confianza al 95% (ConfidenceInterval.m) calculado sobre el vector
    % crudo de las N semillas de cada combinacion (input_dim, latent_dim,
    % symmetric), en vez de la desviacion tipica que da groupsummary.
    %
    % Requiere que ConfidenceInterval.m este en el path de MATLAB (vive en
    % plot_results/ -- añadir esa carpeta al path si se ejecuta este script
    % desde la raiz del proyecto).
    inputDims = unique(sweepA.input_dim);
    dimColors = lines(numel(inputDims));   % paleta con un color distinto por input_dim

    figure('Name', figTitle, 'Color', 'w');
    hold on;
    legendEntries = {};
    for d = 1:numel(inputDims)
        for symVal = [true, false]
            latentDims = unique(sweepA.latent_dim(sweepA.input_dim == inputDims(d) & sweepA.symmetric == symVal));
            latentDims = sort(latentDims);
            if isempty(latentDims)
                continue;
            end

            ratios   = zeros(numel(latentDims), 1);
            meanVals = zeros(numel(latentDims), 1);
            ciVals   = zeros(numel(latentDims), 1);

            for k = 1:numel(latentDims)
                mask = sweepA.input_dim == inputDims(d) & sweepA.symmetric == symVal & sweepA.latent_dim == latentDims(k);

                rawVals = sweepA.(valueVar)(mask);   % un valor por semilla, para esta config exacta
                [meanVals(k), ciVals(k)] = ConfidenceInterval(rawVals);
                ratios(k) = inputDims(d) ./ (latentDims(k) + 3);
            end

            [ratios, order] = sort(ratios);
            meanVals = meanVals(order);
            ciVals = ciVals(order);

            if symVal
                lineSpec = '-o';
                labelSuffix = 'Sim.';
            else
                lineSpec = '--s';
                labelSuffix = 'Asim.';
            end

            errorbar(ratios, meanVals, ciVals, lineSpec, ...
                'Color', dimColors(d, :), 'MarkerFaceColor', dimColors(d, :), ...
                'LineWidth', 1.5, 'MarkerSize', 5);
            legendEntries{end + 1} = sprintf('input\\_dim=%d, %s', inputDims(d), labelSuffix); %#ok<AGROW>
        end
    end
    hold off;

    set(gca, 'XScale', 'log');
    xlabel('Ratio de compresion (escala log)');
    ylabel(yLabelStr);
    title(figTitle);
    legend(legendEntries, 'Location', 'bestoutside');
    grid on;
end

function plotBySymmetric(T, xVar, valueVar, colorSym, colorAsym)
    summaryT = groupsummary(T, {xVar, 'symmetric'}, {'mean', 'std'}, valueVar);
    plotGroupedErrorbar(summaryT, xVar, valueVar, colorSym, colorAsym);
end

function plotGroupedErrorbar(summaryT, xVar, valueVar, colorSym, colorAsym)
    % summaryT viene de groupsummary: contiene columnas
    % mean_<valueVar> y std_<valueVar> ademas de 'symmetric' y xVar.
    meanCol = ['mean_' valueVar];
    stdCol  = ['std_' valueVar];

    hold on;
    labels = [true, false];
    colors = {colorSym, colorAsym};
    for i = 1:numel(labels)
        subset = summaryT(summaryT.symmetric == labels(i), :);
        subset = sortrows(subset, xVar);
        errorbar(subset.(xVar), subset.(meanCol), subset.(stdCol), '-o', 'Color', colors{i}, 'MarkerFaceColor', colors{i}, 'LineWidth', 1.5);
    end
    hold off;
end
