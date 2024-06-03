% Cargar los datos
MelanomaFeatures = readtable('MelanomaLesionFeatures.csv', 'ReadVariableNames', false);
SeborrheicFeatures = readtable('SeborrheicLesionFeatures.csv', 'ReadVariableNames', false);
NevusFeatures = readtable('NevusLesionFeatures.csv', 'ReadVariableNames', false);
MelanomaControlFeatures = readtable('MelanomaControlFeatures.csv', 'ReadVariableNames', false);
SeborrheicControlFeatures = readtable('SeborrheicControlFeatures.csv', 'ReadVariableNames', false);
NevusControlFeatures = readtable('NevusControlFeatures.csv', 'ReadVariableNames', false);
% Preprocesamiento
channel1 = 1:32;
channel2 = 33:64;
channel3 = 65:96;

% Calcular las características medias y de desviación estándar para las características de melanomas
MMelanomaFeatures = (MelanomaFeatures{:, channel1} + MelanomaFeatures{:, channel2} + MelanomaFeatures{:, channel3}) / 3.0;
SMelanomaFeatures = (abs(MelanomaFeatures{:, channel1} - MMelanomaFeatures) + abs(MelanomaFeatures{:, channel2} - MMelanomaFeatures) + abs(MelanomaFeatures{:, channel3} - MMelanomaFeatures)) / 3.0;

% Asignar nombres únicos a las nuevas columnas
numFeatures = size(SMelanomaFeatures, 2);
for i = 1:numFeatures
    newColNames{i} = sprintf('Feature%d', i);
end

% Convertir a tabla y agregar los nuevos nombres de columnas
SMelanomaFeaturesTable = array2table(SMelanomaFeatures ./ (0.01 + abs(MMelanomaFeatures)), 'VariableNames', newColNames);

% Concatenar las nuevas columnas a la tabla original
MelanomaFeatures = [MelanomaFeatures, SMelanomaFeaturesTable];

% Repetir para Nevus y Seborrheic
MNevusFeatures = (NevusFeatures{:, channel1} + NevusFeatures{:, channel2} + NevusFeatures{:, channel3}) / 3.0;
SNevusFeatures = (abs(NevusFeatures{:, channel1} - MNevusFeatures) + abs(NevusFeatures{:, channel2} - MNevusFeatures) + abs(NevusFeatures{:, channel3} - MNevusFeatures)) / 3.0;

SNevusFeaturesTable = array2table(SNevusFeatures ./ (0.001 + abs(MNevusFeatures)), 'VariableNames', newColNames);
NevusFeatures = [NevusFeatures, SNevusFeaturesTable];

MSeborrheicFeatures = (SeborrheicFeatures{:, channel1} + SeborrheicFeatures{:, channel2} + SeborrheicFeatures{:, channel3}) / 3.0;
SSeborrheicFeatures = (abs(SeborrheicFeatures{:, channel1} - MSeborrheicFeatures) + abs(SeborrheicFeatures{:, channel2} - MSeborrheicFeatures) + abs(SeborrheicFeatures{:, channel3} - MSeborrheicFeatures)) / 3.0;

SSeborrheicFeaturesTable = array2table(SSeborrheicFeatures ./ (0.001 + abs(MSeborrheicFeatures)), 'VariableNames', newColNames);
SeborrheicFeatures = [SeborrheicFeatures, SSeborrheicFeaturesTable];

% Agregar las etiquetas de clase
MelanomaFeatures.Class = ones(height(MelanomaFeatures), 1);
NevusFeatures.Class = zeros(height(NevusFeatures), 1);
SeborrheicFeatures.Class = zeros(height(SeborrheicFeatures), 1);

% Combinar todos los datos
AllFeatures = [MelanomaFeatures; NevusFeatures; SeborrheicFeatures];
% Separar los datos en conjunto de entrenamiento y prueba
cv = cvpartition(AllFeatures.Class, 'HoldOut', 0.3);
trainData = AllFeatures(training(cv), :);
testData = AllFeatures(test(cv), :);


% LASSO (Utilizando Regresión Logística)
[B, FitInfo] = lasso(table2array(trainData(:, 1:end-1)), trainData.Class, 'CV', 10);
lassoPred = table2array(testData(:, 1:end-1)) * B(:, FitInfo.IndexMinMSE) + FitInfo.Intercept(FitInfo.IndexMinMSE);
lassoPred = lassoPred > 0.5;

% Convertir testLabels a tipo double si aún no lo es
testLabels = double(testData.Class);

% Asegurarse de que lassoPred tenga valores binarios (0 y 1)
lassoPred = double(lassoPred > 0.5);

% Calcular la matriz de confusión
[confMatLASSO, orderLASSO] = confusionmat(testLabels, lassoPred);

function [accuracy, sensitivity, specificity, F1] = evaluateModel(confMat)
    % Calcular métricas de evaluación
    TP = confMat(1, 1); % Verdaderos Positivos
    TN = confMat(2, 2); % Verdaderos Negativos
    FP = confMat(2, 1); % Falsos Positivos
    FN = confMat(1, 2); % Falsos Negativos
    
    % Calcular métricas
    accuracy = (TP + TN) / sum(confMat(:)); % Exactitud
    sensitivity = TP / (TP + FN); % Sensibilidad
    specificity = TN / (TN + FP); % Especificidad
    precision = TP / (TP + FP); % Precisión
    recall = TP / (TP + FN); % Recall
    F1 = 2 * (precision * recall) / (precision + recall); % Puntuación F1
end


disp('LASSO Confusion Matrix:');
disp(confMatLASSO);
% Evaluar el modelo LASSO
[accLASSO, sensLASSO, specLASSO, F1LASSO] = evaluateModel(confMatLASSO);

disp('LASSO Evaluation:');
disp(['Accuracy: ', num2str(accLASSO)]);
disp(['Sensitivity: ', num2str(sensLASSO)]);
disp(['Specificity: ', num2str(specLASSO)]);
disp(['F1 Score: ', num2str(F1LASSO)]);

% Gráfica ROC para LASSO
scoresLASSO = table2array(testData(:, 1:end-1)) * B(:, FitInfo.IndexMinMSE) + FitInfo.Intercept(FitInfo.IndexMinMSE);
[XLSO, YLSO, ~, AUCLSO] = perfcurve(testData.Class, scoresLASSO, 1);

figure;
plot(XLSO, YLSO, 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for LASSO');
