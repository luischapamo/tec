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

% Entrenamiento y evaluación de modelos
% K-Nearest Neighbors
knnModel = fitcknn(trainData(:, 1:end-1), trainData.Class);
knnPred = predict(knnModel, testData(:, 1:end-1));
[confMatKNN, orderKNN] = confusionmat(testData.Class, knnPred);
disp('KNN Confusion Matrix:');
disp(confMatKNN);

% Support Vector Machine
svmModel = fitcsvm(trainData(:, 1:end-1), trainData.Class);
svmPred = predict(svmModel, testData(:, 1:end-1));
[confMatSVM, orderSVM] = confusionmat(testData.Class, svmPred);
disp('SVM Confusion Matrix:');
disp(confMatSVM);

% Naive Bayes
nbModel = fitcnb(trainData(:, 1:end-1), trainData.Class);
nbPred = predict(nbModel, testData(:, 1:end-1));
[confMatNB, orderNB] = confusionmat(testData.Class, nbPred);
disp('NB Confusion Matrix:');
disp(confMatNB);

% Evaluar cada modelo
[accKNN, sensKNN, specKNN, F1KNN] = evaluateModel(confMatKNN);
[accSVM, sensSVM, specSVM, F1SVM] = evaluateModel(confMatSVM);
[accNB, sensNB, specNB, F1NB] = evaluateModel(confMatNB);

disp('KNN Evaluation:');
disp(['Accuracy: ', num2str(accKNN)]);
disp(['Sensitivity: ', num2str(sensKNN)]);
disp(['Specificity: ', num2str(specKNN)]);
disp(['F1 Score: ', num2str(F1KNN)]);

disp('SVM Evaluation:');
disp(['Accuracy: ', num2str(accSVM)]);
disp(['Sensitivity: ', num2str(sensSVM)]);
disp(['Specificity: ', num2str(specSVM)]);
disp(['F1 Score: ', num2str(F1SVM)]);

disp('Naive Bayes Evaluation:');
disp(['Accuracy: ', num2str(accNB)]);
disp(['Sensitivity: ', num2str(sensNB)]);
disp(['Specificity: ', num2str(specNB)]);
disp(['F1 Score: ', num2str(F1NB)]);

% Predict class probabilities for KNN
scoresKNN = predict(knnModel, testData(:, 1:end-1));

% Calculate distances to the nearest neighbors as a score
[~, distances] = knnModel.predict(testData(:, 1:end-1));
% Convert distances to scores by normalizing them between 0 and 1
maxDist = max(distances);
minDist = min(distances);
scoresKNN = (distances - minDist) / (maxDist - minDist);

% Predict class probabilities for KNN
[~, scoresKNN] = predict(knnModel, testData(:, 1:end-1));

% Predict class probabilities for SVM
[~, scoresSVM] = predict(svmModel, testData(:, 1:end-1));

% Predict class probabilities for Naive Bayes
scoresNB = predict(nbModel, testData(:, 1:end-1));

% Generate ROC curves for all models
[XKNN, YKNN, ~, AUCKNN] = perfcurve(testData.Class, scoresKNN(:, 2), 1);
[XSVM, YSVM, ~, AUCSVM] = perfcurve(testData.Class, scoresSVM(:, 2), 1);
[XNB, YNB, ~, AUCNB] = perfcurve(testData.Class, scoresNB, 1);

% Plot ROC curves
figure;
plot(XKNN, YKNN, 'LineWidth', 2);
hold on;
plot(XSVM, YSVM, 'LineWidth', 2);
plot(XNB, YNB, 'LineWidth', 2);
legend('KNN', 'SVM', 'Naive Bayes', 'Location', 'Best');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');
hold off;

% Mapa de calor
data = table2array(AllFeatures(:, 1:end-1)); % Seleccionar las características sin la columna de clase
classLabels = AllFeatures.Class;
figure;
colormap('jet'); % Set the colormap
heatmap(data, 'XLabel', 'Features', 'YLabel', 'Samples');

function [accuracy, sensitivity, specificity, F1] = evaluateModel(confMat)
    TP = confMat(2,2);
    TN = confMat(1,1);
    FP = confMat(1,2);
    FN = confMat(2,1);

    accuracy = (TP + TN) / sum(confMat(:));
    sensitivity = TP / (TP + FN);
    specificity = TN / (TN + FP);
    F1 = (2 * TP) / (2 * TP + FP + FN);
end