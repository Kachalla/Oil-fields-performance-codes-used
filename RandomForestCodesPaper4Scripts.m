
% Oil-field-performance
%Random forest codes for estimating recovery factor, maximum oil rate and average depletion rate
%Load and Pre-process Data
load('Random Forest unbiasses predictor importance with Matlab.mat')
%The code below will convert loaded numerical variables that are categorical into categorical variable.

Diagenesis = categorical(Diagenesis);
Paleoclimate = categorical(Paleoclimate);
GrossDepEnvironment = categorical(GrossDepEnvironment);
ProductionStrategy = categorical(ProductionStrategy); 
StratigraphicHeterogeneity = categorical(StratigraphicHeterogeneity); 
TrapType = categorical(Traptype);
StructuralComplexity = categorical(StructuralComplexity);
%Make a table X containing all the training data using 32 variables
X = table(API0,AverageGOR,AverageMonthlyDepletionRate,AvgPermeabilitymD,AvgPorosity,BulkRockVolume108m3,CumOilproducedMillSm3,Diageneticimpact,FaultsCompartments,GrossDepEnvironment,MaximumWellRateKbpd,NetGross,NoofFaultpopulation,NoofInjectionwells,NoofProductionwells,OIPMillSm3,Paleoclimate,PayAreakm2,Pressurebar,ProductionStrategy,ReservoirDepthm,StratigraphicHeterogeneity,StructuralComplexity,Temperature0C,Thicknessm,TotalNowells,TrapType,WaterSaturation,WellDensity108wellm3,wellSpacingkm2well,RecoveryFactor);
%Determine the number of Levels in Predictors
countLevels = @(x)numel(categories(categorical(x)));
numLevels = varfun(countLevels,X(:,1:end-1),'OutputFormat','uniform');
%Grow Robust Random Forest
%For the three models we had Mdl1, Mdl2 and Mdl3, with X, Y and Z tables corresponding to recovery factor, maximum reservoir rate and depletion rate models respectively.
t = templateTree('NumVariablesToSample','all',...
  'PredictorSelection','interaction-curvature','Surrogate','on');
Mdl = fitrensemble(X,'RecoveryFactor','Method','bag','NumLearningCycles',500,...  'Learners',t);
%Estimate the model R2 using out-of-bag prediction
yHat = oobPredict(Mdl);
R2 = corr(Mdl.Y,yHat)^2;
%Predictor Importance Estimation
impOOB1 = oobPermutedPredictorImportance(Mdl1);
impOOB2 = oobPermutedPredictorImportance(Mdl2);
impOOB3 = oobPermutedPredictorImportance(Mdl3);
%Make prediction on new data
% X below is the test dataset
YHat = predict(Mdl,X);




