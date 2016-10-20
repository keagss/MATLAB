
X = trnFeature_Set1;
X = X';
X = X(:); % changes into 150x1 cell

data ={};

for sampleI = 1 : length(X)
X_trn = hist(double(X{sampleI})', 0:5:255)'./length(X{sampleI});
Feature = X_trn(:)';
data = [data; {Feature}]; 
end
data = cell2mat(data);

A = 1:10;
Y = repmat(A, 15, 1)
Y = Y(:);
%-------------------------------------------------------------
Z = tstFeature_Set1;
Z = Z';
Z = Z(:);

tstData= {};
for sampleI = 1 : length(Z)
Z_trn = hist(double(Z{sampleI})', 0:5:255)'./length(Z{sampleI});
Feature2 = Z_trn(:)';
tstData = [tstData; {Feature2}]; 
end
tstData = cell2mat(tstData)

%SVM-------------------------------------------------------------


FeatureSpace = [1,4000];
idx = crossvalind('Kfold',Y,1);
X_trn = data(idx==1,FeatureSpace);
Y_trn = Y(idx==1,:);
X_tst = tstData(idx==1,FeatureSpace);
Y_tst = Y(idx==1,:);



tempSVM = templateSVM('Standardize',true,'KernelFunction','gaussian');
% Fit a multi-class SVM classifier
Mdl = fitcecoc(X_trn,Y_trn,'Learners',tempSVM,'FitPosterior',true,'Verbose',2);


Y_tst_Predict = predict(Mdl,X_tst);
[SVMconfusionMatrix] = confusionmat(Y, Y_tst_Predict);

% Check accuracy
nCorrectPredictions = sum(Y_tst==Y_tst_Predict);
Accuracy1 = nCorrectPredictions/length(Y_tst);
fprintf('Accuracy on testing set is: %.4f%%\n',Accuracy1*100);
%NN-------------------------------------------------------------

net = feedforwardnet([8,9]);
net = train(net,X_trn',Y_trn');
view(net);


Y_tst_Predict = net(X_tst');
Y_tst_Predict = round(Y_tst_Predict)';
[NNconfusionMatrix] = confusionmat(Y, Y_tst_Predict);%confusion matrix
nCorrectPredictions = sum(Y_tst==Y_tst_Predict);

Accuracy2 = nCorrectPredictions/length(Y_tst);
fprintf('Accuracy on testing set is: %.4f%%\n',Accuracy2*100);

%Kmeans-------------------------------------------------------------

figure;
U = data(:,2:3);
k = 10;
[idx, C] = kmeans(U,k);
hold on;
gscatter(U(:,1),U(:,2),idx)
plot(C(:,1),C(:,2),'kx','MarkerSize',10,'LineWidth',3) % highlight centroids
title 'trn K-Means Clustering'
xlabel 'X'
ylabel 'Y'




%GMM-------------------------------------------------------------
% Plot the data in the feature space
figure;
gscatter(U(:,1),U(:,2),Y);

xlabel('X');
ylabel('Y');
% GMM tuning parameters
k = 10;
options = statset('MaxIter',1000); % Increase number of EM iterations
% Plotting parameters
d = 500;
x1 = linspace(min(U(:,1)) - 2,max(U(:,1)) + 2,d);
x2 = linspace(min(U(:,2)) - 2,max(U(:,2)) + 2,d);
[x1grid,x2grid] = meshgrid(x1,x2);
X0 = [x1grid(:) x2grid(:)]; % create points that cover the data space of interest
threshold = sqrt(chi2inv(0.99,2)); % select Gaussian with 99% confidence
plotColors = colormap(colorcube)
c = 1;
figure('position', [100, 100, 1400,800]);
% Fit gaussian and plot

for ik = 1 : k
% fit ik Gaussian mixtures
title('trn');
gmModel = fitgmdist(U,ik,'Options',options);
clusterX = cluster(gmModel,U);
% plot Gaussians over truth
mahalDist = mahal(gmModel,X0);
subplot(4,3,c);

h1 = gscatter(U(:,1),U(:,2),clusterX,plotColors); % plot truth 'rbgm'
 hold on;
for m = 1:ik;
idx = mahalDist(:,m)<=threshold; %find coverage fo 99% for Gaussian |m|
h2 = plot(X0(idx,1),X0(idx,2),'.','MarkerSize',1);
uistack(h2,'bottom');
end
plot(gmModel.mu(:,1),gmModel.mu(:,2),'kx','LineWidth',2,'MarkerSize',10)
axis([0,1,0,1])
hold off
c = c + 1;
end
%tstKmeans-------------------------------------------------------
figure;
L = tstData(:,2:3);
k = 10;
[idx, C] = kmeans(L,k);
hold on;
gscatter(L(:,1),L(:,2),idx)
plot(C(:,1),C(:,2),'kx','MarkerSize',10,'LineWidth',3) % highlight centroids
title 'tst K-Means Clustering'
xlabel 'X'
ylabel 'Y'
%tstGMM----------------------------------------------------------
figure;
gscatter(L(:,1),L(:,2),Y);
title('tst');
xlabel('X');
ylabel('Y');
% GMM tuning parameters
k = 10;
options = statset('MaxIter',1000); % Increase number of EM iterations
% Plotting parameters
d = 500;
x1 = linspace(min(L(:,1)) - 2,max(L(:,1)) + 2,d);
x2 = linspace(min(L(:,2)) - 2,max(L(:,2)) + 2,d);
[x1grid,x2grid] = meshgrid(x1,x2);
X0 = [x1grid(:) x2grid(:)]; % create points that cover the data space of interest
threshold = sqrt(chi2inv(0.99,2)); % select Gaussian with 99% confidence
plotColors = colormap(colorcube)
c = 1;
figure('position', [100, 100, 1400,800]);

% Fit gaussian and plot
for ik = 1 : k
    title('tstGMM')
% fit ik Gaussian mixtures
gmModel = fitgmdist(L,ik,'Options',options);
clusterX = cluster(gmModel,L);
% plot Gaussians over truth
mahalDist = mahal(gmModel,X0);
subplot(4,3,c);

h1 = gscatter(L(:,1),L(:,2),clusterX,plotColors); % plot truth 'rbgm'
 hold on;
for m = 1:ik;
idx = mahalDist(:,m)<=threshold; %find coverage fo 99% for Gaussian |m|
h2 = plot(X0(idx,1),X0(idx,2),'.','MarkerSize',1);
uistack(h2,'bottom');
end
plot(gmModel.mu(:,1),gmModel.mu(:,2),'kx','LineWidth',2,'MarkerSize',10)
axis([0,1,0,1])
hold off
c = c + 1;
end
%LDA-------------------------------------------------------------

 LDAModel = fitcdiscr(data(:, 1:4000),Y); 

idx = predict(LDAModel, tstData(:, 1:4000));

[LDAconfusionMatrix] = confusionmat(Y,idx);


Diff = Y-idx;
ind = find(Diff==0);
Correct = numel(ind);
Wrong = numel(Y_tst) - Correct;
Accuracy3 = Correct/(Correct+Wrong);

%Confusion Matrix figures------------------------------------------
figure();

imshow(LDAconfusionMatrix, 'InitialMagnification',5000)  % # you want your cells to be larger than single pixels
title('LDA');
colormap(jet) 

figure();

imshow(SVMconfusionMatrix, 'InitialMagnification',5000)  % # you want your cells to be larger than single pixels
title('SVM');
colormap(jet) 

figure();

imshow(NNconfusionMatrix, 'InitialMagnification',5000)  % # you want your cells to be larger than single pixels
title('NN');
colormap(jet) ;
%Accuracy----------------------------------------------------------
fprintf('SVM accuracy: %.4f%%\n',Accuracy1*100);
fprintf('NN accuracy: %.4f%%\n',Accuracy2*100);
disp(['LDA accuracy: ' num2str(Accuracy3*100) '%']);

Results=cell(3,2);
Results{1, 1} = 'SVM';
Results{2, 1} = 'NN';
Results{3, 1} = 'LDA';

Results{1, 2} = Accuracy1*100;
Results{2, 2} = Accuracy2*100;
Results{3, 2} = Accuracy3*100;

