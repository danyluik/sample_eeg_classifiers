%% Relevant file collection, classification preparation
cd 'C:\Users\Matt\Desktop\Spring-Summer 2020\CML Research\Data\ERPs'

files = dir('*.mat');
subjects = size(files, 1);

% Arrays to store AUC for each subject
subject_scores_lda = zeros(subjects, 1);
subject_scores_svm = zeros(subjects, 1);
feature_weights = zeros(subjects, 30);

%% Manual feature selection
% Electrodes that generally cover scalp, reducing redundancy in data
%selected_electrodes = [8 21 41 64 87 101 137 153 194 214];
selected_electrodes = [36 72 119 173 224 257];

% Step used over time interval to compress data
N = 60;

electrodes = 257;
time = 300;
trials = 112; % 112 intact pairs

%% Data classification for each subject
for index = 1:subjects
    disp(index);
    load(files(index).name);
    data = cat(3, hits_epoch, misses_epoch);
    data = data(:,26:325,:);
    hits = size(hits_epoch, 3);
    misses = size(misses_epoch, 3);
    trials = size(data, 3);

    time_interval = 1; % Will track which time bin is currently being analyzed
    electrode_interval = 1; % Will track which electrode is currently being analyzed

    % Matrix to store final compressed data (6 electrodes, 5 time
    % intervals)
    reduced = zeros(size(selected_electrodes, 2), time/N, trials);

    for i = 1:electrodes
        % Only proceed if electrode has been manually selected
        if ismember(i, selected_electrodes)
            % Gives matrix of samples*trials for recordings from this
            % electrode
            electrode_data = squeeze(data(i,:,:));

            % Temp will store binned data from the electode
            temp = zeros(time/N, trials);
            
            % Loop over data from the electrode time/N times (325/65 = 5)
            for j = 1:N:time
                temp(time_interval,:) = mean(electrode_data(j:j+N-1, :));
                time_interval = time_interval + 1;
            end

            % Store data from this electrode in matrix "reduced"
            reduced(electrode_interval,:,:) = temp;

            % Sets interval variables for next loop
            time_interval = 1;
            electrode_interval = electrode_interval + 1;
        end
    end
    
    % Creates matrix new_reduced, with dimensions of
    % trials * (selected_electrodes * time/N) , 112*(6*5)
    reduced = permute(reduced, [2 1 3]);
    dim = size(reduced);
    new_reduced = zeros(dim(3), dim(1)*dim(2));
    
    % For each trial, gather all recordings (each electrode at each time),
    % and store as a row in new_reduced
    for k = 1:trials
        temp = reduced(:,:,k);
        new_reduced(k,:) = temp(:);
    end
        
    rng('default'); % Ensures replicability
    events = [ones(hits,1); zeros(misses,1)]; % All intact pairs
    
    % No further analysis if < 10 misses
    if sum(events == 0) < 10
        disp('no');
        continue
    end
    
    % Sorts data into 10 folds for cross-validation
    cvp = cvpartition(events,'KFold',10,'Stratify',true);

    % LDA analysis
    model_lda = fitcdiscr(new_reduced,events,'DiscrimType', 'linear','FillCoeffs', 'off','ClassNames', [double(0); double(1)],'CVPartition',cvp);
    [~,score_lda] = kfoldPredict(model_lda); % Gets classifier scores for each trial (after CV)
    [~,~,~,auc_lda] = perfcurve(events,score_lda(:,2),1); % Positive class in column 2 of scores
    subject_scores_lda(index) = auc_lda; % Store final AUC value
    
    % SVM analysis
    model_svm = fitcsvm(new_reduced,events,'ClassNames', [double(0); double(1)],'CVPartition',cvp);
    [~,score_svm] = kfoldPredict(model_svm);
    [~,~,~,auc_svm] = perfcurve(events,score_svm(:,2),1);
    subject_scores_svm(index) = auc_svm;
    
    current_feature_weights = zeros(10,size(feature_weights,2));
    
    for p = 1:10
        current_model_lda = model_lda.Trained{p};
        current_feature_weights(p, :) = current_model_lda.DeltaPredictor;
    end
    
    current_feature_weights = mean(current_feature_weights, 1);
    feature_weights(index, :) = current_feature_weights;
end

%% Further analysis
% subject_scores_lda = subject_scores_lda(subject_scores_lda ~= 0);
% subject_scores_svm = subject_scores_svm(subject_scores_svm ~= 0);
dist_lda = fitdist(subject_scores_lda(subject_scores_lda ~= 0), 'Normal');
dist_svm = fitdist(subject_scores_svm(subject_scores_svm ~= 0), 'Normal');

feature_weights = feature_weights(subject_scores_lda ~= 0, :);
%feature_weights = rescale(feature_weights);
avg_feature_weights = mean(feature_weights, 1);

cluster_values = zeros(4,1);
current = 1;

for m = 2:5
    idx = kmeans(feature_weights, m, 'Replicates', 100);
    s = silhouette(feature_weights, idx);
    cluster_values(current) = mean(s);
    current = current + 1;
end

%% Cluster analysis
nums = [2;3;4;5];
cluster = nums(find(max(cluster_values)));

idx = kmeans(feature_weights, cluster, 'Replicates', 100);
[s,Silhouette_HitMiss_Test] = silhouette(feature_weights, idx);

cluster1 = feature_weights(idx == 1, :);
cluster2 = feature_weights(idx == 2, :);

%cd 'C:\Users\Matt\Desktop\Spring-Summer 2020\CML Research\Results\Multivariate Classifiers'
%save('PairAssoHitMissTest.mat','subject_scores_lda','dist_lda',...
%   'subject_scores_svm','dist_svm','feature_weights','idx','s');
%print('-dpdf','Silhouette_HitMiss_Test');

% current = 1;
% 
% for p = 1:10:120
%     fig1 = topoplot(normalize(mean(cluster1(:, p:p+9), 1)), 'GSN257_classifiers.sfp');
%     fig2 = topoplot(normalize(mean(cluster2(:, p:p+9), 1)), 'GSN257_classifiers.sfp');
%     
%     saveas(fig1,sprintf('HitMissTest_Cluster1_%d.png',current));
%     saveas(fig2,sprintf('HitMissTest_Cluster2_%d.png',current));
%     current = current + 1;
% end
