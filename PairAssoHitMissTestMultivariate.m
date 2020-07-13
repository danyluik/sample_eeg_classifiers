%% Relevant file collection
files = dir('*.mat');
subjects = size(files, 1);

% Array to store AUC for each subject
subject_scores_svm = zeros(subjects, 1);

% Electrodes that generally cover scalp, reducing redundancy in data
selected_electrodes = [36 72 119 173 224 257];

% Step used over time interval to compress data
N = 60;

electrodes = 257;
time = 300;

%% Data classification for each subject
for index = 1:subjects
    disp(index);
    load(files(index).name);
    
    %% Combines epoched hit and miss data, creates event labels
    data = cat(3, hits_epoch, misses_epoch);
    data = data(:,26:325,:);
    hits = size(hits_epoch, 3);
    misses = size(misses_epoch, 3);
    events = [ones(hits,1); zeros(misses,1)];
    trials = size(data, 3);
    
    % No further analysis if < 10 misses (so cv works later)
    if sum(events == 0) < 10
        disp('no');
        continue
    end

    %% Reduce data to relevant electrodes
    selected_data = data(selected_electrodes,:,:);
    
    %% Matrix to store reduced data (electrodes * time * trials)
    reduced_data = zeros(size(selected_electrodes, 2), time/N, trials);
    
    %% Averages recording into time/N bins
    iter = 1;
    for j = 1:N:time
        reduced_data(:,iter,:) = mean(selected_data(:,j:j+N-1,:), 2);
        iter = iter + 1;
    end
    
    %% Matrix to store final reduced data (trials * features)
    dim = size(reduced_data);
    final_reduced_data = zeros(dim(3), dim(1)*dim(2));
    
    % For each trial, gather all recordings (each electrode at each time), and store as a row
    for k = 1:trials
        temp = reduced_data(:,:,k);
        final_reduced_data(k,:) = temp(:);
    end
        
    rng('default'); % Ensures replicability
    
    % Sorts data into 10 folds for cross-validation
    cvp = cvpartition(events,'KFold',10,'Stratify',true);
    
    % SVM analysis
    model_svm = fitcsvm(new_reduced, events, 'ClassNames',[double(0); double(1)], 'CVPartition',cvp);
    [~,score_svm] = kfoldPredict(model_svm); % Classifer scores for partitioned model
    [~,~,~,auc_svm] = perfcurve(events,score_svm(:,2),1); % Score for positive class in column 2
    subject_scores_svm(index) = auc_svm; % Stores final AUC value
end

%% Distribution of data
dist_svm = fitdist(subject_scores_svm(subject_scores_svm ~= 0), 'Normal');
