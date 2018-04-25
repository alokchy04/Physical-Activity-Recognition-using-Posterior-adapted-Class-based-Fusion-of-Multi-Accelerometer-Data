function sensor_fusion_Pamap2()
clc; 
for alok=1:1:2
    rng(3);
    clearvars('*', '-except', 'alok'); 
  
    saveOrLoad = 1;
    
    if(alok==1)
        isPAMAP = 0;
    else
        isPAMAP = 1;
    end
    
    if(saveOrLoad == 1)
        total_sensor = 3;
        if(isPAMAP == 1)
            path{1} = strcat(pwd, '\PAMAP2\');
            file{1} = 'feature_Ankle2_Merge_2s.csv';  
            name{1} = 'Ankle';

            path{2} = strcat(pwd, '\PAMAP2\');
            file{2}='feature_Chest2_Merge_2s.csv';   
            name{2} = 'Chest';

            path{3} = strcat(pwd, '\PAMAP2\');
            file{3}='feature_Hand2_Merge_2s.csv'; 
            name{3} = 'Hand';
        else
            path{1} = strcat(pwd, '\MHEALTH\');
            file{1} = 'Ankle2s.csv';  
            name{1} = 'Ankle';

            path{2} = strcat(pwd, '\MHEALTH\');
            file{2}='Chest2s.csv';   
            name{2} = 'Chest';

            path{3} = strcat(pwd, '\MHEALTH\');
            file{3}='Wrist2s.csv'; 
            name{3} = 'Hand';
        end

        classification_methods = {'KNN' 'SVM' 'RandomForest' 'Bagging' 'BDT' 'DNN' 'Adaboost'}; 
        classification_methods = {'SVM'};
        comb = [1 1 0; 1 0 1; 0 1 1; 1 1 1];
        
        for j=1:1:total_sensor
            fprintf('Processing Sensor ... %s\n',name{j});
            fileName=file{j};
            pathName=path{j};

            %open feature file
            [fetData, fetHeader, error] = read_CSV_File(strcat(pathName,fileName));
            label1=fetData(:,end); 
            IT = fetData(:,end-1);

            %getting userid 
            if(isPAMAP == 1)
                idnty=[];
                for k=1:1:size(IT,1)
                    temp = num2str(IT(k,1));
                    tnum = str2num(temp(1:1)); 
                    idnty =[idnty; tnum];
                end
            else
                idnty=[];
                for k=1:1:size(IT,1)
                    temp = num2str(IT(k,1));
                    tnum = str2num(temp(1:end-1)); 
                    idnty =[idnty; tnum];
                end
            end
            fetData=fetData(:,1:end-2);  

            %Remove NaN Rows  
            total = [fetData idnty label1];
            nanIndicator = ~any(isnan(total),2);
            total = total(nanIndicator,:);
            fetData = total(:,1:size(total,2)-size(label1,2)-1);
            idnty = total(:,end-1);
            label1 = total(:,size(total,2)-size(label1,2)+1:size(total,2));

            %Normalise 
            fetData1 = fetData;
            fetData = zscore(fetData,[ ],1); 

            %Feature Selection
            [data, dataHeader, label, minCorr, numoffeatures] = correlation_based_FS(fetData, fetHeader, label1, 0, 0.25);                  
            [data1, dataHeader1, label1, minCorr, numoffeatures] = correlation_based_FS(fetData1, fetHeader, label1, 0, 0.25); 

            %saving features for machine learning        
            options = unique(label);
            no_of_class = length(unique(label));  
            indices = idnty;
            loop = max(idnty);

            for k = 1:1:loop %fold                        
                fprintf('\tFold...%d\n',k);
                testIndex{k} = (indices == k); trainIndex{k} = ~testIndex{k};                
                testIndex1{k} = find(indices==k);
                testIndex{k} = logical(testIndex{k});
                trainIndex{k} = logical(trainIndex{k});       

                trainData = data(trainIndex{k},:);
                testData = data(testIndex{k},:);
                trainLabel = label(trainIndex{k},:);
                testLabel = label(testIndex{k},:);
                trainDataDNN = data1(trainIndex{k},:);
                testDataDNN = data1(testIndex{k},:);
                trainLabelDNN = label1(trainIndex{k},:);
                testLabelDNN = label1(testIndex{k},:);                
                
                
                if(~isempty(find(strcmpi(classification_methods, 'SVM'), 1)))
                    %apply SVM to sensor data
                        model = fitcecoc(trainData, trainLabel, 'FitPosterior',1); %,'Learners',t);
                        [prediction_SVM{j}{k}, ~, ~, posterior_SVM{j}{k}] = predict(model, testData);    
                        testStat_SVM{j}{k}=confusionmatStats(testLabel, prediction_SVM{j}{k});
                        testStat_SVM{j}{k}.scores=[prediction_SVM{j}{k} testLabel];

                        [trPred_SVM{j}{k}, trainStat_SVM{j}{k}, trainStat_SVM{j}{k}.scores]= my_kfold(trainData, trainLabel, 10, 'svm');   

                        newlabel_SVM{j}(testIndex{k},:) = prediction_SVM{j}{k};
                end
            end
        end
        if(isPAMAP == 1)
            save(strcat(pwd, '\PAMAP2\data_SVM.mat'));
        else
            save(strcat(pwd, '\MHEALTH\data_SVM.mat'));
        end
    else
        if(isPAMAP == 1)
            load(strcat(pwd, '\PAMAP2\data_SVM.mat'));
        else
            load(strcat(pwd, '\MHEALTH\data_SVM.mat'));
        end
    end
    
    for i = 1:1:size(comb,1)
        sensors{i} = [];
        for j = 1:1:total_sensor %sensor
            if(comb(i,j) == 1)
                sensors{i} = [sensors{i} name{j}];
            end
        end
    end
    
    for i = 1:1:size(comb,1)
        for k = 1:1:loop %fold 
            
            if(~isempty(find(strcmpi(classification_methods, 'SVM'), 1)))
                votes_SVM=[];
                posterior_of_votes_SVM=[];
                tr_votes_SVM=[];
                w_SVM=[];
                asw_SVM=[];         
            end
            
            count = 0;
            for j = 1:1:total_sensor %sensor
                if(comb(i,j) == 1)
                    count=count+1;
                    
                    if(~isempty(find(strcmpi(classification_methods, 'SVM'), 1)))
                        votes_SVM = [votes_SVM prediction_SVM{j}{k}];
                        
                        t=[];
                        for a=1:1:size(prediction_SVM{j}{k},1)
                            t=[t; posterior_SVM{j}{k}(a, prediction_SVM{j}{k}(a))];
                        end
                        posterior_of_votes_SVM = [posterior_of_votes_SVM t];
                        tr_votes_SVM = [tr_votes_SVM trPred_SVM{j}{k}];                    
                        w_SVM = [w_SVM computeWeight(trainStat_SVM{j}{k}.Fscore(size(trainStat_SVM{j}{k}.Fscore,1)-1,1))];
                        asw_SVM = [asw_SVM computeWeight(trainStat_SVM{j}{k}.Fscore(1:size(trainStat_SVM{j}{k}.Fscore,1)-2,1))]; 
                    end
                    
                    
                end
            end
            
            if(count > 0) %do fusion
                trainLabel = label(trainIndex{k},:);
                testLabel = label(testIndex{k},:);
                
                
                if(~isempty(find(strcmpi(classification_methods, 'SVM'), 1)))
                    w_SVM = normaliseWeight(w_SVM);
                    asw_SVM = normaliseWeight_colWise(asw_SVM);
                    %prediction_WMV_SVM{i}{k} = weighted_majority_voting(options, votes_SVM, w_SVM); 
                    %testStat_WMV_SVM{i}{k} = confusionmatStats(testLabel, prediction_WMV_SVM{i}{k});                
                    %newlabel_WMV_SVM{i}(testIndex{k},:) = prediction_WMV_SVM{i}{k};                   
                    %prediction_NB_SVM{i}{k} = nb_combiner(votes_SVM,tr_votes_SVM,trainLabel);
                    %testStat_NB_SVM{i}{k} = confusionmatStats(testLabel, prediction_NB_SVM{i}{k});                
                    %newlabel_NB_SVM{i}(testIndex{k},:) = prediction_NB_SVM{i}{k};                   
                    %prediction_BKS_SVM{i}{k} = bks_combiner(votes_SVM,tr_votes_SVM,trainLabel,options,w_SVM); 
                    %testStat_BKS_SVM{i}{k} = confusionmatStats(testLabel, prediction_BKS_SVM{i}{k});                
                    %newlabel_BKS_SVM{i}(testIndex{k},:) = prediction_BKS_SVM{i}{k};                        
                    %prediction_ASWMV_SVM{i}{k} = activity_specific_weighted_voting(options, votes_SVM, asw_SVM, w_SVM); 
                    alpha = 0.5;
                    beta = 0.5;
                    prediction_ASWMV_SVM{i}{k} = activity_specific_weighted_voting2(options, votes_SVM, asw_SVM, w_SVM, posterior_of_votes_SVM, alpha, beta); 
                    testStat_ASWMV_SVM{i}{k} = confusionmatStats(testLabel, prediction_ASWMV_SVM{i}{k});
                    newlabel_ASWMV_SVM{i}(testIndex{k},:) = prediction_ASWMV_SVM{i}{k};
                end
                
                
            end
        end
    end
    
    %save individual stats
    for j = 1:1:total_sensor %sensor 
        
        
        if(~isempty(find(strcmpi(classification_methods, 'SVM'), 1)))
        stat.(name{j}).SVM.fold = testStat_SVM{j};
        stat.(name{j}).SVM.final = confusionmatStats(label, newlabel_SVM{j});
        end
        
    end
    for i = 1:1:size(comb,1)
        for k = 1:1:loop %fold   
            
            if(~isempty(find(strcmpi(classification_methods, 'SVM'), 1)))
            %stat.(sensors{i}).SVM.WMV.fold = testStat_WMV_SVM{i};
            %stat.(sensors{i}).SVM.WMV.final = confusionmatStats(label, newlabel_WMV_SVM{i});            
            %stat.(sensors{i}).SVM.NB.fold = testStat_NB_SVM{i};
            %stat.(sensors{i}).SVM.NB.final = confusionmatStats(label, newlabel_NB_SVM{i});   
            %stat.(sensors{i}).SVM.BKS.fold = testStat_BKS_SVM{i};
            %stat.(sensors{i}).SVM.BKS.final = confusionmatStats(label, newlabel_BKS_SVM{i});            
            stat.(sensors{i}).SVM.PosterioAdaptedFusion.fold = testStat_ASWMV_SVM{i};
            stat.(sensors{i}).SVM.PosterioAdaptedFusion.final = confusionmatStats(label, newlabel_ASWMV_SVM{i});
            end
           
        end
    end
    if(isPAMAP == 1)
        save(strcat(pwd, '\PAMAP2\RES_stat_SVM_1_0.mat'),'stat');
    else        
        save(strcat(pwd, '\MHEALTH\RES_stat_SVM_1_0.mat'),'stat');
    end
end
end

function tempW1 = computeWeight(tempW)
%     tempW1=[];
%     for m=1:1:size(tempW,1)
%         if(tempW(m,1)==1.00)
%             tempW(m,1)=0.999999;
%         end
%         if(tempW(m,1)==1.00)
%             tempW(m,1)=0.000001;
%         end
% 
%         tempW1(m,1)=log(tempW(m,1)/(1-tempW(m,1)));
%     end
    
    tempW1=tempW;
end

function res = normaliseWeight(a)
    % row-wise
%     b = sum(a,2);
%     for i=1:1:size(a,1)
%         a(i,:)=a(i,:)./b(i,1);
%     end
    res=a;

end

function res = normaliseWeight_colWise(a)
    % col-wise
%     b = sum(a,1);
%     for i=1:1:size(a,2)
%         a(:,i)=a(:,i)./b(1,i);
%     end
    res=a;
end

function pred = activity_specific_weighted_voting(options, votes, w, overallw)
    pred=[];
    
    for i=1:1:size(votes,1) %instances        
        W=zeros(size(options,1),1);
        
        for j=1:1:size(votes,2) %sensors
            loc = find(options==votes(i,j));
            if(size(loc,2)>1||size(loc,1)>1)
                a=1;
            end
            W(loc,1) = W(loc,1) + w(loc,j);
        end
        tindex = find(W==max(W));
        t=options(tindex,1);
        sel_pred = t;
        
            if(size(t,2)>1||size(t,1)>1)
                maxm = -1;
                sel_pred = -1;
                for j=1:1:size(t,1)
                    if(size(find(votes(i,:)==t(j,1)),1)~=0)
                        if(overallw(1,find(votes(i,:)==t(j,1))) > maxm)
                            maxm = overallw(1,find(votes(i,:)==t(j,1)));
                            sel_pred = t(j,1);
                        end
                    end
                end
            end
        pred=[pred;sel_pred];
    end
end

function pred = activity_specific_weighted_voting2(options, votes, w1, overallw, w2, alpha, beta)
    pred=[];
    
    for i=1:1:size(votes,1) %instances        
        W=zeros(size(options,1),1);
        
        w=w1;
        for j=1:1:size(votes,2) %sensors
            loc = find(options==votes(i,j));
            
            w(loc,j)=(w1(loc,j)*alpha + w2(i,j)*beta); %combine
            
            W(loc,1) = W(loc,1) + w(loc,j);
        end
        tindex = find(W==max(W));
        t=options(tindex,1);
        sel_pred = t;
        
            if(size(t,2)>1||size(t,1)>1)
                maxm = -1;
                sel_pred = -1;
                for j=1:1:size(t,1)
                    if(size(find(votes(i,:)==t(j,1)),1)~=0)
                        if(overallw(1,find(votes(i,:)==t(j,1))) > maxm)
                            maxm = overallw(1,find(votes(i,:)==t(j,1)));
                            sel_pred = t(j,1);
                        end
                    end
                end
            end
        pred=[pred;sel_pred];
    end
end

function [trPred,trainStat,trainStatScore] = my_kfold(Data, Label, k, classifier)

    no_of_class = length(unique(Label));
    indices = crossvalind('Kfold', Label, k);
    loop = k;
    prediction = zeros(size(Label,1),size(Label,2));
    
    for k = 1:1:loop             
        testIndex = (indices == k); 
        trainIndex = ~testIndex;   
        testIndex = logical(testIndex);
        trainIndex = logical(trainIndex);
        
        trainData = Data(trainIndex,:);
        testData = Data(testIndex,:);
        trainLabel = Label(trainIndex,:);
        testLabel = Label(testIndex,:);
        
        if(strcmp(classifier,'knn'))
            model = fitcknn(...
                        trainData, ...
                        trainLabel, ...                        
                        'NumNeighbors', 7, ...
                        'Standardize',1);
           prediction(testIndex)  = predict(model, testData); 
           
        elseif(strcmp(classifier,'svm'))
            model = fitcecoc(trainData,trainLabel); %,'Learners',t);
            prediction(testIndex) = predict(model, testData);   
            
        elseif(strcmp(classifier,'randomforest'))
            nTrees = 20;
            model = TreeBagger(nTrees,trainData,trainLabel, 'Method', 'classification');
            prediction(testIndex) = str2double(predict(model, testData));
            
        elseif(strcmp(classifier,'bagging'))
            nTrees = 20;
            model = TreeBagger(nTrees,trainData,trainLabel, 'Method', 'classification', 'NumPredictorsToSample', 'all');
            prediction(testIndex) = str2double(predict(model, testData));
            
        elseif(strcmp(classifier,'bdt'))
            model = fitctree(...
                            trainData, ...
                            trainLabel, ...
                            'Surrogate', 'on');                    
                            %  'SplitCriterion', 'gdi', ...
                            %  'MaxNumSplits', 20, ...
                            %  'MaxNumSplits', 50, ...
            prediction(testIndex) = predict(model, testData);   
            
        elseif(strcmp(classifier,'adaboost'))
            learners = 'Discriminant'; %|| 'Tree' - REQUIRED FOR ADABOOST %learners = templateTree('Surrogate','on');
            if(no_of_class > 2)
                model = fitensemble(trainData, trainLabel, 'AdaboostM2', 100, learners);
            else
                model = fitensemble(trainData, trainLabel, 'AdaboostM1', 100, learners);                
            end
            prediction(testIndex) = predict(model, testData);   
                
        elseif(strcmp(classifier,'dnn'))
            level = 2;
            featureCount = [35; 20]; 

            % --Prepare-inputs-Start
            newtrainData = trainData';            
            newtestData = testData';

            temp = trainLabel';
            [c, ia, ic] = unique(temp,'sorted');
            newtrainLabel = zeros(size(c,2), size(trainLabel,1));
            for i=1:1:size(c,2)
                index = find(temp==c(1,i));
                newtrainLabel(i,index)=1;
            end
            % --Prepare-inputs-End

            tempFeatures = newtrainData;
            for i=1:1:level
                hiddenSize = featureCount(i,1);
                autoenc1 = trainAutoencoder(tempFeatures,hiddenSize,...
                    'L2WeightRegularization',0.001,...
                    'SparsityRegularization',4,...
                    'SparsityProportion',0.05,...
                    'MaxEpochs',1000,...
                    'DecoderTransferFunction','purelin');                
                tempFeatures = encode(autoenc1,tempFeatures); 
                % view(autoenc1);

                if(i==1)
                    deepnet = autoenc1;
                else 
                    deepnet = stack(deepnet, autoenc1);
                end
            end
            softnet = trainSoftmaxLayer(tempFeatures,newtrainLabel,'LossFunction','crossentropy',...
                    'MaxEpochs',1000);
            deepnet = stack(deepnet, softnet);
            % view(deepnet);

            deepnet = train(deepnet,newtrainData,newtrainLabel);
            tempprediction = deepnet(newtestData);

            % --Prepare-results-Start
            temp = vec2ind(tempprediction);
            tempprediction=zeros(size(testData,1),1);
            for i=1:1:size(c,2)
                index = find(temp==i);
                tempprediction(index,1)=c(1,i);
            end
            prediction(testIndex) = tempprediction; 
        end
        
    end
    trPred = prediction;   
    trainStat=confusionmatStats(Label, prediction); 
    trainStatScore=[trPred Label];  
end