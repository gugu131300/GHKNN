clear
seed = 12345678;
rand('seed', seed);
nfolds = 5; nruns=1;

load('I:\exercise1\paper1\代码\matlab代码\Gu_data.mat');
train_X = [X];
train_label = [labels];
model_ori = fitcsvm(train_X, train_label,'Holdout',0.1);
model_new = model_ori.Trained{1};

testInd = test(model_ori.Partition);
dataTest = train_X(testInd,:);
labelTest = train_label(testInd,:);
labelpredict = predict(model_new, dataTest);

table(labelTest,labelpredict);
ACC=[];SN=[];Spec=[];PE=[];NPV=[];F_score=[];MCC=[];
[ACC_i,SN_i,Spec_i,PE_i,NPV_i,F_score_i,MCC_i] = roc(labelpredict,labelTest );
ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];

mean_acc=mean(ACC)
mean_sp=mean(Spec)
mean_mcc=mean(MCC)
mean_sn=mean(SN)