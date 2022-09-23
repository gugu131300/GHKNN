clear
seed = 1;
rand('seed', seed);
nfolds = 5; nruns=1;
load('C:\Users\vitus\Desktop\Êý¾Ý¼¯\ADP/train_ADP.mat');
str='dd'
datas = [train_PSSM_DCT,train_PRSA,train_y];
lammda =0.1;k_nn = 100;type = 'rbf';gamma = [2^-6];beta=0.1;%0.9411
dim = size(datas,2);

train_X = datas(:,1:dim-1);

train_X = line_map(train_X);
train_label=datas(:,dim);
train_label(train_label==0)=-1;
train_X(isnan(train_X)) = 0;

bestmcc=0;
bsetk=0;
bestg=0;
crossval_idx = crossvalind('Kfold',train_label(:),nfolds);
MCC_res=[];
SN_res=[];
SP_res=[];
ACC_res=[];
label=[];
value=[];
for fold=1:nfolds
    train_idx = find(crossval_idx~=fold);
    test_idx  = find(crossval_idx==fold);
    train_X_S = train_X(train_idx,:);
    tr_y = train_label(train_idx);
    test_X_S = train_X(test_idx,:);
    te_y = train_label(test_idx);
    model=classRF_train(train_X_S,tr_y);%Model of random forest generation 
    [predict_y,dec_values,prediction_pre_tree]=classRF_predict(test_X_S,model,300);%Forecast and generate  
    %ACC=[over_ACC,ACC];

    [ACC,SN,Spec,PE,NPV,F_score,MCC] =roc1(predict_y,te_y);
    value=[value;dec_values(:,2)];
    label=[label;te_y];
    MCC_res=[MCC_res,MCC]
    SN_res=[SN_res,SN];
    SP_res=[SP_res,Spec];
    ACC_res=[ACC_res,ACC];

end
    meanMCC=mean(MCC_res)
    meanSN=mean(SN_res);
    meanSP=mean(SP_res);
    meanACC=mean(ACC_res);



