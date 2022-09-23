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
    opts.loss = 'exploss'
    opts.shrinkageFactor = 0.1;
    opts.subsamplingFactor = 0.8;
    opts.maxTreeDepth = uint32(5);  % this was the default before customization
    opts.randSeed = uint32(rand()*1000);
    options.mtry=uint32(ceil(sqrt(size(train_X_S,2))));
    numIters =300;
    tic;
    model = SQBMatrixTrain(single(train_X_S), tr_y, uint32(numIters), opts );
    tic;
    dec_values = SQBMatrixPredict( model, single(test_X_S) );
    positive =find(dec_values>0);
    negative=find(dec_values<0);
    predict_y=ones(length(dec_values),1);
    predict_y(positive,:)=1;
    predict_y(negative,:)=-1;
    [ACC,SN,Spec,PE,NPV,F_score,MCC] =roc1(predict_y,te_y);
    MCC_res=[MCC_res,MCC]
    SN_res=[SN_res,SN];
    SP_res=[SP_res,Spec];
    ACC_res=[ACC_res,ACC];

end
    meanMCC=mean(MCC_res)
    meanSN=mean(SN_res);
    meanSP=mean(SP_res);
    meanACC=mean(ACC_res);



