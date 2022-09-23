clear
seed = 123456789;
rand('seed', seed);
nfolds = 5; nruns=1;
load('C:\Users\vitus\Desktop\Êý¾Ý¼¯\AMP/train_AMP.mat');
load('C:\Users\vitus\Desktop\Êý¾Ý¼¯\AMP/test_AMP.mat');
train_datas = [train_PSSM_DCT,train_PRSA,train_y];
test_datas=[test_PSSM_DCT,test_PRSA,test_y];

train_dim = size(train_datas,2);
train_X = train_datas(:,1:train_dim-1);
train_label=train_datas(:,train_dim);

test_dim = size(test_datas,2);
test_X = test_datas(:,1:test_dim-1);
test_label=test_datas(:,test_dim);

COM_X = [train_X;test_X];
COM_X = line_map(COM_X);
train_end = size(train_label,1);
test_strat = train_end + 1;
train_X_S = COM_X(1:train_end,:);
test_X_S = COM_X(test_strat:end,:);

train_X_S(isnan(train_X_S)) = 0;
test_X_S(isnan(test_X_S)) = 0;
lammda =0.1;k_nn = 100;
[predict_y,distance_s,score_f] = hknn(train_X_S,train_label,test_X_S,k_nn,lammda);
[ACC,SN,Spec,PE,NPV,F_score,MCC] =roc(predict_y,test_label);