clear
seed = 123456789;
rand('seed', seed);
nfolds = 5; nruns=1;
%load('I:\exercise1\matlab´úÂë\349\labels.mat');
load('I:\exercise1\matlab´úÂë\Gu_data.mat');

train_X = [X];
train_label = [labels];
lammda =0.01;k_nn = 100;type = 'rbf';gamma = [2^-6];beta=0.1;
train_X = line_map(train_X);

k_nn_l=1:1:20;



MCC_M=[];
ACC_M=[];
SN_M=[];
SP_M=[];
label_M=[];
value_M=[];
crossval_idx = crossvalind('Kfold',train_label(:),nfolds);
for i=1:length(k_nn_l)
    k_nn =k_nn_l(i)
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
		model = fitcknn( train_X_S, tr_y, 'NumNeighbors',k_nn);
		[predict_y,dec_values]=predict(model,test_X_S);
		%[sub_acc,over_ACC] = cal_sub_acc(te_y,predict_y);
		%[over_ACC] = length( find(predict_y==te_y) )/length(te_y);
		%ACC=[over_ACC,ACC];

		[ACC,SN,Spec,PE,NPV,F_score,MCC] =roc(predict_y,te_y);
        value=[value;dec_values(:,2)];
        label=[label;te_y];
        MCC=[MCC_res,MCC]
        SN=[SN_res,SN];
        SP=[SP_res,Spec];
        ACC=[ACC_res,ACC];

    end
        meanMCC=mean(MCC)
        meanSN=mean(SN);
        meanSP=mean(SP);
        meanACC=mean(ACC);
        ACC_M=[ACC_M,meanACC];
        MCC_M=[MCC_M,meanMCC];
        SN_M=[SN_M,meanSN];
        SP_M=[SP_M,meanSP];
        value_M=[value_M,value];
        label_M=[label_M,label];

end

mean_acc=mean(ACC)
mean_sn=mean(SN)
mean_sp=mean(Spec)
mean_mcc=mean(MCC)

