clear
seed = 123456789;
rand('seed', seed);
nfolds = 5; nruns=1;
load('C:\Users\vitus\Desktop\ ˝æ›ºØ\GDP/train_GDP.mat');
str='dd'
datas = [train_PSSM_DCT,train_PRSA,train_y];
dim = size(datas,2);

train_X = datas(:,1:dim-1);

train_X = line_map(train_X);
train_label=datas(:,dim);
train_X(isnan(train_X)) = 0;

bestmcc=0;
bsetk=0;
bestg=0;

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
        MCC_res=[MCC_res,MCC]
        SN_res=[SN_res,SN];
        SP_res=[SP_res,Spec];
        ACC_res=[ACC_res,ACC];

    end
        meanMCC=mean(MCC_res)
        meanSN=mean(SN_res);
        meanSP=mean(SP_res);
        meanACC=mean(ACC_res);
        ACC_M=[ACC_M,meanACC];
        MCC_M=[MCC_M,meanMCC];
        SN_M=[SN_M,meanSN];
        SP_M=[SP_M,meanSP];
        value_M=[value_M,value];
        label_M=[label_M,label];
        dlmwrite('C:\Users\vitus\Desktop\test/value.csv',value_M);
	    dlmwrite('C:\Users\vitus\Desktop\test/label.csv',label_M);
%         dlmwrite('C:\Users\vitus\Desktop\result\∫À‹’À·\train\ADP\KNN/SN.csv',SN_M);
%         dlmwrite('C:\Users\vitus\Desktop\result\∫À‹’À·\train\ADP\KNN/SP.csv',SP_M);
%         dlmwrite('C:\Users\vitus\Desktop\result\∫À‹’À·\train\ADP\KNN/ACC.csv',ACC_M);
%         dlmwrite('C:\Users\vitus\Desktop\result\∫À‹’À·\train\ADP\KNN/MCC.csv',MCC_M);
end



