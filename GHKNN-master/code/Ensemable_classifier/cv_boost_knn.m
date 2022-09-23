clear
seed = 123456789;
rand('seed', seed);
nfolds = 5; nruns=1;
load('C:\Users\vitus\Desktop\ADP/train_ADP.mat');
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

k_nn_l=1:3:30;
m_l=1:2:20;


MCC_M=[];
ACC_M=[];
SN_M=[];
SP_M=[];
value_M=[];
label_M=[];

for iter=1:length(m_l)
    m=m_l(iter);
    %m=3;
	for i=1:1%length(k_nn_l)
	    k_nn = k_nn_l(2);
	    MCC_res=[];
	    SN_res=[];
	    SP_res=[];
	    ACC_res=[];
        value=[];
        label=[];
        crossval_idx = crossvalind('Kfold',train_label(:),nfolds);
	    for fold=1:nfolds
			train_idx = find(crossval_idx~=fold);
	        test_idx  = find(crossval_idx==fold);
			train_X_S = train_X(train_idx,:);
			tr_y = train_label(train_idx);
			test_X_S = train_X(test_idx,:);
			te_y = train_label(test_idx);
	        [Predict_label_f,Score_f] = bootstrap_EnClassifierKNN_np( train_X_S, tr_y, test_X_S, te_y, k_nn,1,m,0.5);
			[ACC,SN,Spec,PE,NPV,F_score,MCC] =roc(Predict_label_f,te_y);
            value=[value;Score_f];
            label=[label;te_y];
	        MCC_res=[MCC_res,MCC];
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
            fprintf('CV K_nn=%d,m=%d,MCC=%d \n',k_nn,m,meanMCC);
        dlmwrite('C:\Users\vitus\Desktop\test/value.csv',value_M);
	    dlmwrite('C:\Users\vitus\Desktop\test/label.csv',label_M);
         
 	    dlmwrite('C:\Users\vitus\Desktop\test/SN.csv',SN_M);
 	    dlmwrite('C:\Users\vitus\Desktop\test/SP.csv',SP_M);
 	    dlmwrite('C:\Users\vitus\Desktop\test/ACC.csv',ACC_M);
 	    dlmwrite('C:\Users\vitus\Desktop\test/MCC.csv',MCC_M);
	end
end


