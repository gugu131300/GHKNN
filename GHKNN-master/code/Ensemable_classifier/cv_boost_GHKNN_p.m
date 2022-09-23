clear
seed = 123456789;
rand('seed', seed);
nfolds = 5; nruns=1;
load('C:\Users\vitus\Desktop\ADP/train_ADP.mat');
str='dd'
%datas = german_numer_scale;
datas=[train_PSSM_DCT,train_PRSA,train_y];
lammda =0.1;type = 'rbf';beta=0.1;
dim = size(datas,2);

train_X = datas(:,1:dim-1);

train_X = line_map(train_X);
train_label=datas(:,dim);
train_X(isnan(train_X)) = 0;

bestmcc=0;
bsetk=0;
bestg=0;


k_nn_l=10:10:200;
gamma_l=5;
gamma_l = 2.^gamma_l;
m_l=1:2:20;
MCC_M=size(length(k_nn_l),length(gamma_l));
ACC_M=size(length(k_nn_l),length(gamma_l));
SN_M=size(length(k_nn_l),length(gamma_l));
SP_M=size(length(k_nn_l),length(gamma_l));
value_M=[];
label_M=[];

for iter=1:length(m_l)
    m=m_l(iter);
    %m=3;
    for i=1:length(k_nn_l)
        for j=1:length(gamma_l)   
            crossval_idx = crossvalind('Kfold',train_label(:),nfolds);
        	k_nn =k_nn_l(i);
            gamma =gamma_l(j);
        	MCC_res=[];
            SN_res=[];
            SP_res=[];
            ACC_res=[];
            value=[];
            label=[];
            for fold=1:nfolds
        		train_idx = find(crossval_idx~=fold);
                test_idx  = find(crossval_idx==fold);
        		train_X_S = train_X(train_idx,:);
        		tr_y = train_label(train_idx);
        		test_X_S = train_X(test_idx,:);
        		te_y = train_label(test_idx);
        		[Predict_label_f,Score_f] = bootstrap_EnClassifierGHKNN_np(train_X_S,tr_y,test_X_S,te_y,k_nn,lammda,gamma,beta,type,1,m,0.5);

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
        	MCC_M(i,j) = meanMCC;
            ACC_M(i,j)=meanACC;
            SN_M(i,j)=meanSN;
            SP_M(i,j)=meanSP;
            value_M=[value_M,value];
            label_M=[label_M,label];
         fprintf('CV K_nn=%d,gamma=%d,lammda=%d beta=%d,m=%d,mCC=%f,SN=%f,SP=%f,ACC=%f',k_nn,gamma,lammda,beta,m,meanMCC,meanSN,meanSP,meanACC);
            disp("finished")
           dlmwrite('C:\Users\vitus\Desktop\test/value.csv',value_M);
          dlmwrite('C:\Users\vitus\Desktop\test/label.csv',label_M);
              dlmwrite('C:\Users\vitus\Desktop\test/SN.csv',SN_M);
              dlmwrite('C:\Users\vitus\Desktop\test/SP.csv',SP_M);
              dlmwrite('C:\Users\vitus\Desktop\test/ACC.csv',ACC_M);
              dlmwrite('C:\Users\vitus\Desktop\test/MCC.csv',MCC_M);
        end
    end

end



