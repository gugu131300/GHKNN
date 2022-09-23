clear
%addpath(genpath('/home/yangchao/prediction/libsvm-3.22'));
seed = 1;
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


index=-6:1:6;
gamma_l = 2.^index;
index1=1:1:5;
c_l=2.^index1;

MCC_M=size(length(c_l),length(gamma_l));
ACC_M=size(length(c_l),length(gamma_l));
SN_M=size(length(c_l),length(gamma_l));
SP_M=size(length(c_l),length(gamma_l));

crossval_idx = crossvalind('Kfold',train_label(:),nfolds);
for i=1:length(c_l)
    for j=1:length(gamma_l)   
	    %crossval_idx = crossvalind('Kfold',train_label(:),nfolds);
		%crossval_idx = crossvali_func(train_label(:),nfolds);
	    
		C= c_l(i);
	    Gamma = gamma_l(j);
	    CG_param=['-c ',num2str(C),' -g ',num2str(Gamma),' -b ',num2str(1),' -q']
		MCC_res=[];
	    SN_res=[];
	    SP_res=[];
	    ACC_res=[];
	    for fold=1:nfolds
			train_idx = find(crossval_idx~=fold);
	        test_idx  = find(crossval_idx==fold);
			train_X_S = train_X(train_idx,:);
			tr_y = train_label(train_idx);
			test_X_S = train_X(test_idx,:);
			te_y = train_label(test_idx);
			%[predict_y,distance_s] = ghknn(train_X_S,tr_y,test_X_S,k_nn,lammda,gamma,beta,type);
	        model1 = [];
			model1=svmtrain(tr_y,train_X_S,CG_param);
			[predict_y,accuracy1,dec_values1]=svmpredict(te_y,test_X_S,model1,'-b 1');
			[ACC,SN,Spec,PE,NPV,F_score,MCC] =roc(predict_y,te_y);
	        
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
	    fprintf('CV C=%d,Gamma=%d \n',C,Gamma);
	    disp("finished")
	    dlmwrite('C:\Users\vitus\Desktop\result\∫À‹’À·\train\GDP\SVM/SN_1.csv',SN_M);
	    dlmwrite('C:\Users\vitus\Desktop\result\∫À‹’À·\train\GDP\SVM/SP_1.csv',SP_M);
	    dlmwrite('C:\Users\vitus\Desktop\result\∫À‹’À·\train\GDP\SVM/ACC_1.csv',ACC_M);
	    dlmwrite('C:\Users\vitus\Desktop\result\∫À‹’À·\train\GDP\SVM/MCC_1.csv',MCC_M);
    end

end

  