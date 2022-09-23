clear
seed = 12345678;
rand('seed', seed);
nfolds = 5; nruns=1;
load('D:\tju_workspace\C30E\MATLAB_WORK\Membrane_Protein\hknn\data\dataset1\D1.mat');lammda =0.1;k_nn = 100;type = 'rbf';gamma = [2^-6];beta=0.1;%0.9411
%load('D:\tju_workspace\C30E\MATLAB_WORK\Membrane_Protein\hknn\data\dataset2\D2.mat');lammda =0.1;k_nn = 150;type = 'rbf';gamma = [2^-6];beta=0.1;%0.9007
%load('D:\tju_workspace\C30E\MATLAB_WORK\Membrane_Protein\hknn\data\dataset3\D3.mat');lammda =0.1;k_nn = 100;type = 'rbf';gamma = [2^-5];beta=0.1;%0.8971
%load('D:\tju_workspace\C30E\MATLAB_WORK\Membrane_Protein\hknn\data\dataset4\D4.mat');lammda =0.1;k_nn = 100;type = 'rbf';gamma = [2^-5];beta=0.1;% 0.9901
train_X = [D1_train_PSSM_AB,D1_train_PSSM_DCT,D1_train_PSSM_DWT,D1_train_PSSM_HOG,D1_train_PSSM_Pse];%[D1_train_PSSM_AB,D1_train_PSSM_DCT,D1_train_PSSM_DWT,D1_train_PSSM_HOG,D1_train_PSSM_Pse];
lammda =0.1;k_nn = 100;type = 'rbf';gamma = [2^-6];beta=0.1;
train_X = line_map(train_X);

ACC=[];
for run=1:nruns
    % split folds
%    
    crossval_idx = crossvalind('Kfold',train_label(:),nfolds);
   

    for fold=1:nfolds
		train_idx = find(crossval_idx~=fold);
        test_idx  = find(crossval_idx==fold);
		train_X_S = train_X(train_idx,:);
		tr_y = train_label(train_idx);
		test_X_S = train_X(test_idx,:);
		te_y = train_label(test_idx);
		[predict_y,distance_s] = ghknn(train_X_S,tr_y,test_X_S,k_nn,lammda,gamma,beta,type);
		%[predict_y,distance_s] = hknn(train_X_S,train_label,test_X_S,k_nn,lammda);
		[sub_acc,over_ACC] = cal_sub_acc(te_y,predict_y);
		ACC=[over_ACC,ACC];
	end
	
end


meanACC = mean(ACC)
