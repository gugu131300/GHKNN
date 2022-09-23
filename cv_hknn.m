clear
seed = 12345678;
rand('seed', seed);
nfolds = 5; nruns=1;
load('I:\exercise1\matlab代码\Gu_data.mat')
%load('I:\exercise1\matlab代码\349\X.mat');lammda =0.01;k_nn = 35;type = 'rbf'%0.9411
train_X = [X];
train_label = [labels];
lammda =0.01;k_nn = 35;type = 'rbf';
train_X = line_map(train_X);

ACC=[];SN=[];Spec=[];PE=[];NPV=[];F_score=[];MCC=[];
for run=1:nruns

    crossval_idx = crossvalind('Kfold',train_label(:),nfolds);
   
    for fold=1:nfolds
		train_idx = find(crossval_idx~=fold);
        test_idx  = find(crossval_idx==fold);
		train_X_S = train_X(train_idx,:);
		tr_y = train_label(train_idx);
		test_X_S = train_X(test_idx,:);
		te_y = train_label(test_idx);
		[predict_y,distance_s] = hknn(train_X_S,tr_y,test_X_S,k_nn,lammda);
		%[predict_y,distance_s] = hknn(train_X_S,train_label,test_X_S,k_nn,lammda);
		%[sub_acc,over_ACC] = cal_sub_acc(te_y,predict_y);
		%ACC=[over_ACC,ACC];
		te_y(find(te_y==2))=-1;
		predict_y(find(predict_y==2))=-1;
		 [ACC_i,SN_i,Spec_i,PE_i,NPV_i,F_score_i,MCC_i] = roc( predict_y,te_y );
		ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];
	end
	
end

mean_acc=mean(ACC)
mean_sn=mean(SN)
mean_sp=mean(Spec)
mean_mcc=mean(MCC)
