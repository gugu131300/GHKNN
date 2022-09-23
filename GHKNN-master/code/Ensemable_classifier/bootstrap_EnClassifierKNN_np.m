function [Predict_label_f,Score_f] = bootstrap_EnClassifierKNN_np( train_feature, train_label, test_feature, test_label, k_nn,rate, m,threshold_T)
%This function is based on LIBSVM toolbox.
%Tianjin University
% train_feature: - An MxN array, the ith instance of training instance is stored in train_feature(i,:). M is the number of sample. N is the dimensions of feature vector
% train_label: - An Mx1 array, "1" Positive samples, "-1" Negative samples. M is the number of sample.
% test_feature: - An MxN array, the ith instance of test instance is stored in test_feature(i,:). M is the number of sample. N is the dimensions of feature vector
% test_label: - An Mx1 array, "1" Positive samples, "-1" Negative samples. M is the number of sample.

% rate: - the rateo of negative sample of ramdom under-sampling (default: 1)
% m: - the numbers of base Classifier 
% threshold_T : - the threshold of decision probability

%divide positive and negative samples from train_feature
max_label = max(train_label);
min_label = min(train_label);

P_feature_train=[];
N_feature_train=[];

P_COMB_train=[];
N_COMB_train=[];
COMB_LABEL_train = [train_label, train_feature];
Num_sample_train = size(COMB_LABEL_train,1);
Num_feature = size(train_feature,2);


Pos_s =find(train_label==max_label);
Neg_s = find(train_label==min_label);
P_feature_train = train_feature(Pos_s,:);
N_feature_train = train_feature(Neg_s,:);

Num_sample_train_Positive = size(P_feature_train,1);
Num_sample_train_Negative = size(N_feature_train,1);

%share negative train samples
neg_rate = (Num_sample_train_Positive*1.0)/Num_sample_train_Negative;
Num_share = m;
Num_sample_train_Positive = floor(Num_sample_train_Positive*1.0);
Num_sample_train_Negative = floor(Num_sample_train_Negative*neg_rate*rate);
%vv='training and voting'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%train  &  test

F_Test = [];Score_ff=[];
for i=1:Num_share
	sub_train_positive_label_1=ones(Num_sample_train_Positive,1);
	sub_train_negative_label_1=min_label*ones(Num_sample_train_Negative,1);
	sub_train_label_1 = [];
	sub_train_label_1=[sub_train_positive_label_1;sub_train_negative_label_1];

	

	
	ram_P = [];
	ram_P=randperm(size(P_feature_train,1));
	P_dataall=[];temp_P_dataall=[];
	P_dataall = P_feature_train;
	temp_P_dataall = P_feature_train;
	P_A = P_dataall(ram_P(1:size(temp_P_dataall,1)),:);
	P_B = P_A(1:Num_sample_train_Positive,:);
	clear P_A;
	clear temp_P_dataall;
	clear P_dataall;
	clear ram_P;
	
	
	ram_N = [];
	ram_N=randperm(size(N_feature_train,1));
	N_dataall=[];temp_N_dataall=[];
	N_dataall = N_feature_train;
	temp_N_dataall = N_feature_train;
	N_A = N_dataall(ram_N(1:size(temp_N_dataall,1)),:);
	N_B = N_A(1:Num_sample_train_Negative,:);
	clear N_A;
	clear temp_N_dataall;
	clear N_dataall;
	clear ram_N;
	
	sub_train_feature_1 = [P_B;N_B];
	clear N_B;
	clear P_B;
	
	model1 = [];
%	model1=svmtrain(sub_train_label_1,sub_train_feature_1,CG_parameter);
%	[Predict_label_f1,accuracy1,dec_values1]=svmpredict(test_label,test_feature,model1,'-b 1');
	model = fitcknn( sub_train_feature_1, sub_train_label_1, 'NumNeighbors',k_nn);
	[Predict_label_f1, dec_values1] =predict(model,test_feature);

% 	accnum=0;SN=0;TP=0;
% 	for j=1:size(test_label,1)
% 		if(test_label(j,1)==Predict_label_f1(j,1))
% 			accnum=accnum+1;
% 			if (test_label(j,1)==max_label)
% 				TP=TP+1;
% 			end
% 		end
% 	end
% 	Acc = accnum/size(test_label,1);
	%SN = TP/sum(test_label==1)
	
    [ACC,SN,Spec,PE,NPV,F_score,MCC] = roc( Predict_label_f1,test_label );
	F_Test = [F_Test,Predict_label_f1];
	Score_ff=[Score_ff,dec_values1(:,2)];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



YES_Num=zeros(size(test_feature,1),1);
NO_Num=zeros(size(test_feature,1),1);


%vote
Predict_label_f = zeros(size(test_feature,1),1);
Score_f=[];
%Score_f = mean(Score_ff')';
for i=1:size(Score_ff,1)
	mms=[];
	mms = Score_ff(i,:);
	meem = mean(mms);
	Score_f = [Score_f;meem];
end
for ii=1:size(test_feature,1)
	Sc = Score_f(ii,1);
	if Sc>threshold_T
		PP_ = max_label;
		Predict_label_f(ii)=PP_;
	else
		PP_ = min_label;
		Predict_label_f(ii)=PP_;
	end
end




% accnum=0;SN=0;TP=0;
% for i=1:size(test_label,1)
%     if(test_label(i,1)==Predict_label_f(i,1))
%         accnum=accnum+1;
% 		if (test_label(i,1)==max_label)
% 			TP=TP+1;
% 		end
%     end
% end
% Acc = accnum/size(test_label,1)
% SN = TP/sum(test_label==max_label)


[ACC,SN,Spec,PE,NPV,F_score,MCC] = roc( Predict_label_f,test_label );

