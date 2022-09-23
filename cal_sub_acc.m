function [sub_acc,over_ACC] = cal_sub_acc(test_label,predict_y)

uniqlabels=unique(test_label);
c=max(size(uniqlabels));

sub_acc=zeros(c,1);

for i=1:c
	i_x_c = find(test_label==uniqlabels(i));
	number_i = length(test_label(i_x_c,:));
	right_i = length( find(predict_y(i_x_c,:)==test_label(i_x_c,:)) );
	[sub_acc(i)] = right_i/number_i;
	fprintf('class %d : %f (%d / %d ) \n', i, sub_acc(i)*100,right_i,number_i)

end


[over_ACC] = length( find(predict_y==test_label) )/length(test_label);
fprintf('whole acc: %f (%d / %d ) \n',  over_ACC*100,length( find(predict_y==test_label) ),length(test_label))

end