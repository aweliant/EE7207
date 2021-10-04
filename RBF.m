%data preprocessing
% data_train=(mapstd(data_train'))';
% data_test=(mapstd(data_test'))';

% %determine hyperparameters through K-fold cross validation
K=5;
indices = crossvalind('Kfold', 330, K);
start=30;
stop=130;
step=10;
score=zeros(2,floor((stop-start)/step)+1);
for N=start:step:stop
    perf=zeros(1,K);
    accuracy=zeros(1,K);
    for i=1:K
        val=(indices==i);
        tr=~val;
        data_tr=data_train(tr,:);
        data_val=data_train(val,:);
        label_tr=label_train(tr,:);
        label_val=label_train(val,:);
        P=data_tr';
        T=label_tr';
        distance=dist(P);
        dm=max(max(distance));
        spread=sqrt(2*330/K*(K-1))/dm;
        net=newrb(P,T,0.00,spread,N,10);
        y=sim(net,data_val');
        perf(i)=perform(net,label_val',y);
        y=((y>=0)-0.5)*2;
        acc=sum(label_val'==y)/length(y)
        accuracy(i)=acc;
    end
    score(1,round((N-start)/step)+1)=mean(perf(:));
    score(2,round((N-start)/step)+1)=mean(accuracy(:));
end
[minMSE, location1]=min(score(1,:));
[maxACC, location2]=max(score(2,:));

%evaluate method by K-fold CV
K=5;
indices = crossvalind('Kfold', 330, K);
accuracy=zeros(1,K);
recall=zeros(1,K);
precision=zeros(1,K);
f1=zeros(1,K);
for i=1:K
    tp=0;
    fn=0;
    fp=0;
    tn=0;
    val=(indices==i);
    tr=~val;
    data_tr=data_train(tr,:);
    data_val=data_train(val,:);
    label_tr=label_train(tr,:);
    label_val=label_train(val,:);
    P=data_tr';
    T=label_tr';
    distance=dist(P);
    dm=max(max(distance));
    spread=sqrt(2*330/K*(K-1))/dm;
    net=newrb(P,T,0.00,spread,60,1);
    y=sim(net,data_val');
    y=((y>=0)-0.5)*2;
    acc=sum(label_val'==y)/length(y);
    accuracy(i)=acc;
    for j=1:length(label_val)
        if label_val(j)==1 && y(j)==1
            tp=tp+1
        elseif label_val(j)==1 && y(j)==-1
            fn=fn+1
        elseif label_val(j)==-1 && y(j)==1
            fp=fp+1
        elseif label_val(j)==-1 && y(j)==-1
            tn=tn+1
        end
    end
    precision(i) = tp/(tp+fp);
    recall(i)=tp/(tp+fn);
    f1(i)=2*precision(i)*recall(i)/(precision(i)+recall(i));
end
mean_acc=mean(accuracy);
mean_f1=mean(f1);

%train a best model and generate prediction for test data
P=data_train';
T=label_train';
distance=dist(P);
dm=max(max(distance));
spread=sqrt(2*330/K*(K-1))/dm;
net=newrb(data_train',label_train',0.00,spread,60,1);
predict=sim(net,data_test')';
predict=((predict>=0)-0.5)*2;