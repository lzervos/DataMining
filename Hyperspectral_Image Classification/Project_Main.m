% This is a supporting MATLAB file for the project
clear
format compact
close all

load Salinas_hyperspectral %Load the Salinas hypercube called "Salinas_Image"
[p,n,l]=size(Salinas_Image); % p,n define the spatial resolution of the image, while l is the number of bands (number of features for each pixel)

load classification_labels 
% This file contains three arrays of dimension 22500x1 each, called
% "Training_Set", "Test_Set" and "Operational_Set". In order to bring them
% in an 150x150 image format we use the command "reshape" as follows:
Training_Set_Image=reshape(Training_Set, p,n); % In our case p=n=150 (spatial dimensions of the Salinas image).
Test_Set_Image=reshape(Test_Set, p,n);
Operational_Set_Image=reshape(Operational_Set, p,n);


%Depicting the various bands of the Salinas image
for i=1:l
    figure(1), imagesc(Salinas_Image(:,:,i))
    pause(0.05) % This command freezes figure(1) for 0.05sec. 
end

% Depicting the training, test and operational sets of pixels (for the
% pixels depicted with a dark blue color, the class label is not known.
% Each one of the other colors in the following figures indicate a class).
figure(2), imagesc(Training_Set_Image)
figure(3), imagesc(Test_Set_Image)
figure(4), imagesc(Operational_Set_Image)

% Constructing the 204xN array whose columns are the vectors corresponding to the
% N vectors (pixels) of the training set (similar codes cane be used for
% the test and the operational sets).
Operational=zeros(p,n,l);
Test=zeros(p,n,l);
Train=zeros(p,n,l); % This is a 3-dim array, which will contain nonzero values only for the training pixels
for i=1:l
     %Multiply elementwise each band of the Salinas_Image with the mask 
     % "Training_Set_Image>0", which identifies only the training vectors.
    Train(:,:,i)=Salinas_Image(:,:,i).*(Training_Set_Image>0);
    Test(:,:,i)=Salinas_Image(:,:,i).*(Test_Set_Image>0);
    Operational(:,:,i)=Salinas_Image(:,:,i).*(Operational_Set_Image>0);
    figure(5), imagesc(Train(:,:,i)) % Depict the training set per band
    pause(0.05)
end


close all;

Test_ar=[]; %This is the 204xN array
Test_array_response=[]; %Here we keep the label of the testing pixels
Test_array_pos=[]; %And here we have the 'position' of each testing pixel
Train_array=[]; %This is the wanted 204xN array
Train_array_response=[]; % This vector keeps the label of each of the training pixels
Train_array_pos=[]; % This array keeps (in its rows) the position of the training pixels in the image.
for i=1:p
    for j=1:n
        if(Training_Set_Image(i,j)>0) %Check if the (i,j) pixel is a training pixel
            Train_array=[Train_array squeeze(Train(i,j,:))];
            Train_array_response=[Train_array_response Training_Set_Image(i,j)];
            Train_array_pos=[Train_array_pos; i j];
        end
        if(Test_Set_Image(i,j)>0) %We check if it is a training pixel
            Test_ar=[Test_ar squeeze(Test(i,j,:))];
            Test_array_response=[Test_array_response Test_Set_Image(i,j)];
            Test_array_pos=[Test_array_pos; i j];
        end
           
    end
end
Datasets=cell(1,5);
classes=cell(1,5);              %Here we have the cell arrays that we use
Partition=cell(1,5);            %to construct the subsets for cross validation
Pixels=cell(1,5);
Tr_array=Train_array';


%%Here we partitioning the train set to the 5 class subsets
%from which we will construct the 5 susbets for cross validation
for i=1:5
    Datasets{1,i}={};
    Partition{1,i}={};
    Pixels{1,i}={};
end
for i=1:size(Tr_array,1)
    Partition{1,Train_array_response(1,i)}=[Partition{1,Train_array_response(1,i)}; Tr_array(i,:)];
    Pixels{1,Train_array_response(1,i)}=[Pixels{1,Train_array_response(1,i)}; Train_array_pos(i,:)];
end    

for i=1:5
    d=size(Partition{1,i},1);
    for j=1:5
        dv=fix(size(Partition{1,j},1)/5);
        for z=1:dv
           Datasets{1,i}=[Datasets{1,i};Partition{1,j}{i*z}];
           classes{1,i}=[classes{1,i};j];
        end
    end
    for c=(d-mod(d,5)+1):d
        Datasets{1,i}=[Datasets{1,i};Partition{1,i}{c}];
        classes{1,i}=[classes{1,i};i];
    end
end   

%%Here are the 5 subsets we will use for cross validation
A=cell2mat(Datasets{1,1});
cl_a=classes{1,1};
B=cell2mat(Datasets{1,2});
cl_b=classes{1,2};
C=cell2mat(Datasets{1,3});
cl_c=classes{1,3};
D=cell2mat(Datasets{1,4});
cl_d=classes{1,4};
E=cell2mat(Datasets{1,5});
cl_e=classes{1,5};

%These are the subsets that we use for training
E_A=cat(1,B,C,D,E);
E_B=cat(1,A,C,D,E);
E_C=cat(1,A,B,D,E);
E_D=cat(1,A,B,C,E);
E_E=cat(1,A,B,C,D);

mn=2; %We initialize it with 2 because the error can be maximum 1
k=1; %A random value to k it doesn't matter

%Here is the cross validation with the 5 classifiers 
for i=1:2:17
    clf1=fitcknn(E_A,[cl_b;cl_c;cl_d;cl_e],'NumNeighbors',i);
    clf2=fitcknn(E_B,[cl_a;cl_c;cl_d;cl_e],'NumNeighbors',i);
    clf3=fitcknn(E_C,[cl_b;cl_a;cl_d;cl_e],'NumNeighbors',i);
    clf4=fitcknn(E_D,[cl_b;cl_c;cl_a;cl_e],'NumNeighbors',i);
    clf5=fitcknn(E_E,[cl_b;cl_c;cl_d;cl_a],'NumNeighbors',i);
    
    l1=predict(clf1,A);
    l2=predict(clf2,B);
    l3=predict(clf3,C);
    l4=predict(clf4,D);
    l5=predict(clf5,E);
    
    %These are the errors for every classifier of the 5
    e1=sum(cl_a-l1(:)~=0);
    e2=sum(cl_b-l2(:)~=0);
    e3=sum(cl_c-l3(:)~=0);
    e4=sum(cl_d-l4(:)~=0);
    e5=sum(cl_e-l5(:)~=0);
    
    %Here is the mean error for every k
    err=(e1/size(l1,1)+e2/size(l2,1)+e3/size(l3,1)+e4/size(l4,1)+e5/size(l5,1))/5;
    fprintf('Error for k=%d is %f\n',i,err);
    
    %Holding k with the smallest mean error
    if (err<mn)
        mn=err;
        k=i;
    end    
end
fprintf('\n');

%Here we are 'fitting' our classifiers
KNN=fitcknn(Tr_array,Train_array_response,'NumNeighbors',k);
eu_cl=Eucl_Class(Tr_array,Train_array_response); %It has as default the euclidean distance and the neighbors 1
Naive_Bayes=fitcnb(Tr_array,Train_array_response); %By default the Naive_Bayes in matlab 'uses' normal distribution

%Here we are making the predictions
label_KNN=predict(KNN,Test_ar');
label_Eucl=predict(eu_cl,Test_ar');
label_Bayes=predict(Naive_Bayes,Test_ar');

%Here we are taking the confusion matrices
KNN_conf=confusionmat(Test_array_response,label_KNN);
Eucl_conf=confusionmat(Test_array_response,label_Eucl);
Bayes_conf=confusionmat(Test_array_response,label_Bayes);

%Here we are computing the accuracy of each classifier from its confusion
%matrix(sum of diagonal elements/sum of the whole array)
acc_KNN=trace(KNN_conf)/sum(KNN_conf(:))
acc_Eucl=trace(Eucl_conf)/sum(Eucl_conf(:))
acc_Bayes=trace(Bayes_conf)/sum(Bayes_conf(:))

%This is the array for the All_In_One image
total_cl=cat(1,label_KNN,label_Eucl,label_Bayes);

%If we sort them then we have the images with postfix _2
%If we do it this way,then we can see more clearly 
%the number of pixels that belong to each class

%label_KNN=sort(label_KNN);
%label_Eucl=sort(label_Eucl);
%label_Bayes=sort(label_Bayes);
%total_cl=sort(total_cl);

%And here we are 'making' the images
figure('Name','All_In_One','NumberTitle','off'),imagesc(total_cl)
colorbar

figure('Name','KNN','NumberTitle','off'),imagesc(label_KNN)
colorbar

figure('Name','Euclidean','NumberTitle','off'),imagesc(label_Eucl)
colorbar

figure('Name','Naive_Bayes','NumberTitle','off'),title('Naive_Bayes'),imagesc(label_Bayes)
colorbar





