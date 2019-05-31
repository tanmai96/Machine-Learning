function testRFC

%this test file is designed for testing every test image of every data set
%and storing the proposed method segmented image in RFC set\Dataset\rfc_output
%where Dataset denotes corresponding data set like DRIVE/STARE/CSHASEDB1

%please add multi-scales's test.m file in the working directory

%load trained model if you want to test with previously trained model
load('trainedRFCModel_H_128')

%default SURF parameters
%Options.upright=true;
%Options.tresh=0.000001;


datasets = {'HRF','DRIVE','STARE','CHASEDB1'};
extensions = {'\*.jpg','\*.tif','\*.ppm','\*.jpg'};
extensionsMask = {'\*.tif','\*.gif','\*.png','\*.jpg'};
path = 'Images\RFC SET';

%%
%testing all test images of all datasets
for dset =1:length(datasets)
    
    %file information for fundus images, mask for multi-scale
    files = dir (fullfile(path,datasets{dset},'test',extensions{dset}));
    files1 = dir (fullfile(path,datasets{dset},'multiscale_mask','\test',extensionsMask{dset}));
    
    
    for iN = 1:length(files)
        
        %%
        %multi-scale line operation on fundus image
        img=imread(fullfile(files(iN).folder,'\',files(iN).name));
        mask = imread(fullfile(files1(iN).folder,'\',files1(iN).name));
        if dset==1
            img = imresize(img, [1000 1000]);
            mask = imresize(img, [1000 1000]);
            
        end
        mSegmentedImg = multi_test(img,mask);
        fName = ['segmented_',strtok(files(iN).name,'.'),'.png'];
        
        %imwrite(mSegmentedImg,fullfile(path,datasets{dset},'test\multiscale_output',fName));
        %%
        
        %%
        %feature extraction for predicting class
        
        %initialize an empty matrix of test image sizes
        [row col] =size(mSegmentedImg);
        new_img1=zeros(row,col);
        
        %for SURF
        %feat = [];
        
        %Ipts1=OpenSurf_Sheen(mSegmentedImg,Options,mSegmentedImg);
        %for i=1:length(Ipts1)
        %    feat(i,:) = Ipts1(1,i).descriptor;
        %end
        
        %for LHP
        [interest_points, featuresH]=create_descriptor(img,mSegmentedImg,32);
        
        %Class prediction for test tuples
        %class=Mdl.predict(feat);
        class=predict(Mdl, featuresH);
        %generate new image with the class result in corresponding pixel
        %locations
        %for i=1:length(Ipts1)
            
         %   new_img1(Ipts1(1,i).y,Ipts1(1,i).x) = str2double(class{i});
            
        %end
        for i=1:length(interest_points)
            
            new_img1(interest_points(i,1),interest_points(i,2)) = str2double(class(i));
            
        end
        %imshow(new_img1);
        
        fName = ['segmented_',strtok(files(iN).name,'.'),'_RFC','.png'];
        path2 = strcat('F:\Thesis_output\Random Forest','\',datasets{dset});
        p1=strcat(path2,'\','test');
        imwrite(new_img1,fullfile(p1,fName));
        
        % imwrite(new_img1,'E:\Thesis\Code\Code\Images\RFC Output\DRIVE\08_test_segmented_RFC.png');
        %%
        
    end
    
end
%%
end
