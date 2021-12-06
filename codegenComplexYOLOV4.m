%% Code Generation for YOLO v4
% The following code demonstrates code generation for pre-trained Complex
% YOLO v4 object detection network, trained on Pandaset dataset.

%% Setup
% Add path to the source directory.
addpath('src');

%% Download the Pretrained Network
% This repository uses two variants of YOLO v4 models.
% *complex-yolov4-pandaset*
% *tiny-complex-yolov4-pandaset*
% Set the modelName from the above ones to download that pretrained model.
modelName = 'complex-yolov4-pandaset';
model = helper.downloadPretrainedYOLOv4(modelName);
net = model.net;

%% Read and Preprocess Input Point Cloud.
% Read point cloud.
ptCld = pcread('pointclouds/0001.pcd');

% Get the configuration parameters.
gridParams = helper.getGridParameters();

% Preprocess the input.
[I,ptCldOut] = helper.preprocess(ptCld, gridParams);

imgSize = [size(I,1),size(I,2)];
inputSize = net.Layers(1).InputSize;
scale = imgSize./inputSize(1:2);

% Provide location of the mat file of the trained network.
matFile = sprintf('models/complex-yolov4-models-master/models/complex-yolov4-pandaset/%s.mat',modelName);

%% Run MEX code generation
% The complexyolov4Predict.m is the entry-point function that takes an
% input image and give output for complex-yolov4-pandaset or
% tiny-complex-yolov4-pandaset models. The functions uses a persistent
% object yolov4obj to load the dlnetwork object and reuses that persistent
% object for prediction on subsequent calls.
%
% To generate CUDA code for the entry-point functions,create a GPU code 
% configuration object for a MEX target and set the target language to C++. 
% 
% Use the coder.DeepLearningConfig (GPU Coder) function to create a CuDNN 
% deep learning configuration object and assign it to the DeepLearningConfig 
% property of the GPU code configuration object. 
%
% Run the codegen command.
cfg = coder.gpuConfig('mex');
cfg.TargetLang = 'C++';
cfg.DeepLearningConfig = coder.DeepLearningConfig('cudnn');
args = {coder.Constant(matFile), single(I)};
codegen -config cfg complexYOLOv4Predict -args args -report

%% Run Generated MEX
% Call tiny-complex-yolov4-pandaset on the input image.
outFeatureMaps = complexYOLOv4Predict_mex(matFile,single(I));

% Get classnames of Pandaset dataset.
classNames = helper.getClassNames;

% Get anchors used in training of the pretrained model.
anchors = helper.getAnchors(modelName);

% Visualize detection results.
[bboxes,scores,labels] = helper.postprocess(outFeatureMaps, anchors, inputSize, scale, classNames);

figure
imshow(I)
showShape('rectangle',bboxes(labels=='Car',:),...
          'Color','green','LineWidth',0.5);hold on;
showShape('rectangle',bboxes(labels=='Truck',:),...
          'Color','magenta','LineWidth',0.5);
showShape('rectangle',bboxes(labels=='Pedestrain',:),...
          'Color','yellow','LineWidth',0.5);
hold off;

%% Project the bounding boxes to the point cloud
% Transfer labels to point cloud.
bboxCuboid = helper.transferbboxToPointCloud(bboxes,gridParams,ptCldOut);

figure
pcshow(ptCldOut.Location);
showShape('cuboid',bboxCuboid(labels=='Car',:),...
          'Color','green','LineWidth',0.5);hold on;
showShape('cuboid',bboxCuboid(labels=='Truck',:),...
          'Color','magenta','LineWidth',0.5);
showShape('cuboid',bboxCuboid(labels=='Pedestrain',:),...
          'Color','yellow','LineWidth',0.5);
hold off;

% Copyright 2021 The MathWorks, Inc.
