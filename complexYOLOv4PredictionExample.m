%% Object Detection Using Complex Pretrained YOLO v4 Network
% The following code demonstrates running object detection on point clouds
% using a pretrained Complex YOLO v4 network, trained on Pandaset dataset.

%% Prerequisites
% To run this example you need the following prerequisites - 
%
% # MATLAB (R2021a or later) with Lidar and Deep Learning Toolbox.
% # Pretrained Complex YOLOv4 network (download instructions below).

%% Setup
% Add path to the source directory.
addpath('src');

%% Download the pre-trained network
% This repository uses two variants of Complex YOLO v4 models.
% *complex-yolov4-pandaset*
% *tiny-complex-yolov4-pandaset*
% Set the modelName from the above ones to download that pretrained model.
modelName = 'complex-yolov4-pandaset';
model = helper.downloadPretrainedYOLOv4(modelName);
net = model.net;

%% Detect Objects using Complex YOLO v4 Object Detector
% Read point cloud.
ptCld = pcread('pointclouds/0001.pcd');

% Get the configuration parameters.
gridParams = helper.getGridParameters;

% Get classnames of Pandaset dataset.
classNames = helper.getClassNames;

% Get the birds's-eye-view RGB map from the point cloud.
[img,ptCldOut] = helper.preprocess(ptCld, gridParams);

% Get anchors used in training of the pretrained model.
anchors = helper.getAnchors(modelName);

% Detect objects in test image.
executionEnvironment = 'auto';
[bboxes, scores, labels] = detectComplexYOLOv4(net, img, anchors, classNames, executionEnvironment);

% Display the results on an image.
figure
imshow(img)
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

% Display the results on point cloud.
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
