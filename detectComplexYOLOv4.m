function [bboxes, scores, labels] = detectComplexYOLOv4(dlnet, image, anchors, classNames, executionEnvironment)
% detectComplexYOLOv4 runs prediction on a trained complex yolov4 network.
%
% Inputs:
% dlnet                - Pretrained complex yolov4 dlnetwork.
% image                - BEV image to run prediction on. (H x W x 3)
% anchors              - Anchors used in training of the pretrained model.
% classNames           - Classnames to be used in detection.
% executionEnvironment - Environment to run predictions on. Specify cpu,
%                        gpu, or auto.
%
% Outputs:
% bboxes     - Final bounding box detections ([x y w h rot]) formatted as
%              NumDetections x 5.
% scores     - NumDetections x 1 classification scores.
% labels     - NumDetections x 1 categorical class labels.

% Copyright 2021 The MathWorks, Inc.

% Get the input size of the network.
inputSize = dlnet.Layers(1).InputSize;

% Process the input image.
imgSize = [size(image,1),size(image,2)];
image =  im2single(imresize(image,inputSize(:,1:2)));
scale = imgSize./inputSize(1:2);

% Convert to dlarray.
dlInput = dlarray(image, 'SSCB');

% If GPU is available, then convert data to gpuArray.
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlInput = gpuArray(dlInput);
end

% Perform prediction on the input image.
outFeatureMaps = cell(length(dlnet.OutputNames), 1);
[outFeatureMaps{:}] = predict(dlnet, dlInput);

% Apply postprocessing on the output feature maps.
[bboxes,scores,labels] = helper.postprocess(outFeatureMaps, anchors, ...
                                            inputSize, scale, classNames);
end