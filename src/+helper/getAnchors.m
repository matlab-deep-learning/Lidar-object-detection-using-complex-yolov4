function anchors = getAnchors(modelName)
% The getAnchors function returns the anchors used in training of the 
% specified Complex YOLO v4 model.
%
% Copyright 2021 The MathWorks, Inc.

if isequal(modelName, 'complex-yolov4-pandaset')
    anchors.anchorBoxes = [10 10; 11 13; 22 35; ...
                           23 48; 23 55; 25 56; ...
                           25 62; 27 71; 35 95];
    anchors.anchorBoxMasks = {[1,2,3]
                             [4,5,6]
                             [7,8,9]};
elseif isequal(modelName, 'tiny-complex-yolov4-pandaset')
    anchors.anchorBoxes = [36 118; 24 59; 23 50; ...
                           11 14; 10 10; 5 4];
    anchors.anchorBoxMasks = {[1,2,3]
                             [4,5,6]};
end
end
