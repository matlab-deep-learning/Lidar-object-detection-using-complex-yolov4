function bboxCuboid = transferbboxToPointCloud(bboxes,gridParams,ptCldOut)
% Transfer labels from images to point cloud.

% Copyright 2021 The MathWorks, Inc.

    % Calculate the height of the ground plane.
    groundPtsIdx = segmentGroundSMRF(ptCldOut,3,'MaxWindowRadius',5,'ElevationThreshold',0.4,'ElevationScale',0.25);
    loc = ptCldOut.Location;
    groundHeight = mean(loc(groundPtsIdx,3));
    
    % Assume height of objects to be a constant based on input data.
    objectHeight = 1.56;
    
    % Transfer Labels back to the point cloud.
    bboxCuboid = zeros(size(bboxes,1),9);
    bboxCuboid(:,1) = (bboxes(:,2) - 1 - gridParams.bevHeight/2)*gridParams.gridH;
    bboxCuboid(:,2) = (bboxes(:,1) - 1 )*gridParams.gridW;
    bboxCuboid(:,4) = bboxes(:,4)*gridParams.gridH;
    bboxCuboid(:,5) = bboxes(:,3)*gridParams.gridW;
    bboxCuboid(:,9) = -bboxes(:,5);
    
    bboxCuboid(:,6) = objectHeight;
    bboxCuboid(:,3) = groundHeight + (objectHeight/2);
end