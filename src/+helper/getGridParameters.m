function gridParams = getGridParameters()
% The getGridParameters function returns the grid parameters that controls
% the range of point cloud.
%
% Copyright 2021 The MathWorks, Inc.

    xMin = -25.0;     % Minimum value along X-axis.
    xMax = 25.0;      % Maximum value along X-axis.
    yMin = 0.0;       % Minimum value along Y-axis.
    yMax = 50.0;      % Maximum value along Y-axis.
    zMin = -7.0;      % Minimum value along Z-axis.
    zMax = 15.0;      % Maximum value along Z-axis.

    pcRange = [xMin xMax yMin yMax zMin zMax];

    % Define the dimensions for the pseudo-image.
    bevHeight = 608;
    bevWidth = 608;

    % Find grid resolution.
    gridW = (pcRange(4) - pcRange(3))/bevWidth;
    gridH = (pcRange(2) - pcRange(1))/bevHeight;
   
    gridParams.xMin = xMin;
    gridParams.xMax = xMax;
    gridParams.yMin = yMin;
    gridParams.yMax = yMax;
    gridParams.zMin = zMin;
    gridParams.zMax = zMax;
    
    gridParams.bevHeight = bevHeight;
    gridParams.bevWidth = bevWidth;
    
    gridParams.gridH = gridH;
    gridParams.gridW = gridW;    
end