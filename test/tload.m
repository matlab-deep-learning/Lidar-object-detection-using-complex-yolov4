classdef(SharedTestFixtures = {DownloadComplexYolov4Fixture}) tload < matlab.unittest.TestCase
    % Test for loading the downloaded models.
    
    % Copyright 2021 The MathWorks, Inc.
    
    % The shared test fixture DownloadYolov4Fixture calls
    % downloadPretrainedYOLOv4. Here we check the properties of
    % downloaded models.
    
    properties        
        DataDir = fullfile(getRepoRoot(),'models','complex-yolov4-models-master','models');
    end
    
     
    properties(TestParameter)
        Model = iGetDifferentModels();       
    end
    
    methods(Test)
        function verifyModelAndFields(test,Model)
            % Test point to verify the fields of the downloaded models are
            % as expected.         
            loadedModel = load(fullfile(test.DataDir,Model.dataFileName));
            
            test.verifyClass(loadedModel.net,'dlnetwork');
            test.verifyEqual(numel(loadedModel.net.Layers), Model.expectedNumLayers);
            test.verifyEqual(size(loadedModel.net.Connections), Model.expectedConnectionsSize);
            test.verifyEqual(loadedModel.net.InputNames, Model.expectedInputNames);
            test.verifyEqual(loadedModel.net.OutputNames, Model.expectedOutputNames);
        end        
    end
end

function Model = iGetDifferentModels()
% Load YOLOv4-coco
dataFileName = 'complex-yolov4-pandaset/complex-yolov4-pandaset.mat';

% Expected anchor boxes and classes.
expectedNumLayers = 363;
expectedConnectionsSize = [397 2];
expectedInputNames = {{'input_1'}};
expectedOutputNames = {{'yoloconv1' 'yoloconv2' 'yoloconv3'}};

detectorYOLOv4Pandaset = struct('dataFileName',dataFileName,...
    'expectedNumLayers',expectedNumLayers,'expectedConnectionsSize',expectedConnectionsSize,...
    'expectedInputNames',expectedInputNames, 'expectedOutputNames',expectedOutputNames);

% Load YOLOv4-tiny-coco
dataFileName = 'tiny-complex-yolov4-pandaset/tiny-complex-yolov4-pandaset.mat';

% Expected anchor boxes and classes.
expectedNumLayers = 74;
expectedConnectionsSize = [80 2];
expectedInputNames = {{'input_1'}};
expectedOutputNames = {{'yoloconv1' 'yoloconv2'}};

detectorTinyYOLOvPandaset = struct('dataFileName',dataFileName,...
    'expectedNumLayers',expectedNumLayers,'expectedConnectionsSize',expectedConnectionsSize,...
    'expectedInputNames',expectedInputNames, 'expectedOutputNames',expectedOutputNames);

Model = struct(...
     'detectorYOLOv4Pandaset',detectorYOLOv4Pandaset,'detectorTinyYOLOvPandaset',detectorTinyYOLOvPandaset);
end