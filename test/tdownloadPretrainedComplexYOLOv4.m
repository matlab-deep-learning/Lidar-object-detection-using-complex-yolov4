classdef(SharedTestFixtures = {DownloadComplexYolov4Fixture}) tdownloadPretrainedComplexYOLOv4 < matlab.unittest.TestCase
    % Test for downloadPretrainedYOLOv4
    
    % Copyright 2021 The MathWorks, Inc.
    
    % The shared test fixture DownloadComplexYolov4Fixture calls
    % downloadPretrainedYOLOv4. Here we check that the downloaded files
    % exists in the appropriate location.
    
    properties        
        DataDir = fullfile(getRepoRoot(),'models','complex-yolov4-models-master','models');
    end
    
     
    properties(TestParameter)
        Model = {'complex-yolov4-pandaset', 'tiny-complex-yolov4-pandaset'};
    end
    
    methods(Test)
        function verifyDownloadedFilesExist(test,Model)
            dataFileName = [Model,'.mat'];
            test.verifyTrue(isequal(exist(fullfile(test.DataDir,Model,dataFileName),'file'),2));
        end
    end
end