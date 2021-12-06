classdef DownloadComplexYolov4Fixture < matlab.unittest.fixtures.Fixture
    % DownloadComplexYolov4Fixture  A fixture for calling
    % downloadPretrainedYOLOv4 if necessary. This is to ensure that this
    % function is only called once and only when tests need it. It also
    % provides a teardown to return the test environment to the expected
    % state before testing.
    
    % Copyright 2021 The MathWorks, Inc
    
    properties(Constant)
        Yolov4DataDir = fullfile(getRepoRoot(),'models','complex-yolov4-models-master','models')
    end
    
    properties
        Yolov4PandasetExist (1,1) logical
        TinyYolov4PandasetExist (1,1) logical
    end
    
    methods
        function setup(this) 
            addpath(fullfile(getRepoRoot(),'..'));
            this.Yolov4PandasetExist = exist(fullfile(this.Yolov4DataDir,'complex-yolov4-pandaset','complex-yolov4-pandaset.mat'),'file')==2;
            this.TinyYolov4PandasetExist = exist(fullfile(this.Yolov4DataDir,'tiny-complex-yolov4-pandaset','tiny-complex-yolov4-pandaset.mat'),'file')==2;
            
            % Call this in eval to capture and drop any standard output
            % that we don't want polluting the test logs.
            if ~this.Yolov4PandasetExist
            	evalc('helper.downloadPretrainedYOLOv4(''complex-yolov4-pandaset'');');                
            end
            if ~this.TinyYolov4PandasetExist
            	evalc('helper.downloadPretrainedYOLOv4(''tiny-complex-yolov4-pandaset'');');                
            end
        end
        
        function teardown(this)
            if ~this.Yolov4PandasetExist
            	delete(fullfile(this.Yolov4DataDir,'complex-yolov4-pandaset','complex-yolov4-pandaset.mat'));
            end
            if ~this.TinyYolov4PandasetExist
            	delete(fullfile(this.Yolov4DataDir,'tiny-complex-yolov4-pandaset','tiny-complex-yolov4-pandaset.mat'));
            end
        end
    end
end