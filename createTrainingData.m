%% Download Pandaset Data Set
% This example uses a subset of PandaSet[2], that contains 2560
% preprocessed organized point clouds. Each point cloud is specified as a
% 64-by-1856 matrix. The corresponding ground truth contains the semantic
% segmentation labels for 12 classes. The point clouds are stored in PCD
% format, and the ground truth data is stored in PNG format. The size of
% the data set is 5.2 GB. Execute this code to download the data set.

url = 'https://ssd.mathworks.com/supportfiles/lidar/data/Pandaset_LidarData.tar.gz';
outputFolder = fullfile(tempdir,'Pandaset');

lidarDataTarFile = fullfile(outputFolder,'Pandaset_LidarData.tar.gz');
if ~exist(lidarDataTarFile, 'file')
    mkdir(outputFolder);
    disp('Downloading Pandaset Lidar driving data (5.2 GB)...');
    websave(lidarDataTarFile, url);
    untar(lidarDataTarFile,outputFolder);
end

% Check if tar.gz file is downloaded, but not uncompressed.
if (~exist(fullfile(outputFolder,'Lidar'), 'file'))...
        &&(~exist(fullfile(outputFolder,'Cuboids'), 'file'))
    untar(lidarDataTarFile,outputFolder);
end

lidarFolder =  fullfile(outputFolder,'Lidar');
labelsFolder = fullfile(outputFolder,'Cuboids');

% Note: Depending on your Internet connection, the download process can
% take some time. The code suspends MATLAB® execution until the download
% process is complete. Alternatively, you can download the data set to your
% local disk using your web browser, and then extract Pandaset_LidarData
% folder. To use the file you downloaded from the web, change the
% outputFolder variable in the code to the location of the downloaded file.

%% Create the Bird's Eye View Image from the point cloud data

% Read the ground truth labels.
gtPath = fullfile(labelsFolder,'PandaSetLidarGroundTruth.mat');
data = load(gtPath,'lidarGtLabels');
Labels = timetable2table(data.lidarGtLabels);
boxLabels = Labels(:,2:end);

% Get the configuration parameters.
gridParams = helper.getGridParameters();

% Get classnames of Pandaset dataset.
classNames = boxLabels.Properties.VariableNames;

numFiles = size(boxLabels,1);
processedLabels = cell(size(boxLabels));

 for i = 1:numFiles
     
    lidarPath = fullfile(lidarFolder,sprintf('%04d.pcd',i));
    ptCloud = pcread(lidarPath);  
    
    groundTruth = boxLabels(i,:);

    [processedData,~] = helper.preprocess(ptCloud,gridParams);

    for ii = 1:numel(classNames)

        labels = groundTruth(1,classNames{ii}).Variables;
        if(iscell(labels))
            labels = labels{1};
        end
        if ~isempty(labels)

            % Get the label indices that are in the selected RoI.
            % Get the label indices that are in the selected RoI.
            labelsIndices = labels(:,1) - labels(:,4) > gridParams.xMin ...
                          & labels(:,1) + labels(:,4) < gridParams.xMax ...
                          & labels(:,2) - labels(:,5) > gridParams.yMin ...
                          & labels(:,2) + labels(:,5) < gridParams.yMax ...
                          & labels(:,4) > 0 ...
                          & labels(:,5) > 0 ...
                          & labels(:,6) > 0;
            labels = labels(labelsIndices,:);

            labelsBEV = labels(:,[2,1,5,4,9]);
            labelsBEV(:,5) = -labelsBEV(:,5);

            labelsBEV(:,1) = int32(floor(labelsBEV(:,1)/gridParams.gridW)) + 1;
            labelsBEV(:,2) = int32(floor(labelsBEV(:,2)/gridParams.gridH)+gridParams.bevHeight/2) + 1;

            labelsBEV(:,3) = int32(floor(labelsBEV(:,3)/gridParams.gridW)) + 1;
            labelsBEV(:,4) = int32(floor(labelsBEV(:,4)/gridParams.gridH)) + 1;

        end
        processedLabels{i,ii} = labelsBEV;
    end
    
    writePath = fullfile(outputFolder,'BEVImages');
    if ~isfolder(writePath)
        mkdir(writePath);
    end
    
    imgSavePath = fullfile(writePath,sprintf('%04d.jpg',i));
    imwrite(processedData,imgSavePath);

end

processedLabels = cell2table(processedLabels);
numClasses = size(processedLabels,2);
for j = 1:numClasses
    processedLabels.Properties.VariableNames{j} = classNames{j};
end

labelsSavePath = fullfile(outputFolder,'Cuboids/BEVGroundTruthLabels.mat');
save(labelsSavePath,'processedLabels');
