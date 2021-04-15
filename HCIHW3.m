% HW3
% Created By: Ryan Bell & Sam Woodworth
% HCI
% Last Modified: 04/14/2021

%Requires Deep Learning, Audio Acquisition, Image Acquisition Toolbox,
%Computer vision toolbox

clear, clc
load('commandNet.mat')

fs = 16000;
classificationRate = 20;
adr = audioDeviceReader('SampleRate',fs,'SamplesPerFrame',floor(fs/classificationRate));
audioBuffer = dsp.AsyncBuffer(fs);

labels = trainedNet.Layers(end).Classes;
YBuffer(1:classificationRate/2) = categorical("background");

probBuffer = zeros([numel(labels),classificationRate/2]);
countThreshold = ceil(classificationRate*0.2);
probThreshold = 0.7;

wordOneLoop = 0;
wordTwoLoop = 0;

tts('Specify face location');

while wordOneLoop == 0
    x = adr();
    write(audioBuffer,x);
    y = read(audioBuffer,fs,fs-adr.SamplesPerFrame);

    spec = helperExtractAuditoryFeatures(y,fs);

    [YPredicted,probs] = classify(trainedNet,spec,'ExecutionEnvironment','cpu');
    YBuffer = [YBuffer(2:end),YPredicted];
    probBuffer = [probBuffer(:,2:end),probs(:)];

    [word,count] = mode(YBuffer);
    maxProb = max(probBuffer(labels == word,:));
    
    if (word ~= "down" && word ~= "up")
        
    else 
        word
        wordOneSelected = word;
        wordOneLoop = 1;
    end
    
    drawnow
end

while wordTwoLoop == 0

    x = adr();
    write(audioBuffer,x);
    y = read(audioBuffer,fs,fs-adr.SamplesPerFrame);

    spec = helperExtractAuditoryFeatures(y,fs);

    [YPredicted,probs] = classify(trainedNet,spec,'ExecutionEnvironment','cpu');
    YBuffer = [YBuffer(2:end),YPredicted];
    probBuffer = [probBuffer(:,2:end),probs(:)];

    [word,count] = mode(YBuffer);
    word;
    maxProb = max(probBuffer(labels == word,:));
    
    if (word ~= "left" && word ~= "right")

    else
        word
        wordTwoSelected = word;
        wordTwoLoop = 1;
    end
    
    drawnow
end

% Create the face detector object.
faceDetector = vision.CascadeObjectDetector();

% Create the point tracker object.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Create the webcam object.
cam = webcam();

% Capture one frame to get its size.
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

% Create the video player object.
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);

runLoop = true;
numPts = 0;
frameCount = 0;

topLeft = [2 2 320 2 320 240 2 240];
topRight = [320 2 640 2 640 240 320 240];
bottomLeft = [2 240 320 240 320 480 2 480];
bottomRight = [320 240 640 240 640 480 320 480];

if (wordOneSelected == "up")
   if (wordTwoSelected == "left")
       currentBoundary = topLeft;
       boundaryNum = 1;
   else
       currentBoundary = topRight;
       boundaryNum = 2;
   end
else
   if (wordTwoSelected == "left")
       currentBoundary = bottomLeft;
       boundaryNum = 3;
   else
       currentBoundary = bottomRight;
       boundaryNum = 4;
   end
end
    
currentBoundary
boundaryNum

% 640 x 640
% w x h
% 320 x 320

while runLoop

    % Get the next frame.
    videoFrame = snapshot(cam);
    videoFrameGray = rgb2gray(videoFrame);
    frameCount = frameCount + 1;

    %videoFrame = insertShape(videoFrame, 'Line', [320 0 320 640], 'LineWidth', 3, 'Color', 'green'); %vertical line
    %videoFrame = insertShape(videoFrame, 'Line', [0 240 640 240], 'LineWidth', 3, 'Color', 'green'); %horizontal line
    videoFrame = insertShape(videoFrame, 'Polygon', topLeft, 'LineWidth', 3, 'Color', 'yellow'); %top left
    videoFrame = insertShape(videoFrame, 'Polygon', topRight, 'LineWidth', 3, 'Color', 'yellow'); %top right
    videoFrame = insertShape(videoFrame, 'Polygon', bottomLeft, 'LineWidth', 3, 'Color', 'yellow'); %bottom left
    videoFrame = insertShape(videoFrame, 'Polygon', bottomRight, 'LineWidth', 3, 'Color', 'yellow'); %bottom right
    
    if numPts < 10
        % Detection mode.
        bbox = faceDetector.step(videoFrameGray);

        if ~isempty(bbox)
            % Find corner points inside the detected region.
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));

            % Re-initialize the point tracker.
            xyPoints = points.Location;
            numPts = size(xyPoints,1);
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);

            % Save a copy of the points.
            oldPoints = xyPoints;

            % Convert the rectangle represented as [x, y, w, h] into an
            % M-by-2 matrix of [x,y] coordinates of the four corners. This
            % is needed to be able to transform the bounding box to display
            % the orientation of the face.
            bboxPoints = bbox2points(bbox(1, :));

            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);

            % Display a bounding box around the detected face.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

            % Display detected corners.
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
        end

    else
        % Tracking mode.
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);

        numPts = size(visiblePoints, 1);

        if numPts >= 10
            % Estimate the geometric transformation between the old points
            % and the new points.
            [xform, inlierIdx] = estimateGeometricTransform2D(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
            oldInliers    = oldInliers(inlierIdx, :);
            visiblePoints = visiblePoints(inlierIdx, :);

            % Apply the transformation to the bounding box.
            bboxPoints = transformPointsForward(xform, bboxPoints);

            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);

            % Display a bounding box around the face being tracked.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

            % Display tracked points.
            videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');

            % Reset the points.
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
        end
        
        % Check if face in boundary
        if boundaryNum == 1 % top left
            
        elseif boundaryNum == 2 % top right
            
        elseif boundaryNum == 3 % bottom left
            
        elseif boundaryNum == 4 % bottom right
            if bboxPoints(1, 1) >= bottomRight(1, 1) && bboxPoints(1, 2) >= bottomRight(1, 2)
                disp('In') 
            end
        end
            
    end

    % Display the annotated video frame using the video player object.
    step(videoPlayer, videoFrame);

    % Check whether the video player window has been closed.
    runLoop = isOpen(videoPlayer);
end

% Clean up.
clear cam;
release(videoPlayer);
release(pointTracker);
release(faceDetector);

function wav = tts(txt,voice,pace,fs)
if ~ispc, error('Microsoft Win32 SAPI is required.'); end
if ~ischar(txt), error('First input must be string.'); end
SV = actxserver('SAPI.SpVoice');
TK = invoke(SV,'GetVoices');
if nargin > 1
    % Select voice;
    for k = 0:TK.Count-1
        if strcmpi(voice,TK.Item(k).GetDescription)
            SV.Voice = TK.Item(k);
            break;
        elseif strcmpi(voice,'list')
            disp(TK.Item(k).GetDescription);
        end
    end
    % Set pace;
    if nargin > 2
        if isempty(pace), pace = 0; end
        if abs(pace) > 10, pace = sign(pace)*10; end        
        SV.Rate = pace;
    end
end
if nargin < 4 || ~ismember(fs,[8000,11025,12000,16000,22050,24000,32000,...
        44100,48000]), fs = 16000; end
if nargout > 0
   % Output variable;
   MS = actxserver('SAPI.SpMemoryStream');
   MS.Format.Type = sprintf('SAFT%dkHz16BitMono',fix(fs/1000));
   SV.AudioOutputStream = MS;  
end
invoke(SV,'Speak',txt);
if nargout > 0
    % Convert uint8 to double precision;
    wav = reshape(double(invoke(MS,'GetData')),2,[])';
    wav = (wav(:,2)*256+wav(:,1))/32768;
    wav(wav >= 1) = wav(wav >= 1)-2;
    delete(MS);
    clear MS;
end
delete(SV); 
clear SV TK;
pause(0.2);
end % TTS;
