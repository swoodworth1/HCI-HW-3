% HW3
% Created By: Ryan Bell & Sam Woodworth
% HCI
% Last Modified: 04/14/2021

%Requires Deep Learning, Audio Acquisition, Image Acquisition Toolbox

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

word1 = 0;
word2 = 0;

tts('Specify face location');

while word1 == 0
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
        word1 = 1;
    end
    
    drawnow
end

while word2 == 0

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
        word2 = 1;
    end
    
    drawnow
end


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
