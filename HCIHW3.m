% HW3
% Created By: Ryan Bell & Sam Woodworth
% HCI
% Last Modified: 04/14/2021

clear, clc
load('commandNet.mat')

[x,fs] = audioread('stop_command.flac');
sound(x,fs)
auditorySpect = helperExtractAuditoryFeatures(x,fs);
command = classify(trainedNet,auditorySpect)

x = audioread('play_command.flac');
sound(x,fs)
auditorySpect = helperExtractAuditoryFeatures(x,fs);
command = classify(trainedNet,auditorySpect)

