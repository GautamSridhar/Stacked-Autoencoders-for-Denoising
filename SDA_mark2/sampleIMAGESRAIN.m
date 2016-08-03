function [patches,patches_rain] = sampleIMAGESRAIN()
% sampleIMAGES
% Returns 10000 patches for training

load sample_images;    % load images from disk 
load sample_images_noise;

patchsize = 11;  % we'll use 8x8 patches 
numpatches = 20000;

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchsize*patchsize, numpatches);
patches_rain = zeros(patchsize*patchsize, numpatches);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data 
%  from IMAGES.  
%  
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1
for i=1:numpatches
    r = randi(85,1);
    c = randi(85,1);
    d = randi(10,1);
    patch = sample_images(r:r+10,c:c+10,d);
    patches(:,i) = reshape(patch,[121 1]);
%     patches_rain(:,i) = reshape(imnoise(patch,'gaussian',0.1),[225 1]);
    patches_rain(:,i) = reshape(sample_images_noise(r:r+10,c:c+10,d),[121,1]);
    i
end

%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
% patches = normalizeData(patches);
% patches_rain = normalizeData(patches_rain);

end
% 
% 
% %% ---------------------------------------------------------------
