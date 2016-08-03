I = imread('C:\Users\Gautam Sridhar\Documents\MATLAB\Rainstreak\Train_data\Train_rainstreak\picrainstreak10.jpg');
I = rgb2gray(I);
I = im2double(I);
J = I;  
% I = imnoise(I,'gaussian');
patchsize = 11;
[m,n] = size(I);
q = (patchsize-1)/2;
image_padded = padarray(I,[q,q],'replicate');
patchez = zeros(patchsize*patchsize,m*n);
count = 1;
for i=1:m
    for j=1:n
        i1 = i + q;
        j1 = j + q;
        patch =image_padded(i1-q:i1+q,j1-q:j1+q);
%         patch = imnoise(patch,'gaussian',0.1);
        patchez(:,count) = reshape(patch,[patchsize*patchsize,1]);
        count = count+1;
    end
end
% patchez = normalizeData(patchez);
patch_out = predict(opttheta3,inputSize,hiddenSizeL1,hiddenSizeL2,hiddenSizeL3,patchez);
count = 1;
for i=1:m
    for j=1:n
        patch = reshape(patch_out(:,count),[patchsize,patchsize]);
        img(i,j) = patch(q+1,q+1);
        count = count+1;
    end
end
subplot(2,1,1);imshow(I);
subplot(2,1,2);imshow(img);