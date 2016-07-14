I = imread('C:\Users\Gautam Sridhar\Documents\MATLAB\Rainstreak\Train_data\pic12.jpg');
I = rgb2gray(I);
I = im2double(I)
patchsize = 21;
[m,n] = size(I);
q = (patchsize-1)/2;
image_padded = padarray(I,[q,q],'replicate');
patches = zeros(patchsize*patchsize,m*n);
count = 1;
for i=1:m
    for j=1:n
        i1 = i + q;
        j1 = j + q;
        patch =image_padded(i1-q:i1+q,j1-q:j1+q);
        patches(:,count) = reshape(patch,[441,1]);
        count = count+1;
    end
end
patch_out = predict(opttheta,hiddenSizeL1,hiddenSizeL1,inputSize,patches);
count = 1;
for i=1:m
    for j=1:n
        patch = reshape(patch_out(:,count),[21,21]);
        img(i,j) = patch(q+1,q+1);
        count = count+1;
    end
end
subplot(2,1,1);imshow(I);
subplot(2,1,2);imshow(img);