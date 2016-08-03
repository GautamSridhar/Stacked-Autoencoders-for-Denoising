% base code for starting on stacked autoencoders.
% Layers used in this program is 2.

%%======================================================================
inputSize = 11*11;
hiddenSizeL1 = 605;    % Layer 1 Hidden Size
hiddenSizeL2 = 605;    % Layer 2 Hidden Size
hiddenSizeL3 = 605;    % Layer 3 Hidden Size 
sparsityParam = 0.05;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho) 
lambda = 1e-4;         % weight decay parameter       
beta = 1e-2;              % weight of sparsity penalty term
batchSize = 2000;
maxIter = 50;
%%======================================================================
%Load the data for training
[patches,patches_noise] = sampleIMAGESRAIN;
display_network(patches_noise(:,randi(size(patches,2),200,1)),8);

%%======================================================================
%Train the first layer
sae1Theta = initializeParameters(hiddenSizeL1,inputSize);                  %initialize the weights

addpath minFunc/
options.Method = 'lbfgs';                                   
options.maxIter =  20;
options.display = 'on';
theta = sae1Theta;
for i=1:maxIter
    startIndex = mod((i-1) * batchSize, size(patches,2)) + 1;
    data = patches(:, startIndex:startIndex + batchSize-1);
    data_noise = patches_noise(:, startIndex:startIndex + batchSize-1);
    costFunc = @(p) sparseAutoencoderCost(p,inputSize,hiddenSizeL1,lambda,sparsityParam,beta,data,data_noise);
    [opttheta1,cost1] = minFunc(costFunc,theta,options);
    theta = opttheta1;
    if cost1<=0.1
        costfinal = sparseAutoencoderCost(theta,inputSize,hiddenSizeL1,...
                                           lambda,sparsityParam,beta,patches,patches_noise);
        if costfinal <=0.1
            break;
        end
    end
    i
end
    
%%======================================================================
%Feedforward through the first autoencoder
[sae1Features,W1,b1,W4,b4] = feedForwardAutoencoder(opttheta1, hiddenSizeL1, ...
                                        inputSize, patches);
[sae1Features_noise,~,~] = feedForwardAutoencoder(opttheta1,hiddenSizeL1,inputSize,patches_noise);                                    
%%======================================================================
%Train the second layer
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);%initialize the weights
theta2 = sae2Theta;
for i=1:maxIter
    startIndex = mod((i-1) * batchSize, size(patches,2)) + 1;
    data2 = sae1Features(:, startIndex:startIndex + batchSize-1);
    data_noise2 = sae1Features_noise(:, startIndex:startIndex + batchSize-1);
    costFunc2 = @(p) sparseAutoencoderCost(p,hiddenSizeL1,hiddenSizeL2,...          %calculate cost and gradient  
                                       lambda,sparsityParam,beta,data2,data_noise2);
    [opttheta2,cost2] = minFunc(costFunc2,theta2,options);
    theta2 = opttheta2;
    i
end

%%======================================================================
%Feedforward through the second autoencoder
[sae2Features,W2,b2,W3,b3] = feedForwardAutoencoder(opttheta2, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);
[sae2Features_noise,~,~] = feedForwardAutoencoder(opttheta2, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features_noise);                                    
%%======================================================================
%Train the third Layer
% sae3Theta = initializeParameters(hiddenSizeL3,hiddenSizeL2);
% costFunc3 = @(p) sparseAutoencoderCost(p,hiddenSizeL2,hiddenSizeL3,...          %calculate cost and gradient  
%                                        lamda,sparsityParam,beta,sae1Features);
% [opttheta3,cost3] = fmincg(costFunc3,sae3Theta,options);                                   
%%======================================================================
%Feedforward through the third autoencoder
% [sae3Features,W3,b3] = feedForwardAutoencoder(opttheta3, hiddenSizeL2, ...
%                                         hiddenSizeL3, patches);
%%======================================================================
%%======================================================================
%Fine Tuning
init_theta = [W1(:);W2(:);W3(:);W4(:);b1(:);b2(:);b3(:);b4(:)];

theta3 = init_theta;
for i=1:maxIter
    startIndex = mod((i-1) * batchSize, size(patches,2)) + 1;
    data3 = patches(:, startIndex:startIndex + batchSize-1);
    data_noise3 = patches_noise(:, startIndex:startIndex + batchSize-1);                            
    costfin = @(p) finetune(p,inputSize,hiddenSizeL1,hiddenSizeL2,hiddenSizeL3,...
                                    lambda,data3,data_noise3);
     [opttheta3,cost3] = minFunc(costfin,theta3,options);
    theta3 = opttheta3;
    i
end
                                
%%======================================================================
%Prediction





%%======================================================================
