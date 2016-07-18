function [cost,grad] = finetune(theta,visibleSize,hiddenSizeL1,hiddenSizeL2,...
                                    lambda,data_clean,data_noise)

q = (hiddenSizeL1*hiddenSizeL2)+(hiddenSizeL1*visibleSize);
q1 = (hiddenSizeL2*hiddenSizeL3);
q2 = (hiddenSizeL3*visibleSize);
W1 = reshape(theta(1:hiddenSizeL1*visibleSize), hiddenSizeL1, visibleSize);
W2 = reshape(theta(hiddenSizeL1*visibleSize+1:q), hiddenSizeL2, hiddenSizeL1);
W3 = reshape(theta(q+1:q+q1),hiddenSizeL3,hiddenSizeL2);
W4 = reshape(theta(q+q1+1:q+q1+q2),visibleSize,hiddenSizeL3);
b1 = theta(q+q1+q2+1:q+q1+q2+hiddenSizeL1);
b2 = theta(q+q1+q2+hiddenSizeL1+1:q+q1+q2+hiddenSizeL1+hiddenSizeL2);
b3 = theta(q+q1+q3+hiddenSizeL1+hiddenSizeL2+1:q+q1+q3+hiddenSizeL1+hiddenSizeL2+hiddenSizeL3);
b4 = theta(q+q1+q3+hiddenSizeL1+hiddenSizeL2+hiddenSizeL3+1:end);

W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
W3grad = zeros(size(W3));
W4grad = zeros(size(W4));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));
b3grad = zeros(size(b3));
b4grad = zeros(size(b4));
[~, nSamples] = size(data_noise);

[a1, a2, a3,a4,a5] = getActivation(W1, W2,W3,W4, b1, b2,b3,b4, data_noise);
errtp = ((a5 - data_clean) .^ 2) ./ 2;
err = sum(sum(errtp)) ./ nSamples;

% err2 = sum(sum(W4 .^ 2));
% err2 = err2 * lambda / 2;

cost = err;

delta5 = -(data_clean - a5) .* dsigmoid(a5);
delta4 = (W4' * delta5); 
delta4 = delta4 .* dsigmoid(a4);
delta3 = (W3' * delta4); 
delta3 = delta3 .* dsigmoid(a3);
delta2 = (W2' * delta3);
delta2 = delta2 .* dsigmoid(a2);
nablaW1 = delta2 * a1';
nablab1 = delta2;
nablaW2 = delta3 * a2';
nablab2 = delta3;
nablaW3 = delta4 * a3';
nablab3 = delta4;
nablaW4 = delta5*a4';
nablab4 = delta5;
W1grad = nablaW1 ./ nSamples;
W2grad = nablaW2 ./ nSamples;
W3grad = nablaW3 ./ nSamples;
W4grad = nablaW4 ./ nSamples;
b1grad = sum(nablab1, 2) ./ nSamples;
b2grad = sum(nablab2, 2) ./ nSamples;
b3grad = sum(nablab3, 2) ./ nSamples;
b4grad = sum(nablab4, 2) ./ nSamples;
grad = [W1grad(:);W2grad(:);W3grad(:);W4grad(:);b1grad(:);b2grad(:);b3grad(:);b4grad(:)];
end


function sigm = sigmoid(x)
 
sigm = 1 ./ (1 + exp(-x));
end

function dsigm = dsigmoid(a)
dsigm = a .* (1.0 - a);
 
end
function [ainput, ahidden1,ahidden2,ahidden3, aoutput] = getActivation(W1, W2,W3,W4, b1, b2,b3,b4, data)
 
ainput = data;
ahidden1 = bsxfun(@plus, W1 * ainput, b1);
ahidden1 = sigmoid(ahidden1);
ahidden2 = bsxfun(@plus, W2 * ahidden1, b2);
ahidden2 = sigmoid(ahidden2);
ahidden3 = bsxfun(@plus, W3 * ahidden2, b3);
ahidden3 = sigmoid(ahidden3);
aoutput = bsxfun(@plus, W4 * ahidden3, b4);
aoutput = sigmoid(aoutput);
end