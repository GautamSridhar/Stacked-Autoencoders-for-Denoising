function [activation] = predict(theta,visibleSize,hiddenSizeL1,hiddenSizeL2,hiddenSizeL3,data)

q = (hiddenSizeL1*hiddenSizeL2)+(hiddenSizeL1*visibleSize);
q1 = (hiddenSizeL2*hiddenSizeL3);
q2 = (hiddenSizeL3*visibleSize);
W1 = reshape(theta(1:hiddenSizeL1*visibleSize), hiddenSizeL1, visibleSize);
W2 = reshape(theta(hiddenSizeL1*visibleSize+1:q), hiddenSizeL2, hiddenSizeL1);
W3 = reshape(theta(q+1:q+q1),hiddenSizeL3,hiddenSizeL2);
W4 = reshape(theta(q+q1+1:q+q1+q2),visibleSize,hiddenSizeL3);
b1 = theta(q+q1+q2+1:q+q1+q2+hiddenSizeL1);
b2 = theta(q+q1+q2+hiddenSizeL1+1:q+q1+q2+hiddenSizeL1+hiddenSizeL2);
b3 = theta(q+q1+q2+hiddenSizeL1+hiddenSizeL2+1:q+q1+q2+hiddenSizeL1+hiddenSizeL2+hiddenSizeL3);
b4 = theta(q+q1+q2+hiddenSizeL1+hiddenSizeL2+hiddenSizeL3+1:end);

[~,~,~,~,activation] = getActivation(W1,W2,W3,W4,b1,b2,b3,b4,data);
% patch_out = reshape(activation,[21,21]);


function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
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
end