function [activation] = predict(W1,W2,W3,W4,b1,b2,b3,b4,data)


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