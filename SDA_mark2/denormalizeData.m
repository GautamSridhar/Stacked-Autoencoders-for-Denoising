function patches = denormalizeData(patches,mean,pstd)
patches = (patches - 0.1)/0.4 -1;
patches = min(max(patches, -pstd), pstd) * pstd;
patches = bsxfun(@plus, patches, mean);
end