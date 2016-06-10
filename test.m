function res = test(model, opt,test_data, test_label)

fprintf('test all...\n');
unitSize = 1000;
numTest = size(test_data,1);
errNum = 0;
for i=1:numTest/unitSize 
    fprintf('test all... (%d/%d)\n',i,numTest/unitSize);
    testres = forward(model, opt, test_data((i-1)*unitSize+1:i*unitSize,:,:), 1);
    [~, ind] = max(testres{size(opt.layer,2)},[],4);
    [~, ind2] = max(test_label((i-1)*unitSize+1:i*unitSize,:),[],2);
    errNum = errNum+sum((min(abs(ind-ind2),ones(size(ind)))));
end
errRate = errNum/numTest*100;
res = errRate;

end