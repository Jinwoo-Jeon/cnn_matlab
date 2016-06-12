costLog = sprintf('cost: %d', opt.solver.costArr(end));
disp(costLog);
subplot(2,1,1);
plot(1:1:size(opt.solver.costArr,2),opt.solver.costArr);
axis([0 size(opt.solver.costArr,2) 0 inf]);
title(costLog)

errLog = sprintf('err rate: %.2f', opt.solver.errArr(end));
disp(errLog);
subplot(2,1,2);
plot([1:1:size(opt.solver.errArr,2)]*opt.solver.testPeriod, ...
    opt.solver.errArr);
axis([0 size(opt.solver.errArr,2)*opt.solver.testPeriod 0 inf]);
title(errLog)