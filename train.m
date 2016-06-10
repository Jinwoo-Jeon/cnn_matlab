function train(model, opt, data, label, test_data, test_label)

cur_epoch = opt.solver.startepoch;
cur_iter = opt.solver.startiter;
train_num = size(data,1);
batch_num = opt.solver.batchsize;
rand_train_idx = randperm(train_num);
while cur_epoch <= opt.solver.epoch
    while cur_iter <= floor(train_num/batch_num)
        opt.solver.lr = opt.solver.baselr * (1 + opt.solver.gamma ...
            * ((cur_epoch-1)*floor(train_num/batch_num)+cur_iter)) ^ (-opt.solver.power);
        tic
        fprintf('---------------- epoch: %d, batch: %d, lr: %f----------------\n', ...
            cur_epoch, cur_iter, opt.solver.lr);
        %         fprintf('forward...\n');
        data_batch = data(rand_train_idx((cur_iter-1)*batch_num+1:cur_iter*batch_num),:,:);
        [cnn_res, opt] = forward(model, opt, data_batch, 0);
        
        batch_label = permute(label(rand_train_idx((cur_iter-1)*batch_num+1:cur_iter*batch_num),:), [1 3 4 2]);
        error = batch_label-cnn_res{size(opt.layer,2)};
        %                 [squeeze(batch_label(1,:)); squeeze(cnn_res{size(opt.layer,2)}(1,:))]'
        
        if strcmp(opt.layer(size(opt.layer,2)).type, 'SOFTMAX')
            cost = batch_label.*log(cnn_res{size(opt.layer,2)}) + ...
                (ones(size(batch_label))-batch_label) .*  ...
                log(ones(size(batch_label))-cnn_res{size(opt.layer,2)});
            cost = -mean(mean(cost));
        else
            cost = mean(1/2*sum(error.^2,4),1);
        end
        
        costLog = sprintf('cost: %d', cost);
        disp(costLog);
        opt.solver.costArr = [opt.solver.costArr cost];
        subplot(2,1,1);
        plot(1:1:size(opt.solver.costArr,2),opt.solver.costArr);
        axis([0 size(opt.solver.costArr,2) 0 inf]);
        title(costLog)
        
        %         fprintf('backward...\n');
        model = backward(model, opt, cnn_res, data_batch, error);
        
        if rem(cur_iter,opt.solver.testPeriod)==0
            errRate = test(model, opt,test_data, test_label);
            errLog = sprintf('err rate: %.2f', errRate);
            disp(errLog);
            opt.solver.errArr = [opt.solver.errArr errRate];
            subplot(2,1,2);
            plot([1:1:size(opt.solver.errArr,2)]*opt.solver.testPeriod, ...
                opt.solver.errArr);
            axis([0 size(opt.solver.errArr,2)*opt.solver.testPeriod 0 inf]);
            title(errLog)
        end
        if rem((cur_epoch-1)*floor(train_num/batch_num)+cur_iter,opt.solver.savePeriod)==0
            c=clock;
            opt.solver.startepoch = cur_epoch;
            opt.solver.startiter = cur_iter+1;
            filename = sprintf('%.2d%.2d%.2d%.2d_epoch_%d_iter_%d_err_%.2f.mat' ...
                ,c(2),c(3),c(4),c(5),cur_epoch,cur_iter, errRate);
            fprintf('saved <%s> \n', filename);
            save(filename,'model' ,'opt');
        end
        drawnow;
        cur_iter = cur_iter+1;
        toc
    end
    cur_epoch = cur_epoch+1;
    cur_iter = 1;
end

end