function train(model, opt, data, label, test_data, test_label)

costArr= [];
errArr = [];
cur_epoch = 0;
train_num = size(data,1);
test_num = size(test_data,1);
batch_num = opt.solver.batchsize;
rand_train_idx = randperm(train_num);
while cur_epoch<opt.solver.epoch
    for i=1:floor(train_num/batch_num)
        opt.solver.lr = opt.solver.baselr * (1 + opt.solver.gamma ...
            * (cur_epoch*floor(train_num/batch_num)+i)) ^ (-opt.solver.power);
        tic
        fprintf('---------------- epoch: %d, batch: %d, lr: %f----------------\n', ...
            cur_epoch, i, opt.solver.lr);
        %         fprintf('forward...\n');
        data_batch = data(rand_train_idx((i-1)*batch_num+1:i*batch_num),:,:);
        [cnn_res, opt] = forward(model, opt, data_batch);
        
        batch_label = permute(label(rand_train_idx((i-1)*batch_num+1:i*batch_num),:), [1 3 4 2]);
        error = batch_label-cnn_res{size(opt.layer,2)};
        %         [squeeze(batch_label(1,:)); squeeze(cnn_res{size(option.layer,2)}(1,:))]'
        
        if strcmp(opt.layer(size(opt.layer,2)).type, 'SOFTMAX')
            cost = batch_label.*log(cnn_res{size(opt.layer,2)}) + ...
                (ones(size(batch_label))-batch_label) .*  ...
                log(ones(size(batch_label))-cnn_res{size(opt.layer,2)});
            cost = -mean(mean(cost));
        else
            cost = mean(1/2*sum(error.^2,4),1);
        end
        
        fprintf('cost: %d \n', cost);
        costArr = [costArr cost]; %#ok<AGROW>
        subplot(2,1,1);
        plot(costArr);
        
        %         fprintf('backward...\n');
        model = backward(model, opt, cnn_res, data_batch, error);
        toc
        
        if rem(i,opt.solver.testPeriod)==0
            tic
            fprintf('test...\n');
            rand_test_idx = randperm(test_num);
            testres = forward(model, opt, test_data(rand_test_idx(1:1000),:,:));
            [~, ind] = max(testres{size(opt.layer,2)},[],4);
            [~, ind2] = max(test_label(rand_test_idx(1:1000),:),[],2);
            errRate = mean(min(abs(ind-ind2),ones(size(ind))))*100;
            fprintf('err rate: %.2f \n', errRate);
            errArr = [errArr errRate]; %#ok<AGROW>
            subplot(2,1,2);
            plot(errArr);
            toc
        end
        if rem(cur_epoch*floor(train_num/batch_num)+i,opt.solver.savePeriod)==0
            c=clock;
            filename = sprintf('%d%d%d%d [epoch: %d, iter: %d, err: %.2f].mat' ...
                ,c(2),c(3),c(4),c(5),cur_epoch,i, errRate);
            fprintf('saved <%s> \n', filename);
            save(filename,'model' ,'opt');
        end
        drawnow;
    end
    cur_epoch = cur_epoch+1;
end

end