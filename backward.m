function res = backward(model_cnn,option,cnn_res, data, err)

res = model_cnn;    
lr = option.solver.lr;
lr_b = lr *2;
batch_size = size(data,1);
layer_error_in = err;
for i=size(option.layer,2):-1:1
    if i>1
        pre_layer_val = cnn_res{i-1};
    else
        pre_layer_val = data;
    end 
    switch(option.layer(i).type)
        case 'CONV'    
            if option.solver.verbose
                fprintf('---------- conv %d ----------\n\n', i);
                tic            
            end           
            weight = permute(res.weight{i},[1 2 3 5 4]);
            error = layer_error_in;
            
%             weight update
            delta = zeros(size(res.weight{i}),'gpuArray');
            pre_layer_val = gather(pre_layer_val);
            error = gather(error);
            for j=1:size(error,4)                
                delta(:,:,:,:,j) = convn(pre_layer_val, flip(flip(flip(error(:,:,:,j),1),2),3),'valid');
%                 delta(:,:,:,:,j) = convn(pre_layer_val, error(:,:,:,j),'valid');
            end
            
            
            res.prev_del_weight{i} = lr * delta/batch_size - lr * option.solver.weight_decay * res.weight{i} ...
                + option.solver.momentum * res.prev_del_weight{i};
            res.weight{i} = res.weight{i} + res.prev_del_weight{i};
            res.prev_del_bias{i} = lr_b * permute(mean(mean(mean(error,1),2),3),[4 3 2 1]) ...
                -  lr_b * option.solver.weight_decay * res.bias{i}...
                + option.solver.momentum * res.prev_del_bias{i};
            res.bias{i} = res.bias{i} + res.prev_del_bias{i};
            
%             error propagate
            out_ch_size = size(weight,5);
            layer_error_out = zeros(batch_size, size(pre_layer_val,2), size(pre_layer_val,3), ...
                out_ch_size,'gpuArray');

            weight = gather(weight);
            error = gather(error);
            for j=1:out_ch_size
                tmp_out = padarray(error,[0  size(weight,2)-1 size(weight,3)-1 0]);
                layer_error_out(:,:,:,j)=convn(tmp_out,flip(weight(:,:,:,:,j),4),'valid');
            end
            
            
            if option.solver.verbose
%                 disp('err');
%                 disp(size(layer_error_in));
%                 disp('err_out');
%                 disp(size(layer_error_out));
                toc
                fprintf('----------------------------\n\n');
            end
        case 'POOL' 
            if option.solver.verbose
                fprintf('---------- pool %d ----------\n\n', i);
                tic            
            end
            switch(option.solver.pooling)
                case 'MAX'
                    kernel = option.layer(i).kernel;
                    layer_error_out = repelem(layer_error_in, 1, kernel, kernel, 1);
                otherwise
            end
            
            
            if option.solver.verbose
%                 disp('err');
%                 disp(size(layer_error_in));
%                 disp('err_out');
%                 disp(size(layer_error_out));
                toc
                fprintf('----------------------------\n\n');
            end
        case 'FC'
            if option.solver.verbose
                fprintf('---------- fc %d ----------\n\n', i);
                tic            
            end
            weight = res.weight{i};
            error = permute(layer_error_in,[1 2 3 5 4]);
            
%             weight update
            delta_b = error;
            
            pre_layer_val = repmat(pre_layer_val, [1 1 1 1 size(error,5)]);
            error = repmat(error, [1 size(pre_layer_val,2) size(pre_layer_val,3) size(pre_layer_val,4) 1]);            
            delta_w = pre_layer_val.*error;
            
            res.prev_del_weight{i} = lr * mean(delta_w,1) - lr*option.solver.weight_decay*res.weight{i} ...
                + option.solver.momentum * res.prev_del_weight{i};
            res.weight{i} = weight + res.prev_del_weight{i};
            
            res.prev_del_bias{i} = lr_b * permute(mean(delta_b,1),[5 1 2 3 4]) ...
                -  lr_b * option.solver.weight_decay*res.bias{i} ...
                + option.solver.momentum * res.prev_del_bias{i};
            res.bias{i} = res.bias{i} + res.prev_del_bias{i};
            
%             error propagate
            error2d = reshape(layer_error_in, [size(layer_error_in,1) size(layer_error_in,4)]);
            weight2d = reshape(weight, [size(weight,2)*size(weight,3)*size(weight,4), size(weight,5)]);
            layer_error_out = error2d*weight2d';
            layer_error_out = reshape(layer_error_out, [size(layer_error_out,1) size(weight,2) ...
                size(weight,3) size(weight,4)]);

            if option.solver.verbose
%                 disp('err');
%                 disp(size(layer_error_in));
%                 disp('weight');
%                 disp(size(res.weight{i}));
%                 disp('bias');
%                 disp(size(res.bias{i}));
%                 disp('err_out');
%                 disp(size(layer_error_out));
                toc
                fprintf('----------------------------\n\n');
            end
        case 'RELU'
            if option.solver.verbose
                fprintf('---------- relu %d ----------\n\n', i);
                tic            
            end
            
            layer_error_out = layer_error_in;
            layer_error_out(pre_layer_val<0)=0;
            
            if option.solver.verbose
%                 disp('input');
%                 disp(size(layer_error_in));
%                 disp('output');
%                 disp(size(layer_error_out));
                toc
                fprintf('----------------------------\n\n');
            end
        case 'SOFTMAX'
            if option.solver.verbose
                fprintf('--------- softmax %d ---------\n\n', i);
                tic            
            end
            layer_error_out = layer_error_in;
            
            if option.solver.verbose
%                 disp('input');
%                 disp(size(layer_error_in));
%                 disp('output');
%                 disp(size(layer_error_out));
                toc
                fprintf('----------------------------\n\n');
            end
            
        case 'DROP'            
            if option.solver.verbose
                fprintf('---------- drop %d ----------\n\n', i);
                tic            
            end
            layer_error_out = layer_error_in.*option.layer(i).mask;
            if option.solver.verbose
%                 disp('input');
%                 disp(size(layer_error_in));
%                 disp('output');
%                 disp(size(layer_error_out));
                toc
                fprintf('----------------------------\n\n');
            end
        otherwise
    end
    layer_error_in=layer_error_out;
end
end