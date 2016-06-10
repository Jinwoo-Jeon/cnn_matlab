function [res, option] = forward(model_cnn,option,input,isTest)
clear layer_out
res = cell(size(option.layer,2),1);

layer_input = input;
batch_size = size(input,1);
for i=1:size(option.layer,2)
    out_img_size = option.layer(i).outputSize;
    
    switch(option.layer(i).type)
        case 'CONV'
            if option.solver.verbose
                fprintf('---------- conv %d ----------\n\n', i);
                tic            
            end      
            out_ch_size = size(model_cnn.weight{i},5);
            
            layer_out = zeros(batch_size, out_img_size, out_img_size, out_ch_size,'gpuArray');
            bias = permute(model_cnn.bias{i}, [2 3 4 1]);
            bias = repmat(bias,[batch_size out_img_size out_img_size 1]);
            
            layer_input = gather(layer_input);
            model_cnn.weight{i} = gather(model_cnn.weight{i});
            for j=1:out_ch_size
                weight = model_cnn.weight{i}(:,:,:,:,j);
                layer_out(:,:,:,j)=convn(layer_input,flip(flip(flip(weight,2),3),4),'valid');
            end            
            layer_out = layer_out + bias;
            
            if option.solver.verbose
%                 disp('input');
%                 disp(size(layer_input));
%                 disp('weight');
%                 disp(size(model_cnn.weight{i}));
%                 disp('bias');
%                 disp(size(model_cnn.bias{i}));
%                 disp('output');
%                 disp(size(layer_out));
                toc
                fprintf('----------------------------\n\n');
            end
            
        case 'POOL'
            if option.solver.verbose
                fprintf('---------- pool %d ----------\n\n', i);
                tic            
            end
            out_ch_size = size(layer_input,4);
            kernel = option.layer(i).kernel;
            stride = option.layer(i).stride;
            
            switch(option.solver.pooling)
                case 'MAX'
                    layer_out = zeros(batch_size, out_img_size, out_img_size, ...
                        out_ch_size,'gpuArray');
                    for h_i=1:out_img_size
                        for v_i=1:out_img_size
                            t= layer_input(:, ...
                                stride*(h_i-1)+1:min(size(layer_input,2),stride*(h_i-1)+kernel), ...
                                stride*(v_i-1)+1:min(size(layer_input,3),stride*(v_i-1)+kernel), :);
                            layer_out(:, h_i, v_i, :) = max(max(t,[],2),[],3);
                        end
                    end
                otherwise
            end
            
            if option.solver.verbose
%                 disp('input');
%                 disp(size(layer_input));
%                 disp('output');
%                 disp(size(layer_out));
                toc
                fprintf('----------------------------\n\n');
            end
            
        case 'FC'
            if option.solver.verbose
                fprintf('----------- fc %d -----------\n\n', i);
                tic            
            end           
            
            layer_input_redim = reshape(layer_input, size(layer_input,1), size(layer_input,2) * ...
                size(layer_input,3) * size(layer_input,4));
            weight_redim = reshape(model_cnn.weight{i}, size(model_cnn.weight{i},1) * ...
                size(model_cnn.weight{i},2) * size(model_cnn.weight{i},3) * ...
                size(model_cnn.weight{i},4), size(model_cnn.weight{i},5));
           layer_input_redim= gather(layer_input_redim);
           weight_redim= gather(weight_redim);
            layer_out = permute(layer_input_redim*weight_redim,[1 3 4 2]);
            bias = permute(model_cnn.bias{i}, [2 3 4 1]);
            bias = repmat(bias, [batch_size 1 1 1]);
            
            layer_out = layer_out + bias;
                        
            if option.solver.verbose
%                 disp('input');
%                 disp(size(layer_input));
%                 disp('weight');
%                 disp(size(model_cnn.weight{i}));
%                 disp('bias');
%                 disp(size(model_cnn.bias{i}));
%                 disp('output');
%                 disp(size(layer_out));
                toc
                fprintf('----------------------------\n\n');
            end
        case 'RELU'
            if option.solver.verbose
                fprintf('---------- relu %d ----------\n\n', i);
                tic            
            end
            temp_zero = zeros(size(layer_input),'gpuArray');
            layer_out = max(temp_zero, layer_input);
            if option.solver.verbose
%                 disp('input');
%                 disp(size(layer_input));
%                 disp('output');
%                 disp(size(layer_out));
                toc
                fprintf('----------------------------\n\n');
            end
        case 'PRELU'
            if option.solver.verbose
                fprintf('---------- prelu %d ----------\n\n', i);
                tic            
            end
            temp_zero = zeros(size(layer_input),'gpuArray');            
            layer_out = max(temp_zero, layer_input)+min(temp_zero,layer_input)*model_cnn.weight{i};
            if option.solver.verbose
%                 disp('input');
%                 disp(size(layer_input));
%                 disp('output');
%                 disp(size(layer_out));
                toc
                fprintf('----------------------------\n\n');
            end
        case 'SOFTMAX'
            if option.solver.verbose
                fprintf('--------- softmax %d ---------\n\n', i);
                tic            
            end
            exp_input = exp(layer_input);
            batch_sum = sum(exp_input,4);
            layer_out = exp_input./repmat(batch_sum,[1 1 1 size(layer_input,4)]);   
            
            if option.solver.verbose
%                 disp('input');
%                 disp(size(layer_input));
%                 disp('output');
%                 disp(size(layer_out));
                toc
                fprintf('----------------------------\n\n');
            end
        case 'DROP'
            if option.solver.verbose
                fprintf('---------- drop %d ----------\n\n', i);
                tic            
            end
            
            if isTest
                layer_out = layer_input;
            else
                rate = option.layer(i).rate;
                drop_mask = rand(size(layer_input));
                drop_mask = drop_mask<rate;
                layer_out = layer_input.*drop_mask/rate;
                option.layer(i).mask = drop_mask;                
            end
            
            if option.solver.verbose
%                 disp('input');
%                 disp(size(layer_input));
%                 disp('output');
%                 disp(size(layer_out));
                toc
                fprintf('----------------------------\n\n');
            end
        otherwise
            
    end
    res{i} = layer_out;
    layer_input = layer_out;
end
end