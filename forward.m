function res = forward(model_cnn,option,input)
clear layer_out
% model_cnn = model;
% option = opt;
% input = train_img(1:100,:,:);

layer_input = input;
batch_size = size(layer_input,1);
for i=1:size(option.layer,2)
    out_img_size = option.layer(i).outputSize;
    
    switch(option.layer(i).type)
        case 'CONV'
            
            out_ch_size = size(model_cnn.weight{i},5);
            %             stride & padding 적용 안돼있음
            layer_out = zeros(batch_size, out_img_size, out_img_size, out_ch_size,'gpuArray');
            for j=1:out_ch_size
                layer_out(:,:,:,j)=convn(layer_input,model_cnn.weight{i}(:,:,:,:,j),'valid');
            end
            
            if option.solver.verbose
                fprintf('---------- conv %d ----------\n\n', i);
                disp('input');
                disp(size(layer_input));
                disp('weight');
                disp(size(model_cnn.weight{i}));
                disp('output');
                disp(size(layer_out));
                fprintf('----------------------------\n\n');
            end
            
        case 'POOL'
            
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
                                stride*(h_i-1)+1:stride*(h_i-1)+kernel, ...
                                stride*(v_i-1)+1:stride*(v_i-1)+kernel, :);
                            layer_out(:, h_i, v_i, :) = max(max(t,[],2),[],3);
                        end
                    end
                otherwise
            end
            
            if option.solver.verbose
                fprintf('---------- pool %d ----------\n\n', i);
                disp('input');
                disp(size(layer_input));
                disp('output');
                disp(size(layer_out));
                fprintf('----------------------------\n\n');
            end
            
        case 'FC'
            
            layer_input2 = reshape(layer_input, size(layer_input,1), size(layer_input,2) * ...
                size(layer_input,3) * size(layer_input,4));
            weight2 = reshape(model_cnn.weight{i}, size(model_cnn.weight{i},1) * ...
                size(model_cnn.weight{i},2) * size(model_cnn.weight{i},3) * ...
                size(model_cnn.weight{i},4), size(model_cnn.weight{i},5));
            layer_out = permute(layer_input2*flipud(weight2),[1 3 4 2]);
            if option.solver.verbose
                fprintf('----------- fc %d -----------\n\n', i);
                disp('input');
                disp(size(layer_input));
                disp('weight');
                disp(size(model_cnn.weight{i}));
                disp('output');
                disp(size(layer_out));
                fprintf('----------------------------\n\n');
            end
        case 'RELU'
            temp_zero = zeros(size(layer_input));
            layer_out = max(temp_zero, layer_input);
            if option.solver.verbose
                fprintf('---------- relu %d ----------\n\n', i);
                disp('input');
                disp(size(layer_input));
                disp('output');
                disp(size(layer_out));
                fprintf('----------------------------\n\n');
            end
        otherwise
            
    end
    layer_input = layer_out;
end
res = layer_out;
end