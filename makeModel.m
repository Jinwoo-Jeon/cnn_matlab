function [res, option] = makeModel(option)

num_layers = size(option.layer,2);
tmp_outputCH = 1;
tmp_outputSize = option.solver.inputsize;

for i=1:num_layers
    switch(option.layer(i).type)
        case 'CONV'
%             weightVar = 1/sqrt(tmp_outputCH*option.layer(i).kernel*option.layer(i).kernel*option.layer(i).num_output);
            weightVar = 1/(tmp_outputCH);
            % weightVar = 2/(tmp_outputCH+option.layer(i).num_output);
            res.weight{i} = (weightVar) * randn(1, option.layer(i).kernel, ...
                option.layer(i).kernel, tmp_outputCH, option.layer(i).num_output, 'gpuArray');
            res.prev_del_weight{i} = zeros(size(res.weight{i}));
            res.bias{i} = 0 * randn(option.layer(i).num_output,1, 'gpuArray');
            res.prev_del_bias{i} = zeros(size(res.bias{i}));
            tmp_outputCH = option.layer(i).num_output;
            tmp_outputSize = ceil(( tmp_outputSize + option.layer(i).padding - option.layer(i).kernel) ...
                / option.layer(i).stride) + 1;
            option.layer(i).outputSize = tmp_outputSize;
            
            if option.solver.verbose
                fprintf('---------- conv %d ----------\n\n', i);
                disp('weight');
                disp(size(res.weight{i}));
                disp('bias');
                disp(size(res.bias{i}));
            end
        case 'POOL'
            tmp_outputSize = ceil(( tmp_outputSize + option.layer(i).padding - option.layer(i).kernel) ...
                / option.layer(i).stride) + 1;
            option.layer(i).outputSize = tmp_outputSize;
            
            if option.solver.verbose
                fprintf('---------- pool %d ----------\n\n', i);
            end
        case 'FC'
%             weightVar = 1/sqrt(tmp_outputCH*tmp_outputSize*tmp_outputSize*option.layer(i).num_output);
            weightVar = 1/(tmp_outputCH);
            % weightVar = 2/(tmp_outputCH+option.layer(i).num_output);
            res.weight{i} = (weightVar) * randn(1, tmp_outputSize, tmp_outputSize, ...
                tmp_outputCH, option.layer(i).num_output,'gpuArray');
            res.prev_del_weight{i} = zeros(size(res.weight{i}));
            res.bias{i} = 0 * randn(option.layer(i).num_output,1, 'gpuArray');
            res.prev_del_bias{i} = zeros(size(res.bias{i}));
            tmp_outputCH = option.layer(i).num_output;
            tmp_outputSize = 1;
            option.layer(i).outputSize = tmp_outputSize;
            
            if option.solver.verbose
                fprintf('----------- fc %d -----------\n\n', i);
                disp('weight');
                disp(size(res.weight{i}));
                disp('bias');
                disp(size(res.bias{i}));
            end
            
        case 'RELU'
            if option.solver.verbose
                fprintf('---------- relu %d ----------\n\n', i);
            end
        case 'SOFTMAX'
            if option.solver.verbose
                fprintf('--------- softmax %d ---------\n\n', i);
            end
        case 'DROP'
            if option.solver.verbose
                fprintf('---------- drop %d ----------\n\n', i);
            end
            
        otherwise
            
    end
    if option.solver.verbose
        if option.layer(i).num_output
            fprintf('output: %d x %d x %d x %d\n',option.solver.batchsize, tmp_outputSize, tmp_outputSize, option.layer(i).num_output);
        else
            fprintf('output: %d x %d x %d x %d\n',option.solver.batchsize, tmp_outputSize, tmp_outputSize, option.layer(i-1).num_output);
        end
        fprintf('----------------------------\n\n');
    end
end

end