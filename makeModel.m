function [res, option] = makeModel(option)

num_layers = size(option.layer,2);
tmp_outputCH = 1;
tmp_outputSize = option.solver.inputsize;

for i=1:num_layers
    switch(option.layer(i).type)
        case 'CONV'
            fprintf('---------- conv %d ----------\n\n', i);
            res.weight{i} = randn(1, option.layer(i).kernel, ...
                option.layer(i).kernel, tmp_outputCH, option.layer(i).num_output, 'gpuArray');
            res.bias{i} = randn(option.layer(i).num_output,1, 'gpuArray');
            tmp_outputCH = option.layer(i).num_output;
            tmp_outputSize = ceil(( tmp_outputSize + option.layer(i).padding - option.layer(i).kernel) ...
                / option.layer(i).stride) + 1;
            option.layer(i).outputSize = tmp_outputSize;
            
            disp('weight');
            disp(size(res.weight{i}));
            disp('bias');
            disp(size(res.bias{i}));
        case 'POOL'
            fprintf('---------- pool %d ----------\n\n', i);
            tmp_outputSize = ceil(( tmp_outputSize + option.layer(i).padding - option.layer(i).kernel) ...
                / option.layer(i).stride) + 1;
            option.layer(i).outputSize = tmp_outputSize;
        case 'FC'
            fprintf('----------- fc %d -----------\n\n', i);
            res.weight{i} = randn(1, tmp_outputSize, tmp_outputSize, ...
                tmp_outputCH, option.layer(i).num_output,'gpuArray');
            res.bias{i} = randn(option.layer(i).num_output,1, 'gpuArray');   
            tmp_outputCH = option.layer(i).num_output;
            tmp_outputSize = 1;            
            option.layer(i).outputSize = tmp_outputSize;
            
            disp('weight');
            disp(size(res.weight{i}));
            disp('bias');
            disp(size(res.bias{i}));
           
        case 'RELU'
            fprintf('---------- relu %d ----------\n\n', i);
            
        otherwise
            
    end
    if option.layer(i).num_output
        fprintf('output: %d x %d x %d x %d\n',option.solver.batchsize, tmp_outputSize, tmp_outputSize, option.layer(i).num_output);
    else
        fprintf('output: %d x %d x %d x %d\n',option.solver.batchsize, tmp_outputSize, tmp_outputSize, option.layer(i-1).num_output);    
    end
    fprintf('----------------------------\n\n');
end

end