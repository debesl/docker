baseCleanDir = 'D:\Results\Raspi_deployed\model_fp32.tflite\origHiRec';

% List of noisy (NB) directories to compare
nbDirs = {
    'D:\Results\Raspi_deployed\model_fp16.tflite\bwe', ...
    'D:\Results\Raspi_deployed\model_fp32.tflite\bwe', ...
    'D:\Results\Raspi_deployed\model_int8_dynamic.tflite\bwe', ...
};

% Get list of all clean files recursively
cleanFiles = dir(fullfile(baseCleanDir, '**', '*.wav'));

results = struct();

for d = 1:numel(nbDirs)
    baseNBDir = nbDirs{d};
    visqolScores = [];
    fprintf('\n=== Comparing clean to: %s ===\n', baseNBDir);

    for i = 1:length(cleanFiles)
        cleanFilePath = fullfile(cleanFiles(i).folder, cleanFiles(i).name);
        relativePath = extractAfter(cleanFilePath, length(baseCleanDir));

        nbFilePath = fullfile(baseNBDir, relativePath);
        if exist(nbFilePath, 'file')
            try
                [cleanAudio, fs] = audioread(cleanFilePath);
                [nbAudio, fsNb] = audioread(nbFilePath);

                % Ensure sample rates match (just in case)
                if fs ~= fsNb
                    warning('Sample rate mismatch for %s (clean: %d Hz, noisy: %d Hz). Skipping.', relativePath, fs, fsNb);
                    continue;
                end

                speechMOS = visqol(nbAudio, cleanAudio, fs, Mode="speech");
                if speechMOS >= 2
                    visqolScores(end+1,1) = speechMOS; %#ok<AGROW>
                else
                    fprintf('Excluding low VISQOL score (%.3f) for file: %s\n', speechMOS, relativePath);
                end
            catch ME
                fprintf('⚠️ Skipping unreadable file: %s (%s)\n', nbFilePath, ME.message);
                continue;
            end
        else
            fprintf('Skipping unmatched file: %s\n', relativePath);
        end
    end

    if ~isempty(visqolScores)
        meanVisqol = mean(visqolScores);
        fprintf('Mean VISQOL for %s: %.3f\n', baseNBDir, meanVisqol);
    else
        fprintf('No valid matching files found in %s for VISQOL computation.\n', baseNBDir);
        meanVisqol = NaN;
    end

    % Store result
    [~, name] = fileparts(baseNBDir);
    results.(name) = meanVisqol;
end

% Final summary
fprintf('\n=== Summary of Mean VISQOL Scores ===\n');
fields = fieldnames(results);
for k = 1:numel(fields)
    fprintf('%s: %.3f\n', fields{k}, results.(fields{k}));
end