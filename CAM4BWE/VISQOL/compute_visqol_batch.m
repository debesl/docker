function results = compute_visqol_batch(baseCleanDir, nbDir)
%COMPUTE_VISQOL_BATCH Compute mean VISQOL over matching *.wav pairs found
%in baseCleanDir and nbDir (recursively). If baseCleanDir/nbDir are files,
%falls back to single-pair behavior.

    if isstring(baseCleanDir); baseCleanDir = char(baseCleanDir); end
    if isstring(nbDir); nbDir = char(nbDir); end

    results = struct();

    visqolScores = [];
    meanVisqol = 0;
    stats.nSampleRateMismatch = 0;

    % ========================= ADDED =========================
    % Allow passing either directories OR single files.
    isCleanDir = isfolder(baseCleanDir);
    isNbDir    = isfolder(nbDir);

    if isCleanDir && isNbDir
        % ADDED: Recursively gather all clean wav files
        cleanFiles = dir(fullfile(baseCleanDir, '**', '*.wav'));

        % ADDED: Loop over clean wavs, find corresponding nb wav by relative path
        for i = 1:numel(cleanFiles)
            cleanFilePath = fullfile(cleanFiles(i).folder, cleanFiles(i).name);

            % ADDED: compute relative path from baseCleanDir
            rel = cleanFilePath(numel(baseCleanDir)+1:end);  % keeps leading \ or /
            nbFilePath = fullfile(nbDir, rel);

            % ADDED: skip if no corresponding file exists
            if ~exist(nbFilePath, 'file')
                continue;
            end

            try
                [cleanAudio, fs] = audioread(cleanFilePath);
                [nbAudio, fsNb]  = audioread(nbFilePath);

                % ADDED: count SR mismatches and skip those pairs
                if fs ~= fsNb
                    stats.nSampleRateMismatch = stats.nSampleRateMismatch + 1;
                    continue;
                end

                speechMOS = visqol(nbAudio, cleanAudio, fs, Mode="speech");

                if speechMOS >= 2
                    visqolScores(end+1,1) = speechMOS; %#ok<AGROW>
                end

            catch
                % ADDED: silently skip unreadable files (keeps behavior minimal)
                continue;
            end
        end

    else
        % (existing behavior) treat inputs as single file paths
        [cleanAudio, fs] = audioread(baseCleanDir);
        [nbAudio, fsNb] = audioread(nbDir);

        if fs ~= fsNb
            stats.nSampleRateMismatch = stats.nSampleRateMismatch + 1;
        end

        speechMOS = visqol(nbAudio, cleanAudio, fs, Mode="speech");

        if speechMOS >= 2
            visqolScores(end+1,1) = speechMOS;
        end
    end
    % ======================= END ADDED =======================

    if ~isempty(visqolScores)
        meanVisqol = mean(visqolScores);
    else
        meanVisqol = NaN;
    end

    results = struct( ...
        'meanVisqol', meanVisqol, ...
        'nSampleRateMismatch', stats.nSampleRateMismatch ...
    );
end
