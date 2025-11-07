import {parse} from "shell-quote";

const {console, core, preferences, utils, http, file} = iina;

const HOME_PATH = '~/Library/Application Support/com.colliderli.iina/plugins/';
const STREAM_CHUNK_SECONDS = 10;
const SAMPLE_RATE = 16000;
const PCM_BYTES_PER_SAMPLE = 2;
const WAV_HEADER_BYTES = 44;

export async function transcribe(model) {
    await downloadOrGetModel(model);
    const url = core.status.url;
    if (!url.startsWith("file://")) {
        throw new Error(`Subtitle generation doesn't work with non-local file ${url}.`);
    }
    const fileName = url.substring(7);
    core.osd("Generating temporary wave file...");
    const tempWavFile = await generateTemporaryWaveFiles(fileName);

    core.osd("Transcribing...");
    const subtitlePath = await transcribeAudio(tempWavFile, model, fileName);

    core.osd("Transcription succeeded.");
    return [subtitlePath];
}

async function downloadOrGetModel(model) {
    if (utils.fileInPath(`@data/ggml-${model}.bin`)) {
        core.osd(`Model ${model} already exists.`);
    } else if (utils.ask(`Model ${model} does not exist. Would you like to download it now?`)) {
        await execWrapped(`${HOME}/bin/download-ggml-model.sh`, [model], DATA);
        core.osd(`Model ${model} has been successfully downloaded.`);
    } else {
        throw Error(`No such model ${model}.`);
    }
}

async function generateTemporaryWaveFiles(fileName) {
    const tempWavFile = utils.resolvePath("@tmp/whisper_tmp.wav");
    await execWrapped(getFfmpegPath(), ['-y', '-i', fileName, '-ar', '16000', '-ac', '1', "-c:a", "pcm_s16le", tempWavFile]);
    return tempWavFile;
}

async function transcribeAudio(tempWavName, modelName, sourceMediaPath) {
    const server = await startWhisperServer(modelName);
    const subtitlePath = utils.resolvePath("@tmp/whisper_tmp.wav.srt");
    try {
        await streamTranscription(server, tempWavName, subtitlePath);
        await persistSubtitleCopy(subtitlePath, sourceMediaPath);
        return subtitlePath;
    } finally {
        await stopWhisperServer(server);
    }
}

async function streamTranscription(serverInfo, wavPath, subtitlePath) {
    const aggregatedSegments = [];
    const totalDurationMs = await estimateAudioDurationMs(wavPath);
    if (!totalDurationMs || totalDurationMs <= STREAM_CHUNK_SECONDS * 1000) {
        const srtContent = await requestTranscriptionFromServer(serverInfo, wavPath);
        aggregatedSegments.push(...shiftSrtBlocks(parseSrtBlocks(srtContent), 0));
        const rendered = renderSegmentsToSrt(aggregatedSegments);
        file.write(subtitlePath, rendered);
        await reloadSubtitleTrack(subtitlePath);
        return;
    }
    let processedMs = 0;
    let chunkIndex = 0;
    while (processedMs < totalDurationMs) {
        const remainingMs = totalDurationMs - processedMs;
        const chunkDurationMs = Math.min(STREAM_CHUNK_SECONDS * 1000, remainingMs);
        const chunkPath = utils.resolvePath(`@tmp/whisper_chunk_${chunkIndex}.wav`);
        try {
            await extractAudioChunk(wavPath, chunkPath, processedMs, chunkDurationMs);
            const chunkSrt = await requestTranscriptionFromServer(serverInfo, chunkPath);
            aggregatedSegments.push(...shiftSrtBlocks(parseSrtBlocks(chunkSrt), processedMs));
            const rendered = renderSegmentsToSrt(aggregatedSegments);
            file.write(subtitlePath, rendered);
            await reloadSubtitleTrack(subtitlePath);
            processedMs += chunkDurationMs;
            chunkIndex += 1;
        } finally {
            await removeFileQuietly(chunkPath);
        }
    }
}

async function extractAudioChunk(sourcePath, outputPath, startMs, durationMs) {
    const startSeconds = (startMs / 1000).toFixed(3);
    const durationSeconds = (durationMs / 1000).toFixed(3);
    await execWrapped(getFfmpegPath(), [
        '-y',
        '-ss',
        startSeconds,
        '-t',
        durationSeconds,
        '-i',
        sourcePath,
        '-ar',
        `${SAMPLE_RATE}`,
        '-ac',
        '1',
        '-c:a',
        'pcm_s16le',
        outputPath,
    ]);
}

async function reloadSubtitleTrack(subtitlePath) {
    try {
        core.subtitle.loadTrack(subtitlePath);
    } catch (error) {
        console.warn(`Failed to reload subtitle track: ${error.message}`);
    }
}

async function removeFileQuietly(path) {
    try {
        await utils.exec("/bin/rm", ["-f", path]);
    } catch (error) {
        console.warn(`Unable to remove temp file ${path}: ${error.message}`);
    }
}

async function startWhisperServer(modelName) {
    const serverPath = getWhisperServerPath();
    const host = preferences.get("wserver_host") || "127.0.0.1";
    const port = parseInt(preferences.get("wserver_port"), 10) || 17896;
    const logPath = utils.resolvePath("@tmp/whisper_server.log");
    const args = ['-m', `${DATA}/ggml-${modelName}.bin`, '--host', host, '--port', `${port}`].concat(getServerOptions());
    const command = `${shellEscape(serverPath)} ${args.map(shellEscape).join(" ")} > ${shellEscape(logPath)} 2>&1 & echo $!`;
    const stdout = await execWrapped("/bin/sh", ["-c", command], null, {silent: true});
    const pid = parseInt(stdout.trim().split("\n").pop() || "", 10);
    if (!Number.isFinite(pid)) {
        throw new Error("Unable to start whisper-server. Check your server path and options.");
    }
    const serverInfo = {pid, host, port, baseUrl: `http://${host}:${port}`, logPath};
    try {
        await waitForServerReady(serverInfo.baseUrl);
    } catch (error) {
        console.error(`Failed to start whisper-server (log: ${logPath})`);
        await stopWhisperServer(serverInfo);
        throw error;
    }
    return serverInfo;
}

async function waitForServerReady(baseUrl, timeoutMs = 20000) {
    const healthUrl = `${baseUrl}/health`;
    const deadline = Date.now() + timeoutMs;
    let lastError;
    while (Date.now() < deadline) {
        try {
            const response = await http.get(healthUrl);
            if (response.statusCode === 200) {
                return;
            }
            lastError = new Error(`Unexpected server status: ${response.statusCode}`);
        } catch (error) {
            lastError = error;
        }
        await sleep(300);
    }
    const reason = lastError ? lastError.message : "unknown";
    throw new Error(`Timed out waiting for whisper-server to become ready (${reason}).`);
}

async function requestTranscriptionFromServer(serverInfo, wavPath) {
    const inferenceUrl = `${serverInfo.baseUrl}/inference`;
    const stdout = await execWrapped("/usr/bin/env", [
        "curl",
        "-sS",
        "-f",
        "-X",
        "POST",
        "-F",
        `file=@${wavPath}`,
        "-F",
        "response_format=srt",
        inferenceUrl,
    ], null, {silent: true});
    return stdout;
}

async function stopWhisperServer(serverInfo) {
    if (!serverInfo || !serverInfo.pid) {
        return;
    }
    try {
        await utils.exec("/bin/kill", ["-TERM", `${serverInfo.pid}`]);
    } catch (error) {
        console.warn(`Failed to stop whisper-server (pid ${serverInfo.pid}): ${error.message}`);
    }
}

async function persistSubtitleCopy(subtitlePath, mediaFile) {
    const archiveDir = resolveArchiveDirectory();
    if (!archiveDir) {
        return;
    }
    try {
        await execWrapped("/bin/mkdir", ["-p", archiveDir]);
        const archiveName = `${sanitizeFileStem(mediaFile)}-${formatTimestampSuffix()}.srt`;
        const destination = `${archiveDir}/${archiveName}`;
        await execWrapped("/bin/cp", ["-f", subtitlePath, destination]);
        console.log(`Stored subtitle copy at ${destination}`);
    } catch (error) {
        console.warn(`Failed to archive subtitle: ${error.message}`);
    }
}

function getServerOptions() {
    return parseArgumentList(preferences.get("wserver_options"));
}

function getFfmpegPath() {
    const ffmpegPath = preferences.get("ffmpeg_path");
    if (!utils.fileInPath(ffmpegPath)) {
        throw new Error(`Unable to locate ffmpeg executable at: ${ffmpegPath}. Check the preference page for more details.`);
    }
    return ffmpegPath;
}

function getPluginHomePath() {
    const PLUGIN_NAME = 'io.github.yuxiqian.whisperina.iinaplugin';
    const PLUGIN_NAME_DEV = 'whisperina.iinaplugin-dev';
    if (utils.fileInPath(HOME_PATH + PLUGIN_NAME)) {
        return utils.resolvePath(HOME_PATH + PLUGIN_NAME);
    } else if (utils.fileInPath(HOME_PATH + PLUGIN_NAME_DEV)) {
        return utils.resolvePath(HOME_PATH + PLUGIN_NAME_DEV);
    } else {
        throw new Error("Unable to locate plugin folder.");
    }
}

function getPluginDataPath() {
    return utils.resolvePath("@data/")
}

async function estimateAudioDurationMs(wavPath) {
    const durationFromCore = core?.status?.duration;
    if (typeof durationFromCore === "number" && durationFromCore > 0) {
        return Math.floor(durationFromCore * 1000);
    }
    try {
        const sizeOutput = await execWrapped("/usr/bin/stat", ["-f", "%z", wavPath], null, {silent: true});
        const totalBytes = parseInt(sizeOutput.trim(), 10);
        if (Number.isFinite(totalBytes) && totalBytes > WAV_HEADER_BYTES) {
            const payloadBytes = totalBytes - WAV_HEADER_BYTES;
            const samples = payloadBytes / PCM_BYTES_PER_SAMPLE;
            const seconds = samples / SAMPLE_RATE;
            return Math.floor(seconds * 1000);
        }
    } catch (error) {
        console.warn(`Unable to inspect wav duration: ${error.message}`);
    }
    return null;
}

function parseSrtBlocks(content) {
    if (!content) {
        return [];
    }
    return content.split(/\r?\n\r?\n/).map(block => block.trim()).filter(Boolean).map(block => {
        const lines = block.split(/\r?\n/).map(line => line.trimEnd());
        if (lines.length < 2) {
            return null;
        }
        let timingLineIndex = 0;
        if (/^\d+$/.test(lines[0])) {
            timingLineIndex = 1;
        }
        const timingLine = lines[timingLineIndex];
        if (!isTimingLine(timingLine)) {
            return null;
        }
        const textLines = lines.slice(timingLineIndex + 1);
        return {timing: timingLine, text: textLines};
    }).filter(Boolean);
}

function shiftSrtBlocks(blocks, offsetMs) {
    if (!blocks || blocks.length === 0) {
        return [];
    }
    return blocks.map(block => ({
        timing: shiftTimingLine(block.timing, offsetMs),
        text: block.text || [],
    }));
}

function renderSegmentsToSrt(segments) {
    if (!segments || segments.length === 0) {
        return "";
    }
    const lines = [];
    segments.forEach((segment, index) => {
        lines.push(String(index + 1));
        lines.push(segment.timing);
        if (segment.text.length > 0) {
            lines.push(...segment.text);
        }
        lines.push("");
    });
    return lines.join("\n");
}

function isTimingLine(line) {
    return /^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}$/.test(line || "");
}

function shiftTimingLine(line, offsetMs) {
    if (!isTimingLine(line) || !Number.isFinite(offsetMs) || offsetMs === 0) {
        return line;
    }
    const [startRaw, endRaw] = line.split(/\s+-->\s+/);
    const startMs = parseTimestampMs(startRaw);
    const endMs = parseTimestampMs(endRaw);
    return `${formatTimestamp(Math.max(0, startMs + offsetMs))} --> ${formatTimestamp(Math.max(0, endMs + offsetMs))}`;
}

function parseTimestampMs(value) {
    const match = /^(\d{2}):(\d{2}):(\d{2}),(\d{3})$/.exec(value || "");
    if (!match) {
        return 0;
    }
    const [, hh, mm, ss, ms] = match;
    return (((parseInt(hh, 10) * 60 + parseInt(mm, 10)) * 60) + parseInt(ss, 10)) * 1000 + parseInt(ms, 10);
}

function formatTimestamp(ms) {
    const clamped = Math.max(0, Math.floor(ms));
    const hours = Math.floor(clamped / 3600000);
    const minutes = Math.floor((clamped % 3600000) / 60000);
    const seconds = Math.floor((clamped % 60000) / 1000);
    const millis = clamped % 1000;
    const pad = (num, len = 2) => String(num).padStart(len, "0");
    return `${pad(hours)}:${pad(minutes)}:${pad(seconds)},${pad(millis, 3)}`;
}

function resolveArchiveDirectory() {
    const configured = preferences.get("subtitle_archive_dir");
    if (!configured) {
        return null;
    }
    try {
        return utils.resolvePath(configured);
    } catch (error) {
        console.warn(`Unable to resolve archive directory "${configured}": ${error.message}`);
        return null;
    }
}

function sanitizeFileStem(filePath) {
    if (!filePath) {
        return "subtitle";
    }
    const normalized = filePath.replace(/\\/g, "/");
    const baseName = normalized.substring(normalized.lastIndexOf("/") + 1) || "subtitle";
    const stem = baseName.replace(/\.[^/.]+$/, "") || "subtitle";
    return stem.replace(/[^A-Za-z0-9._-]+/g, "_") || "subtitle";
}

function formatTimestampSuffix(date = new Date()) {
    const pad = (value) => String(value).padStart(2, "0");
    return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
}

function shellEscape(value) {
    if (value === undefined || value === null) {
        return "''";
    }
    const str = String(value);
    if (/^[A-Za-z0-9_\/.:=-]+$/.test(str)) {
        return str;
    }
    return `'${str.replace(/'/g, `'\\''`)}'`;
}

function parseArgumentList(rawValue) {
    if (!rawValue) {
        return [];
    }
    try {
        const parsed = parse(rawValue);
        if (!Array.isArray(parsed) || parsed.length === 0) {
            return [];
        }
        return parsed.filter(token => typeof token === "string" && token.length > 0).map(token => `${token}`);
    } catch (error) {
        console.warn(`Failed to parse option string "${rawValue}": ${error.message}`);
        return [];
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function execWrapped(file, commands, cwd, options = {}) {
    const {
        status, stdout, stderr
    } = await utils.exec(file, commands, cwd);
    if (!options.silent) {
        console.log(status);
        console.log(stdout);
        console.log(stderr);
    }
    if (status !== 0) {
        throw new Error(`Bad return status code: ${status}`);
    }
    return stdout;
}

const HOME = getPluginHomePath();
const DATA = getPluginDataPath();

function getWhisperServerPath() {
    const serverPath = preferences.get("wserver_path");
    if (!utils.fileInPath(serverPath)) {
        throw new Error(`Unable to locate whisper-server executable at: ${serverPath}. Check the preference page for more details.`);
    }
    return serverPath;
}
