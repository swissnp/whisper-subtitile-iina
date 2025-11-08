import {parse} from "shell-quote";

const {console, core, preferences, utils, http, file} = iina;

const HOME_PATH = '~/Library/Application Support/com.colliderli.iina/plugins/';
const SERVER_PID_FILE = "@tmp/whisper_server.pid";
const LOG_POLL_INTERVAL_MS = 1000;
const LOG_SEGMENT_REGEX = /^\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})]\s+(.*)$/;
const OPENAI_SIZE_LIMIT_BYTES = 25 * 1024 * 1024;
const DEFAULT_OPENAI_AUDIO = {
    ext: "mp3",
    mime: "audio/mpeg",
    ffmpegArgs: ['-vn', '-ar', '16000', '-ac', '1', '-b:a', '64k', '-f', 'mp3'],
};

let activeServerInfo = null;

export async function transcribe(model) {
    const useOpenAI = isOpenAIMode();
    console.log(`[Whisperina] Selected backend: ${useOpenAI ? "OpenAI Streaming API" : "Local whisper.cpp server"}.`);
    if (!useOpenAI) {
        await downloadOrGetModel(model);
    }
    const url = core.status.url;
    if (!url.startsWith("file://")) {
        throw new Error(`Subtitle generation doesn't work with non-local file ${url}.`);
    }
    const fileName = url.substring(7);
    core.osd("Generating temporary wave file...");
    const tempWavFile = await generateTemporaryWaveFiles(fileName);
    console.log(`[Whisperina] Temporary WAV ready at ${tempWavFile}.`);

    core.osd(useOpenAI ? "Transcribing with OpenAI..." : "Transcribing...");

    const subtitlePath = useOpenAI
        ? await transcribeWithOpenAI(tempWavFile, fileName)
        : await transcribeWithWhisperServer(tempWavFile, model, fileName);

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
    console.log(`[Whisperina] Running ffmpeg to produce intermediate WAV from ${fileName}.`);
    await execWrapped(getFfmpegPath(), ['-y', '-i', fileName, '-ar', '16000', '-ac', '1', "-c:a", "pcm_s16le", tempWavFile]);
    return tempWavFile;
}

async function transcribeWithWhisperServer(tempWavName, modelName, sourceMediaPath) {
    console.log(`[Whisperina] Starting whisper.cpp transcription with model ${modelName}.`);
    const server = await startWhisperServer(modelName);
    const subtitlePath = utils.resolvePath("@tmp/whisper_tmp.wav.srt");
    try {
        await streamTranscription(server, tempWavName, subtitlePath);
        await persistSubtitleCopy(subtitlePath, sourceMediaPath);
        console.log("[Whisperina] whisper.cpp transcription finished.");
        return subtitlePath;
    } finally {
        await stopWhisperServer(server);
    }
}

async function transcribeWithOpenAI(tempWavName, sourceMediaPath) {
    console.log("[Whisperina] Starting OpenAI streaming transcription.");
    const subtitlePath = utils.resolvePath("@tmp/whisper_tmp.wav.srt");
    const rawResponsePath = `${subtitlePath}.openai.json`;
    const preparedAudio = await prepareAudioForOpenAI(tempWavName);
    const streamHandler = createOpenAIStreamHandler(subtitlePath, rawResponsePath);
    try {
        await executeOpenAIStreamingRequest(preparedAudio, streamHandler);
        await streamHandler.waitForFlush();
        await persistSubtitleCopy(subtitlePath, sourceMediaPath);
        console.log("[Whisperina] OpenAI transcription finished.");
        return subtitlePath;
    } finally {
        await streamHandler.waitForFlush().catch(() => {});
    }
}

async function streamTranscription(serverInfo, wavPath, subtitlePath) {
    const monitor = startLogMonitor(serverInfo.logPath, subtitlePath);
    console.log("[Whisperina] Streaming captions from whisper-server log output.");
    let transcriptionError = null;
    try {
        const finalSrt = await requestTranscriptionFromServer(serverInfo, wavPath);
        await monitor.finalize(finalSrt);
    } catch (error) {
        transcriptionError = error;
        await monitor.finalize(null);
    }
    if (transcriptionError) {
        throw transcriptionError;
    }
}

async function reloadSubtitleTrack(subtitlePath) {
    try {
        core.subtitle.loadTrack(subtitlePath);
    } catch (error) {
        console.warn(`Failed to reload subtitle track: ${error.message}`);
    }
}

export function isOpenAIMode() {
    const modeRaw = preferences.get("transcriber_mode");
    const mode = (modeRaw === undefined || modeRaw === null ? "whisper_server" : `${modeRaw}`).trim().toLowerCase();
    return mode === "openai";
}

async function prepareAudioForOpenAI(wavPath) {
    const currentSize = await statFileSize(wavPath);
    console.log(`[Whisperina][OpenAI] Source WAV size: ${(currentSize / (1024 * 1024)).toFixed(2)} MB.`);
    if (currentSize <= OPENAI_SIZE_LIMIT_BYTES) {
        return {path: wavPath, mime: "audio/wav"};
    }
    const outputPath = utils.resolvePath("@tmp/whisper_tmp_openai.mp3");
    console.log("[Whisperina][OpenAI] WAV exceeds 25 MB, re-encoding to MP3.");
    await convertAudioWithFfmpeg(wavPath, outputPath, DEFAULT_OPENAI_AUDIO);
    const newSize = await statFileSize(outputPath);
    console.log(`[Whisperina][OpenAI] Re-encoded MP3 size: ${(newSize / (1024 * 1024)).toFixed(2)} MB.`);
    if (newSize > OPENAI_SIZE_LIMIT_BYTES) {
        throw new Error("Audio file is still larger than 25 MB even after compression. Please trim the media or lower its quality.");
    }
    return {path: outputPath, mime: DEFAULT_OPENAI_AUDIO.mime};
}

async function statFileSize(path) {
    const output = await execWrapped("/usr/bin/stat", ["-f", "%z", path], null, {silent: true});
    const size = parseInt(output.trim(), 10);
    return Number.isFinite(size) ? size : 0;
}

async function convertAudioWithFfmpeg(inputPath, outputPath, codecOptions) {
    const args = ['-y', '-i', inputPath].concat(codecOptions.ffmpegArgs || []).concat([outputPath]);
    await execWrapped(getFfmpegPath(), args);
    return outputPath;
}

async function executeOpenAIStreamingRequest(upload, handler) {
    const apiKey = (preferences.get("openai_api_key") || "").trim();
    if (!apiKey) {
        throw new Error("OpenAI API key is not configured. Please update the preferences page.");
    }
    const baseUrl = (preferences.get("openai_base_url") || "https://api.openai.com/v1/audio/transcriptions").trim();
    const model = (preferences.get("openai_model") || "gpt-4o-transcribe-diarize").trim();
    const responseFormat = (preferences.get("openai_response_format") || "json").trim();
    const chunkingStrategy = (preferences.get("openai_chunking_strategy") || "").trim();

    console.log(`[Whisperina][OpenAI] Streaming request -> model=${model}, response_format=${responseFormat}, chunking=${chunkingStrategy || "default"}, endpoint=${baseUrl}`);
    const args = [
        "curl",
        "-sN",
        "-X", "POST",
        baseUrl,
        "-H", `Authorization: Bearer ${apiKey}`,
        "-H", "Accept: text/event-stream",
        "-F", `file=@${upload.path}`,
        "-F", `model=${model}`,
        "-F", `response_format=${responseFormat}`,
        "-F", "stream=true",
    ];
    if (chunkingStrategy) {
        args.push("-F", `chunking_strategy=${chunkingStrategy}`);
    }
    const result = await utils.exec("/usr/bin/env", args, null, handler.handleChunk, handler.handleError);
    handler.finalize();
    if (result.status !== 0) {
        throw new Error(`OpenAI streaming request failed (status ${result.status}). Check your API key, quota, or model settings.`);
    }
}

function createOpenAIStreamHandler(subtitlePath, rawResponsePath) {
    let buffer = "";
    const segments = [];
    const segmentMap = new Map();
    let writeChain = Promise.resolve();
    const rawEvents = [];
    let rawLogReported = false;
    let rawText = "";

    function nowIso() {
        return new Date().toISOString();
    }

    function recordRawEvent(payload) {
        const entry = {
            timestamp: nowIso(),
            payload,
        };
        rawEvents.push(entry);
        return entry;
    }

    function writeRawDump() {
        if (!rawResponsePath) {
            return;
        }
        try {
            const dump = {
                generated_at: nowIso(),
                raw_text: rawText,
                events: rawEvents,
            };
            file.write(rawResponsePath, JSON.stringify(dump, null, 2));
            if (!rawLogReported) {
                rawLogReported = true;
                console.log(`[Whisperina][OpenAI] Saved raw streaming response to ${rawResponsePath}`);
            }
        } catch (error) {
            console.warn(`Failed to write raw OpenAI response: ${error.message}`);
        }
    }

    function handleChunk(data) {
        const chunk = coerceChunkToString(data);
        if (!chunk) {
            return;
        }
        rawText += chunk;
        buffer += chunk;
        processBuffer();
    }

    function handleError(data) {
        if (data && data.trim().length > 0) {
            console.warn(`OpenAI stream stderr: ${data}`);
        }
    }

    function finalize() {
        processBuffer(true);
        writeRawDump();
    }

    function processBuffer(force = false) {
        let newlineIndex;
        while ((newlineIndex = buffer.indexOf("\n")) >= 0) {
            const line = buffer.slice(0, newlineIndex).trim();
            buffer = buffer.slice(newlineIndex + 1);
            processLine(line);
        }
        if (force && buffer.trim().length > 0) {
            processLine(buffer.trim());
            buffer = "";
        }
    }

    function processLine(line) {
        if (!line || line.startsWith(":")) {
            return;
        }
        if (!line.startsWith("data:")) {
            return;
        }
        const payload = line.slice(5).trim();
        const debugEntry = recordRawEvent(payload);
        if (!payload || payload === "[DONE]") {
            return;
        }
        let event;
        try {
            event = JSON.parse(payload);
            debugEntry.parsed = event;
        } catch (error) {
            debugEntry.parse_error = error.message;
            console.warn(`Unable to parse OpenAI SSE payload: ${error.message}`);
            return;
        }
        handleEvent(event);
    }

    function handleEvent(event) {
        if (!event || typeof event !== "object") {
            return;
        }
        if (event.type === "transcript.text.segment" && event.segment) {
            if (upsertSegment(event.segment)) {
                scheduleFlush();
            }
        } else if (event.type === "transcript.text.done") {
            if (Array.isArray(event.segments) && event.segments.length > 0) {
                segments.length = 0;
                segmentMap.clear();
                event.segments.forEach(segment => upsertSegment(segment, {skipFlush: true}));
                scheduleFlush();
            } else if (typeof event.text === "string") {
                setTextOnly(event.text);
            }
            console.log(`[Whisperina][OpenAI] Received final transcript with ${segments.length} segment(s).`);
        } else if (event.type === "transcript.text.delta" && typeof event.delta === "string") {
            setTextOnly(event.delta, {append: true});
        } else if (event.type === "response.error" && event.error) {
            console.error(`OpenAI response error: ${event.error.message || "unknown error"}`);
        }
    }

    function upsertSegment(rawSegment, options = {}) {
        const normalized = normalizeOpenAISegment(rawSegment);
        if (!normalized) {
            return false;
        }
        const existingIndex = segmentMap.has(normalized.id) ? segmentMap.get(normalized.id) : -1;
        if (existingIndex >= 0) {
            segments[existingIndex] = normalized;
        } else {
            segmentMap.set(normalized.id, segments.length);
            segments.push(normalized);
            segments.sort((a, b) => a.startMs - b.startMs);
            segments.forEach((segment, index) => segmentMap.set(segment.id, index));
        }
        if (!options.skipFlush) {
            scheduleFlush();
        }
        return true;
    }

    function setTextOnly(text, options = {}) {
        const content = (options.append && segments.length > 0)
            ? `${segments[segments.length - 1].textLines.join(" ")} ${text}`.trim()
            : (text || "").trim();
        if (!content) {
            return;
        }
        const estimatedDuration = getMediaDurationMs() || estimateDurationFromText(content);
        segments.length = 0;
        segmentMap.clear();
        segments.push({
            id: "text-only",
            startMs: 0,
            endMs: estimatedDuration,
            textLines: splitTextIntoLines(content),
        });
        segmentMap.set("text-only", 0);
        scheduleFlush();
    }

    function scheduleFlush() {
        writeChain = writeChain.then(async () => {
            writeRawDump();
            const rendered = renderSegmentsToSrt(segments);
            file.write(subtitlePath, rendered);
            await reloadSubtitleTrack(subtitlePath);
        }).catch(error => {
            console.warn(`Failed to update  streaming subtitles: ${error.message}`);
        });
        return writeChain;
    }

    function normalizeOpenAISegment(segment) {
        if (!segment) {
            return null;
        }
        const text = (segment.text || "").trim();
        if (!text) {
            return null;
        }
        const startMs = secondsToMs(segment.start ?? segment.start_time ?? 0);
        const endMs = secondsToMs(segment.end ?? segment.end_time ?? startMs + 2);
        const id = segment.id || `${startMs}-${endMs}-${text.length}`;
        return {
            id,
            startMs,
            endMs,
            textLines: splitTextIntoLines(text),
        };
    }

        return {
            handleChunk,
            handleError,
            finalize,
            async waitForFlush() {
                await writeChain;
                writeRawDump();
                console.log(`[Whisperina][OpenAI] Subtitle file flushed with ${segments.length} segment(s).`);
            },
        };
}

function coerceChunkToString(data) {
    if (data === undefined || data === null) {
        return "";
    }
    if (typeof data === "string") {
        return data;
    }
    if (typeof data === "object") {
        try {
            if (typeof data.string === "function") {
                const strValue = data.string();
                if (typeof strValue === "string") {
                    return strValue;
                }
            }
            if (typeof data.toString === "function") {
                const repr = data.toString();
                if (typeof repr === "string" && repr !== "[object Object]") {
                    return repr;
                }
            }
            const json = JSON.stringify(data);
            if (typeof json === "string") {
                return json;
            }
        } catch (error) {
            console.warn(`Failed to convert stream chunk to string: ${error.message}`);
        }
    }
    return `${data}`;
}

function startLogMonitor(logPath, subtitlePath) {
    const seen = new Set();
    const segments = [];
    let stopRequested = false;
    console.log(`[Whisperina] Monitoring whisper-server log at ${logPath}.`);
    const loopPromise = (async () => {
        while (!stopRequested) {
            try {
                const updated = collectNewSegments(logPath, seen, segments);
                if (updated) {
                    const rendered = renderSegmentsToSrt(segments);
                    file.write(subtitlePath, rendered);
                    await reloadSubtitleTrack(subtitlePath);
                }
            } catch (error) {
                console.warn(`Log monitor error: ${error.message}`);
            }
            await sleep(LOG_POLL_INTERVAL_MS);
        }
    })();

    return {
        async finalize(finalSrt) {
            stopRequested = true;
            await loopPromise;
            if (finalSrt) {
                file.write(subtitlePath, finalSrt);
                await reloadSubtitleTrack(subtitlePath);
            }
        },
    };
}

function collectNewSegments(logPath, seen, segments) {
    if (!logPath || !file.exists(logPath)) {
        return false;
    }
    const content = file.read(logPath) || "";
    if (!content) {
        return false;
    }
    const lines = content.split(/\r?\n/);
    let updated = false;
    let addedCount = 0;
    for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line) {
            continue;
        }
        const match = LOG_SEGMENT_REGEX.exec(line);
        if (!match) {
            continue;
        }
        const [, start, end, textRaw] = match;
        const text = textRaw.trim();
        const key = `${start}|${end}|${text}`;
        if (seen.has(key)) {
            continue;
        }
        seen.add(key);
        segments.push({
            timing: `${convertLogTimestamp(start)} --> ${convertLogTimestamp(end)}`,
            text: [text],
        });
        updated = true;
        addedCount += 1;
    }
    if (updated) {
        console.log(`[Whisperina] Added ${addedCount} new whisper-server segments (total=${segments.length}).`);
    }
    return updated;
}

function convertLogTimestamp(value) {
    if (!value) {
        return "00:00:00,000";
    }
    const [whole = "", fractional = "000"] = value.split(".");
    const frac = (fractional + "000").slice(0, 3);
    return `${whole},${frac}`;
}

async function startWhisperServer(modelName) {
    const serverPath = getWhisperServerPath();
    const host = preferences.get("wserver_host") || "127.0.0.1";
    const port = parseInt(preferences.get("wserver_port"), 10) || 17896;
    const logPath = utils.resolvePath("@tmp/whisper_server.log");
    const args = ['-m', `${DATA}/ggml-${modelName}.bin`, '--host', host, '--port', `${port}`].concat(getServerOptions());
    const command = `${shellEscape(serverPath)} ${args.map(shellEscape).join(" ")} > ${shellEscape(logPath)} 2>&1 & echo $!`;
    console.log(`[Whisperina] Launching whisper-server via: ${command}`);
    const stdout = await execWrapped("/bin/sh", ["-c", command], null, {silent: true});
    const pid = parseInt(stdout.trim().split("\n").pop() || "", 10);
    if (!Number.isFinite(pid)) {
        throw new Error("Unable to start whisper-server. Check your server path and options.");
    }
    const serverInfo = {pid, host, port, baseUrl: `http://${host}:${port}`, logPath};
    activeServerInfo = serverInfo;
    rememberServerPid(pid);
    try {
        await waitForServerReady(serverInfo.baseUrl);
        console.log("[Whisperina] whisper-server reported healthy.");
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
    const info = serverInfo || activeServerInfo;
    const recordedPid = readRecordedServerPid();
    const pid = info?.pid || recordedPid;
    if (!pid) {
        clearRecordedServerPid();
        return;
    }
    try {
        await utils.exec("/bin/kill", ["-TERM", `${pid}`]);
        await sleep(200);
    } catch (error) {
        if (!/No such process/i.test(error?.message || "")) {
            console.warn(`Failed to terminate whisper-server (pid ${pid}): ${error.message}`);
        }
    }
    try {
        await utils.exec("/bin/kill", ["-KILL", `${pid}`]);
    } catch (error) {
        if (!/No such process/i.test(error?.message || "")) {
            console.warn(`Failed to force-stop whisper-server (pid ${pid}): ${error.message}`);
        }
    }
    if (activeServerInfo && activeServerInfo.pid === pid) {
        activeServerInfo = null;
    }
    clearRecordedServerPid();
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

function getServerPidPath() {
    try {
        return utils.resolvePath(SERVER_PID_FILE);
    } catch (error) {
        console.warn(`Unable to resolve server PID file: ${error.message}`);
        return null;
    }
}

function rememberServerPid(pid) {
    const pidPath = getServerPidPath();
    if (!pidPath) {
        return;
    }
    try {
        file.write(pidPath, `${pid}`);
    } catch (error) {
        console.warn(`Unable to record whisper-server pid: ${error.message}`);
    }
}

function clearRecordedServerPid() {
    const pidPath = getServerPidPath();
    if (!pidPath) {
        return;
    }
    try {
        if (file.exists(pidPath)) {
            file.delete(pidPath);
        }
    } catch (error) {
        console.warn(`Unable to remove PID record: ${error.message}`);
    }
}

function readRecordedServerPid() {
    const pidPath = getServerPidPath();
    if (!pidPath) {
        return null;
    }
    try {
        if (!file.exists(pidPath)) {
            return null;
        }
        const content = file.read(pidPath);
        const pid = parseInt((content || "").trim(), 10);
        return Number.isFinite(pid) ? pid : null;
    } catch (error) {
        console.warn(`Unable to read PID record: ${error.message}`);
        return null;
    }
}

function renderSegmentsToSrt(segments) {
    if (!segments || segments.length === 0) {
        return "";
    }
    const normalized = segments.map(normalizeRenderableSegment).filter(Boolean);
    if (normalized.length === 0) {
        return "";
    }
    normalized.sort((a, b) => a.startMs - b.startMs);
    const lines = [];
    normalized.forEach((segment, index) => {
        lines.push(String(index + 1));
        lines.push(`${formatTimestamp(segment.startMs)} --> ${formatTimestamp(segment.endMs)}`);
        segment.textLines.forEach(textLine => lines.push(textLine));
        lines.push("");
    });
    return lines.join("\n");
}

function normalizeRenderableSegment(segment) {
    if (!segment) {
        return null;
    }
    if (segment.timing) {
        const [startRaw, endRaw] = segment.timing.split(/\s+-->\s+/);
        return {
            startMs: parseTimestampMs(startRaw),
            endMs: parseTimestampMs(endRaw),
            textLines: Array.isArray(segment.text) ? segment.text : Array.isArray(segment.textLines) ? segment.textLines : [segment.text || ""],
        };
    }
    const startMs = typeof segment.startMs === "number" ? segment.startMs : 0;
    const endMsCandidates = [
        segment.endMs,
        startMs + estimateDurationFromTextLines(segment.textLines || segment.text || []),
    ].filter(value => typeof value === "number" && value > startMs);
    const endMs = endMsCandidates.length > 0 ? Math.max(...endMsCandidates) : startMs + 2000;
    const textLines = Array.isArray(segment.textLines)
        ? segment.textLines
        : Array.isArray(segment.text)
            ? segment.text
            : [segment.text || ""];
    const filtered = textLines.filter(line => typeof line === "string" ? line.trim().length > 0 : false);
    return {startMs, endMs, textLines: filtered.length > 0 ? filtered : [""]};
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

function getMediaDurationMs() {
    const duration = core?.status?.duration;
    if (typeof duration === "number" && duration > 0) {
        return Math.floor(duration * 1000);
    }
    return null;
}

function estimateDurationFromText(text) {
    const words = (text || "").trim().split(/\s+/).filter(Boolean).length;
    const averageMsPerWord = 320; // ~187 wpm
    return words > 0 ? words * averageMsPerWord : 4000;
}

function estimateDurationFromTextLines(lines) {
    if (!lines || lines.length === 0) {
        return 2000;
    }
    const joined = Array.isArray(lines) ? lines.join(" ") : String(lines);
    return estimateDurationFromText(joined);
}

function splitTextIntoLines(text) {
    if (!text) {
        return [];
    }
    return text.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
}

function secondsToMs(value) {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return 0;
    }
    return Math.max(0, Math.round(value * 1000));
}

function parseTimestampMs(value) {
    const match = /^(\d{2}):(\d{2}):(\d{2}),(\d{3})$/.exec(value || "");
    if (!match) {
        return 0;
    }
    const [, hh, mm, ss, ms] = match;
    return ((((parseInt(hh, 10) * 60) + parseInt(mm, 10)) * 60) + parseInt(ss, 10)) * 1000 + parseInt(ms, 10);
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
