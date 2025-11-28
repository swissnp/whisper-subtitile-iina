import {parse} from "shell-quote";

const {console, core, preferences, utils, http, file} = iina;

const HOME_PATH = '~/Library/Application Support/com.colliderli.iina/plugins/';
const SERVER_PID_FILE = "@tmp/whisper_server.pid";
const LOG_POLL_INTERVAL_MS = 1000;
const LOG_SEGMENT_REGEX = /^\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})]\s+(.*)$/;
const OPENAI_SIZE_LIMIT_BYTES = 25 * 1024 * 1024;
const OPENAI_AUDIO_PROFILES = [
    {
        name: "aac_high",
        ext: "m4a",
        mime: "audio/mp4",
        description: "AAC 96 kbps mono 16 kHz",
        ffmpegArgs: ['-vn', '-ar', '16000', '-ac', '1', '-c:a', 'aac', '-b:a', '96k', '-movflags', '+faststart'],
    },
    {
        name: "aac_medium",
        ext: "m4a",
        mime: "audio/mp4",
        description: "AAC 64 kbps mono 16 kHz",
        ffmpegArgs: ['-vn', '-ar', '16000', '-ac', '1', '-c:a', 'aac', '-b:a', '64k', '-movflags', '+faststart'],
    },
    {
        name: "opus_voice",
        ext: "webm",
        mime: "audio/webm",
        description: "Opus 48 kbps mono 16 kHz",
        ffmpegArgs: ['-vn', '-ar', '16000', '-ac', '1', '-c:a', 'libopus', '-b:a', '48k', '-vbr', 'on'],
    },
    {
        name: "mp3_speech",
        ext: "mp3",
        mime: "audio/mpeg",
        description: "MP3 64 kbps mono 16 kHz",
        ffmpegArgs: ['-vn', '-ar', '16000', '-ac', '1', '-b:a', '64k', '-f', 'mp3'],
    },
    {
        name: "mp3_low",
        ext: "mp3",
        mime: "audio/mpeg",
        description: "MP3 48 kbps mono 16 kHz",
        ffmpegArgs: ['-vn', '-ar', '16000', '-ac', '1', '-b:a', '48k', '-f', 'mp3'],
    },
];

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
    console.log("[Whisperina] Starting OpenAI transcription (non-streaming).");
    const subtitlePath = utils.resolvePath("@tmp/whisper_tmp.wav.srt");
    const rawResponsePath = `${subtitlePath}.openai.json`;

    const preparedAudio = await prepareAudioForOpenAI(tempWavName);

    try {
        const responseBody = await executeOpenAIRequest(preparedAudio);

        // Save raw response for debugging
        file.write(rawResponsePath, responseBody);
        console.log(`[Whisperina][OpenAI] Saved raw response to ${rawResponsePath}`);

        let srtContent = "";

        // Try to parse as JSON
        try {
            const json = JSON.parse(responseBody);
            // Handle standard OpenAI verbose_json or similar formats
            if (json.segments && Array.isArray(json.segments)) {
                 const segments = json.segments.map(normalizeOpenAISegment).filter(Boolean);
                 srtContent = renderSegmentsToSrt(segments);
            } else if (json.text) {
                // Fallback to text-only
                 const segments = [{
                    text: json.text,
                    startMs: 0,
                    endMs: estimateDurationFromText(json.text)
                }].map(normalizeOpenAISegment);
                srtContent = renderSegmentsToSrt(segments);
            } else if (json.error) {
                throw new Error(json.error.message || JSON.stringify(json.error));
            } else {
                 // Unknown JSON structure, maybe it IS srt wrapped in JSON? Unlikely.
                 // If response_format was srt, we wouldn't be here (parsing would likely fail or it would be a string).
                 console.warn("Unknown JSON structure:", json);
            }
        } catch (e) {
            // Not JSON? Maybe it is raw SRT or VTT?
            if (responseBody.trim().startsWith("WEBVTT") || responseBody.includes("-->")) {
                 srtContent = responseBody;
            } else {
                 // Maybe it failed to parse JSON but wasn't SRT.
                 // Re-throw if it wasn't a "valid but not JSON" case?
                 // Actually if responseBody is just text, maybe treat as text?
                 console.warn("Failed to parse response as JSON, treating as text/srt if possible.", e);
                 if (responseBody.length > 0) {
                     srtContent = responseBody;
                 }
            }
        }

        if (!srtContent) {
            throw new Error("No subtitle content generated from API response.");
        }

        file.write(subtitlePath, srtContent);
        await persistSubtitleCopy(subtitlePath, sourceMediaPath);
        console.log("[Whisperina] OpenAI transcription finished.");
        return subtitlePath;
    } catch (error) {
        console.error(`[Whisperina] OpenAI transcription failed: ${error.message}`);
        throw error;
    }
}

async function executeOpenAIRequest(upload) {
    const apiKey = (preferences.get("openai_api_key") || "").trim();
    if (!apiKey) {
        throw new Error("OpenAI API key is not configured. Please update the preferences page.");
    }
    const baseUrl = (preferences.get("openai_base_url") || "https://api.openai.com/v1/audio/transcriptions").trim();
    const model = (preferences.get("openai_model") || "gpt-4o-transcribe-diarize").trim();
    // Default to verbose_json to get segments
    let responseFormat = (preferences.get("openai_response_format") || "").trim();
    if (!responseFormat || responseFormat === "diarized_json") {
         // diarized_json was for the streaming custom backend; fallback to verbose_json for standard compatibility
         // or if the custom backend supports verbose_json with diarization.
         responseFormat = "verbose_json";
    }

    console.log(`[Whisperina][OpenAI] Request -> model=${model}, response_format=${responseFormat}, endpoint=${baseUrl}`);

    const args = [
        "curl",
        "-sS",
        "-X", "POST",
        baseUrl,
        "-H", `Authorization: Bearer ${apiKey}`,
        "-F", `file=@${upload.path}`,
        "-F", `model=${model}`,
        "-F", `response_format=${responseFormat}`,
    ];

    const stdout = await execWrapped("/usr/bin/env", args, null, {silent: true});
    return stdout;
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
    // Standard OpenAI segments use 'id' as integer usually, but we convert to string or whatever
    const id = segment.id !== undefined ? String(segment.id) : `${startMs}-${endMs}`;
    const speaker = typeof segment.speaker === "string" ? segment.speaker.trim() : null;
    const formattedText = speaker ? `${speaker}: ${text}` : text;
    
    return {
        id,
        startMs,
        endMs,
        textLines: splitTextIntoLines(formattedText),
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
