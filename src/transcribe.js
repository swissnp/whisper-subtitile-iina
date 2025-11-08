import {parse} from "shell-quote";

const {console, core, preferences, utils, http, file} = iina;

const HOME_PATH = '~/Library/Application Support/com.colliderli.iina/plugins/';
const SERVER_PID_FILE = "@tmp/whisper_server.pid";
const LOG_POLL_INTERVAL_MS = 1000;
const LOG_SEGMENT_REGEX = /^\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})]\s+(.*)$/;

let activeServerInfo = null;

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
    const monitor = startLogMonitor(serverInfo.logPath, subtitlePath);
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

function startLogMonitor(logPath, subtitlePath) {
    const seen = new Set();
    const segments = [];
    let stopRequested = false;
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
