const {event, utils, console, file} = iina;

const SERVER_PID_FILE = "@tmp/whisper_server.pid";

cleanupLingeringServer();

event.on("iina.window-will-close", cleanupLingeringServer);
event.on("iina.window-did-close", cleanupLingeringServer);

function cleanupLingeringServer() {
    killRecordedWhisperServer().catch(error => {
        console.warn(`Failed to cleanup whisper-server: ${error.message}`);
    });
}

async function killRecordedWhisperServer() {
    const pidPath = resolvePidPath();
    if (!pidPath || !file.exists(pidPath)) {
        return;
    }
    const pidContent = file.read(pidPath);
    const pid = parseInt((pidContent || "").trim(), 10);
    if (!Number.isFinite(pid)) {
        safeDelete(pidPath);
        return;
    }
    await sendSignal(pid, "-TERM");
    await delay(200);
    await sendSignal(pid, "-KILL");
    safeDelete(pidPath);
}

async function sendSignal(pid, signal) {
    try {
        await utils.exec("/bin/kill", [signal, `${pid}`]);
    } catch (error) {
        if (/No such process/i.test(error?.message || "")) {
            return;
        }
        throw error;
    }
}

function resolvePidPath() {
    try {
        return utils.resolvePath(SERVER_PID_FILE);
    } catch (error) {
        console.warn(`Unable to resolve PID file: ${error.message}`);
        return null;
    }
}

function safeDelete(path) {
    try {
        if (file.exists(path)) {
            file.delete(path);
        }
    } catch (error) {
        console.warn(`Unable to delete PID file ${path}: ${error.message}`);
    }
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
