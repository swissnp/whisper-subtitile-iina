import {listModels} from "./models";
import {transcribe, isOpenAIMode} from "./transcribe";

const {subtitle, preferences, console} = iina;

subtitle.registerProvider("whisper", {
    search: async () => {
        logCurrentSettings();
        if (isOpenAIMode()) {
            return [
                subtitle.item({
                    id: "openai",
                    name: "OpenAI Streaming",
                    size: "cloud",
                    sha: "n/a",
                    format: "srt",
                }),
            ];
        }
        return listModels().map(model => subtitle.item({
            id: model.name, name: model.name, size: model.size, sha: model.sha, format: "srt",
        }));
    }, description: (item) => ({
        name: item.data.id, left: item.data.size, right: item.data.sha,
    }), download: async (item) => {
        return Promise.resolve(transcribe(item.data.id));
    },
});

function logCurrentSettings() {
    const mode = (preferences.get("transcriber_mode") || "whisper_server").toString();
    if (mode === "openai") {
        const model = preferences.get("openai_model") || "gpt-4o-transcribe-diarize";
        const responseFormat = preferences.get("openai_response_format") || "json";
        const chunking = preferences.get("openai_chunking_strategy") || "auto";
        const endpoint = preferences.get("openai_base_url") || "https://api.openai.com/v1/audio/transcriptions";
        console.log(`[Whisperina] Provider search -> mode=openai, model=${model}, format=${responseFormat}, chunking=${chunking}, endpoint=${endpoint}`);
    } else {
        const serverPath = preferences.get("wserver_path") || "(unset)";
        const host = preferences.get("wserver_host") || "127.0.0.1";
        const port = preferences.get("wserver_port") || "17896";
        console.log(`[Whisperina] Provider search -> mode=whisper_server, path=${serverPath}, host=${host}, port=${port}`);
    }
}
