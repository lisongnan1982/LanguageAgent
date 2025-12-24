const express = require('express');
const bodyParser = require('body-parser');
const path = require('path');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const crypto = require('crypto');
const WebSocket = require('ws');
const zlib = require('zlib');
const { execFile } = require('child_process');
const ffmpegPath = require('ffmpeg-static');

const app = express();
const upload = multer({ dest: 'uploads/' });
const PORT = 3000;

app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));

// 服务主页面
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// 定义文生图工具（供 LLM 调用）
const TEXT_TO_IMAGE_TOOL = {
    type: 'function',
    function: {
        name: 'generate_image',
        description: '根据文本描述生成图片。当用户要求生成、创建、画图片时使用此工具。**重要：prompt 必须使用英文描述，如果用户提供中文描述，你需要先将其翻译成详细的英文提示词。**',
        parameters: {
            type: 'object',
            properties: {
                prompt: {
                    type: 'string',
                    description: '图片描述文本（必须使用英文）。详细描述想要生成的图片内容，包括主体、风格、质量等。例如："A cute cat sitting on clouds"。如果用户提供中文描述，请先翻译成英文。'
                },
                negative_prompt: {
                    type: 'string',
                    description: '负面提示词（英文），描述不想在图片中出现的内容',
                    default: '(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck'
                },
                width: {
                    type: 'number',
                    description: '图片宽度（像素），默认 512',
                    default: 512,
                    enum: [512, 768, 1024]
                },
                height: {
                    type: 'number',
                    description: '图片高度（像素），默认 728',
                    default: 728,
                    enum: [512, 728, 768, 1024]
                }
            },
            required: ['prompt']
        }
    }
};

// 图片生成模型配置
const IMAGE_MODELS = {
    'realistic-vision-v5.1': {
        version: 'lucataco/realistic-vision-v5.1:2c8e954decbf70b7607a4414e5785ef9e4de4b8c51d50fb8b8b349160e0ef6bb',
        defaultWidth: 512,
        defaultHeight: 512,
        steps: 20,
        guidance: 5,
        scheduler: 'EulerA'
    },
    'realvisxl-v3.0-turbo': {
        version: 'adirik/realvisxl-v3.0-turbo:3dc73c805b11b4b01a60555e532fd3ab3f0e60d26f6584d9b8ba7e1b95858243',
        defaultWidth: 768,
        defaultHeight: 768,
        steps: 25,
        guidance: 2,
        scheduler: 'DPM++_SDE_Karras'
    },
    'dreamshaper-xl-turbo': {
        version: 'lucataco/dreamshaper-xl-turbo:0a1710e0187b01a255302738ca0158ff02a22f4638679533e111082f9dd1b615',
        defaultWidth: 1024,
        defaultHeight: 1024,
        steps: 7,           // Turbo 模型只需要 7 步
        guidance: 2,        // 低 guidance 效果更好
        scheduler: 'K_EULER'  // 必须指定调度器
    },
    'qwen-image-fast': {
        version: 'prunaai/qwen-image-fast:01b324d214eb4870ff424dc4215c067759c4c01a8751e327a434e2b16054db2f',
        defaultAspectRatio: '1:1',
        creativity: 0.62,
        disable_safety_checker: true,
        useAspectRatio: true  // 标记使用 aspect_ratio 而非 width/height
    },
    'p-image': {
        modelEndpoint: 'prunaai/p-image',  // 使用 model endpoint 而非 version
        defaultAspectRatio: '16:9',
        promptUpsampling: false,
        useAspectRatio: true,
        disable_safety_checker: true,
        useModelEndpoint: true  // 标记使用 model endpoint
    }
};

// LLM 代理接口 - 解决前端直接调用 OpenRouter 的 CORS 和身份验证问题
// 支持 function calling（工具调用）
app.post('/api/proxy-llm', async (req, res) => {
    const { apiKey, model, messages, response_format, tools, tool_choice, replicateToken, replicateModel } = req.body;
    if (!apiKey || !model || !messages) {
        return res.status(400).json({ success: false, message: '缺少必要参数' });
    }

    // 获取图片模型配置
    const imageModelConfig = IMAGE_MODELS[replicateModel] || IMAGE_MODELS['realistic-vision-v5.1'];

    try {
        // 构建请求参数
        const requestBody = {
            model,
            messages,
            response_format
        };

        // 如果提供了工具，添加到请求中
        if (tools && tools.length > 0) {
            requestBody.tools = tools;
            if (tool_choice) {
                requestBody.tool_choice = tool_choice;
            }
        }

        const response = await axios.post('https://openrouter.ai/api/v1/chat/completions', requestBody, {
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://github.com/Roo-Code/Roo-Code',
                'X-Title': 'RolePlay Chat'
            }
        });

        const data = response.data;

        // 检查是否有工具调用
        if (data.choices && data.choices[0].message.tool_calls) {
            const toolCalls = data.choices[0].message.tool_calls;
            const toolResults = [];

            // 处理每个工具调用
            for (const toolCall of toolCalls) {
                if (toolCall.function.name === 'generate_image') {
                    try {
                        const args = JSON.parse(toolCall.function.arguments);

                        if (!replicateToken) {
                            toolResults.push({
                                tool_call_id: toolCall.id,
                                role: 'tool',
                                name: 'generate_image',
                                content: JSON.stringify({
                                    error: '未配置 Replicate API Token，请在设置中配置'
                                })
                            });
                            continue;
                        }

                        // 构建图片生成请求参数
                        let imageInput;
                        let apiEndpoint;
                        let requestBody;

                        if (imageModelConfig.useAspectRatio) {
                            // 新模型使用 aspect_ratio 参数
                            imageInput = {
                                prompt: args.prompt,
                                disable_safety_checker: true,
                                aspect_ratio: args.aspect_ratio || imageModelConfig.defaultAspectRatio
                            };

                            if (imageModelConfig.creativity !== undefined) {
                                imageInput.creativity = args.creativity || imageModelConfig.creativity;
                            }
                            if (imageModelConfig.promptUpsampling !== undefined) {
                                imageInput.prompt_upsampling = imageModelConfig.promptUpsampling;
                            }

                            if (imageModelConfig.useModelEndpoint) {
                                // p-image 使用 model endpoint
                                apiEndpoint = `https://api.replicate.com/v1/models/${imageModelConfig.modelEndpoint}/predictions`;
                                requestBody = { input: imageInput };
                            } else {
                                // qwen-image-fast 使用 version endpoint
                                apiEndpoint = 'https://api.replicate.com/v1/predictions';
                                requestBody = {
                                    version: imageModelConfig.version,
                                    input: imageInput
                                };
                            }
                        } else {
                            // 传统模型使用 width/height 参数
                            imageInput = {
                                seed: Math.floor(Math.random() * 10000),
                                steps: imageModelConfig.steps,
                                width: args.width || imageModelConfig.defaultWidth,
                                height: args.height || imageModelConfig.defaultHeight,
                                prompt: args.prompt,
                                disable_safety_checker: true,
                                negative_prompt: args.negative_prompt || '(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck'
                            };

                            // 添加模型特定参数
                            if (imageModelConfig.guidance) {
                                imageInput.guidance = imageModelConfig.guidance;
                            }
                            if (imageModelConfig.scheduler) {
                                imageInput.scheduler = imageModelConfig.scheduler;
                            }

                            apiEndpoint = 'https://api.replicate.com/v1/predictions';
                            requestBody = {
                                version: imageModelConfig.version,
                                input: imageInput
                            };
                        }

                        console.log(`使用图片模型: ${replicateModel || 'realistic-vision-v5.1'}`);
                        console.log(`API 端点: ${apiEndpoint}`);

                        // 调用 Replicate API 生成图片
                        const imageResponse = await axios.post(
                            apiEndpoint,
                            requestBody,
                            {
                                headers: {
                                    'Authorization': `Bearer ${replicateToken}`,
                                    'Content-Type': 'application/json',
                                    'Prefer': 'wait'
                                },
                                timeout: 120000
                            }
                        );

                        // 调试日志：查看 Replicate API 的响应
                        console.log('Replicate API response:', JSON.stringify(imageResponse.data, null, 2));

                        // 处理不同的输出格式
                        let imageUrl = null;
                        const output = imageResponse.data.output;
                        if (Array.isArray(output) && output.length > 0) {
                            imageUrl = output[0];
                        } else if (typeof output === 'string') {
                            imageUrl = output;
                        }

                        console.log('Extracted imageUrl:', imageUrl);

                        if (!imageUrl) {
                            throw new Error('No image URL in response');
                        }

                        toolResults.push({
                            tool_call_id: toolCall.id,
                            role: 'tool',
                            name: 'generate_image',
                            content: JSON.stringify({
                                success: true,
                                imageUrl: imageUrl,
                                prompt: args.prompt
                            })
                        });

                    } catch (error) {
                        console.error('Image generation error:', error.response?.data || error.message);
                        toolResults.push({
                            tool_call_id: toolCall.id,
                            role: 'tool',
                            name: 'generate_image',
                            content: JSON.stringify({
                                error: `图片生成失败: ${error.response?.data?.detail || error.message}`
                            })
                        });
                    }
                }
            }

            // 返回工具调用结果，让前端继续对话
            data.tool_results = toolResults;
        }

        res.json(data);
    } catch (error) {
        console.error('LLM Proxy Error:', error.response?.data || error.message);
        res.status(error.response?.status || 500).json(error.response?.data || { message: error.message });
    }
});

// 火山引擎语音合成接口 (TTS)
app.post('/api/tts', async (req, res) => {
    let { text, appid, token, cluster, voice_type } = req.body;
    if (!text || !appid || !token) {
        return res.status(400).json({ success: false, message: '缺少参数' });
    }

    // 去除可能存在的空格
    appid = appid.trim();
    token = token.trim();
    if (cluster) cluster = cluster.trim();
    if (voice_type) voice_type = voice_type.trim();

    try {
        const tts_url = 'https://openspeech.bytedance.com/api/v1/tts';
        const requestData = {
            app: { appid, token, cluster: cluster || 'volcano_tts' },
            user: { uid: 'roleplay_user' },
            audio: {
                voice_type: voice_type || 'zh_female_vv_uranus_bigtts',
                encoding: 'mp3',
                speed_ratio: 1.0,
                volume_ratio: 1.0,
                pitch_ratio: 1.0,
            },
            request: {
                reqid: crypto.randomUUID(),
                text: text,
                text_type: 'plain',
                operation: 'query',
            }
        };

        console.log('Submitting TTS request to Volcengine...');
        console.log('AppID:', appid);
        console.log('Cluster:', cluster || 'volcano_tts');
        console.log('Voice Type:', voice_type || 'zh_female_vv_uranus_bigtts');

        const response = await axios.post(tts_url, requestData, {
            headers: {
                'Authorization': `Bearer;${token}`,
                'Content-Type': 'application/json'
            }
        });

        if (response.data && response.data.data) {
            res.json({ success: true, audio: response.data.data });
        } else {
            console.error('TTS API Error Response:', response.data);
            let msg = response.data.message || `TTS 转换失败 (Code: ${response.data.code})`;
            if (msg.includes('requested resource not granted')) {
                msg = `资源未授权: 请检查您的 AppID 是否已在火山引擎控制台开通并授权了 Resource ID 为 "${cluster || 'volcano_tts'}" 的服务。`;
            }
            throw new Error(msg);
        }
    } catch (error) {
        const errorDetail = error.response?.data || error.message;
        console.error('TTS Error:', errorDetail);
        res.status(500).json({ 
            success: false, 
            message: typeof errorDetail === 'string' ? errorDetail : (errorDetail.message || 'TTS 请求失败'),
            detail: errorDetail
        });
    }
});

// Volcengine ASR Streaming Protocol Constants
const PROTOCOL_VERSION = 0b0001;
const HEADER_SIZE = 1;
const MESSAGE_TYPE = {
    FULL_CLIENT_REQUEST: 0b0001,
    AUDIO_ONLY_REQUEST: 0b0010,
    FULL_SERVER_RESPONSE: 0b1001,
    SERVER_ERROR_RESPONSE: 0b1111
};
const MESSAGE_FLAGS = {
    NO_SEQUENCE: 0b0000,
    POS_SEQUENCE: 0b0001,
    NEG_SEQUENCE: 0b0010,
    NEG_WITH_SEQUENCE: 0b0011
};
const SERIALIZATION = {
    NO: 0b0000,
    JSON: 0b0001
};
const COMPRESSION = {
    NO: 0b0000,
    GZIP: 0b0001
};

// Protocol Helpers
function constructHeader(msgType, msgFlags, serialization, compression) {
    const header = Buffer.alloc(4);
    header[0] = (PROTOCOL_VERSION << 4) | HEADER_SIZE;
    header[1] = (msgType << 4) | msgFlags;
    header[2] = (serialization << 4) | compression;
    header[3] = 0x00;
    return header;
}

function constructFullRequest(seq, payload) {
    const header = constructHeader(
        MESSAGE_TYPE.FULL_CLIENT_REQUEST,
        MESSAGE_FLAGS.POS_SEQUENCE,
        SERIALIZATION.JSON,
        COMPRESSION.GZIP
    );
    const payloadBytes = Buffer.from(JSON.stringify(payload));
    const compressedPayload = zlib.gzipSync(payloadBytes);
    
    const seqBuf = Buffer.alloc(4);
    seqBuf.writeInt32BE(seq);
    
    const sizeBuf = Buffer.alloc(4);
    sizeBuf.writeUInt32BE(compressedPayload.length);
    
    return Buffer.concat([header, seqBuf, sizeBuf, compressedPayload]);
}

function constructAudioRequest(seq, audioData, isLast) {
    const header = constructHeader(
        MESSAGE_TYPE.AUDIO_ONLY_REQUEST,
        isLast ? MESSAGE_FLAGS.NEG_WITH_SEQUENCE : MESSAGE_FLAGS.POS_SEQUENCE,
        SERIALIZATION.NO,
        COMPRESSION.GZIP
    );
    
    // Last package uses negative sequence in demonstration code
    const actualSeq = isLast ? -seq : seq;
    const seqBuf = Buffer.alloc(4);
    seqBuf.writeInt32BE(actualSeq);
    
    const compressedAudio = zlib.gzipSync(audioData);
    const sizeBuf = Buffer.alloc(4);
    sizeBuf.writeUInt32BE(compressedAudio.length);
    
    return Buffer.concat([header, seqBuf, sizeBuf, compressedAudio]);
}

function parseResponse(data) {
    if (data.length < 4) return null;
    const headerSize = data[0] & 0x0f;
    const msgType = data[1] >> 4;
    const msgFlags = data[1] & 0x0f;
    const serialization = data[2] >> 4;
    const compression = data[2] & 0x0f;
    
    let offset = headerSize * 4;
    
    // Skip sequence if present
    if ((msgFlags & 0x01) || (msgFlags & 0x03) === 0x03) { // POS_SEQUENCE or NEG_WITH_SEQUENCE
        offset += 4;
    }
    
    // Skip event if present (server implementation detail, usually not present in simple response)
    
    let payloadMsg = null;
    let payloadSize = 0;
    let errorCode = 0;
    
    if (msgType === MESSAGE_TYPE.FULL_SERVER_RESPONSE) {
        payloadSize = data.readUInt32BE(offset);
        offset += 4;
    } else if (msgType === MESSAGE_TYPE.SERVER_ERROR_RESPONSE) {
        errorCode = data.readInt32BE(offset);
        offset += 4;
        payloadSize = data.readUInt32BE(offset);
        offset += 4;
    }
    
    if (payloadSize > 0 && offset + payloadSize <= data.length) {
        let payload = data.subarray(offset, offset + payloadSize);
        try {
            if (compression === COMPRESSION.GZIP) {
                payload = zlib.gunzipSync(payload);
            }
            if (serialization === SERIALIZATION.JSON) {
                payloadMsg = JSON.parse(payload.toString());
            }
        } catch (e) {
            console.error('Payload parse error:', e);
        }
    }
    
    return { msgType, errorCode, payloadMsg };
}

function convertToWav(inputPath) {
    return new Promise((resolve, reject) => {
        const args = [
            '-i', inputPath,
            '-acodec', 'pcm_s16le',
            '-ac', '1',
            '-ar', '16000',
            '-f', 'wav',
            '-'
        ];
        
        execFile(ffmpegPath, args, { encoding: 'buffer', maxBuffer: 50 * 1024 * 1024 }, (error, stdout, stderr) => {
            if (error) {
                console.error('FFmpeg error:', stderr.toString());
                reject(error);
            } else {
                resolve(stdout);
            }
        });
    });
}

// 火山引擎语音识别接口 (ASR) - Streaming WebSocket Implementation
app.post('/api/asr', upload.single('audio'), async (req, res) => {
    const { appid, token, cluster } = req.body;
    const audioFile = req.file;

    if (!audioFile || !appid || !token) {
        return res.status(400).json({ success: false, message: '缺少参数' });
    }

    let ws = null;
    try {
        // 1. Convert audio to required format (WAV, 16k, 1ch, s16le)
        console.log('Converting audio...');
        const wavBuffer = await convertToWav(audioFile.path);
        console.log('Audio converted. Size:', wavBuffer.length);

        // 2. Setup WebSocket Connection
        // 兼容旧的 cluster 标识，如果为 volc_auc_common 则自动转换为 volc.bigasr.auc
        let targetResource = cluster || 'volc.bigasr.auc';
        if (targetResource === 'volc_auc_common') {
            targetResource = 'volc.bigasr.auc';
        }
        
        // Use streaming endpoint (sauc)
        const wsUrl = 'wss://openspeech.bytedance.com/api/v3/sauc/bigmodel';
        const reqId = crypto.randomUUID();
        
        const headers = {
            "X-Api-Resource-Id": targetResource,
            "X-Api-Request-Id": reqId,
            "X-Api-Access-Key": token.trim(),
            "X-Api-App-Key": appid.trim()
        };

        console.log('Connecting to ASR WebSocket:', wsUrl);
        ws = new WebSocket(wsUrl, { headers });

        await new Promise((resolve, reject) => {
            ws.on('open', resolve);
            ws.on('error', reject);
        });
        console.log('WebSocket Connected');

        // 3. Send Full Client Request
        let seq = 1;
        const requestPayload = {
            user: { uid: "roleplay_chat_user" },
            audio: {
                format: "wav",
                codec: "raw", // We are sending raw wav bytes (including header) as per demo
                rate: 16000,
                bits: 16,
                channel: 1
            },
            request: {
                model_name: "bigmodel",
                enable_itn: true,
                enable_punc: true,
                enable_ddc: true,
                show_utterances: true
            }
        };

        const fullRequest = constructFullRequest(seq++, requestPayload);
        ws.send(fullRequest);
        console.log('Sent Full Request');

        // 4. Send Audio Chunks
        // Skip WAV header (44 bytes) if strictly sending PCM?
        // Demo says "codec: raw" but input is "format: wav".
        // Demo reads file and converts to wav using ffmpeg and sends the whole thing including header.
        // So we send `wavBuffer` as is, but chunked.
        
        const CHUNK_SIZE = 16000 * 2 * 0.2; // 200ms chunks (16k sample rate * 2 bytes/sample * 0.2s)
        let offset = 0;
        let finalResultText = '';
        let completed = false;

        // Setup listener for results
        const processingPromise = new Promise((resolve, reject) => {
            ws.on('message', (data) => {
                const response = parseResponse(data);
                if (!response) return;

                if (response.msgType === MESSAGE_TYPE.SERVER_ERROR_RESPONSE) {
                    console.error('ASR Server Error:', response.errorCode);
                    reject(new Error(`ASR Server Error Code: ${response.errorCode}`));
                } else if (response.msgType === MESSAGE_TYPE.FULL_SERVER_RESPONSE) {
                    const result = response.payloadMsg;
                    if (result) {
                        // Check if final result
                        // In streaming, we might get partial results, but for this file-upload simulation,
                        // we wait for the final one or accumulate.
                        // The demo prints "Received response".
                        // Usually we look for 'result.text'.
                        if (result.result) {
                             // Assuming last message contains full text or we just take the last update
                             finalResultText = result.result.text;
                        }
                    }
                }
            });
            
            ws.on('close', () => {
                console.log('WebSocket Closed');
                resolve();
            });
            
            ws.on('error', (err) => {
                reject(err);
            });
        });

        // Streaming loop
        while (offset < wavBuffer.length) {
            const end = Math.min(offset + CHUNK_SIZE, wavBuffer.length);
            const chunk = wavBuffer.subarray(offset, end);
            const isLast = end >= wavBuffer.length;
            
            const audioRequest = constructAudioRequest(seq++, chunk, isLast);
            ws.send(audioRequest);
            
            offset += CHUNK_SIZE;
            // Slight delay to simulate stream? Not strictly necessary for file upload but good for stability
            await new Promise(r => setTimeout(r, 20)); 
        }
        console.log('Sent all audio chunks');

        // Wait for connection close or timeout?
        // Streaming ASR usually sends result then closes, or we close.
        // But since we sent "isLast", server should process and finish.
        // We need to wait a bit for final response.
        
        // Wait for a few seconds max for results, then close if not closed
        const timeoutPromise = new Promise(r => setTimeout(r, 5000));
        await Promise.race([processingPromise, timeoutPromise]);
        
        ws.close();

        if (fs.existsSync(audioFile.path)) fs.unlinkSync(audioFile.path);

        // Fallback: if text is empty, maybe try to check if we got anything
        console.log('ASR Result:', finalResultText);
        
        if (finalResultText) {
            res.json({ success: true, text: finalResultText });
        } else {
            res.status(500).json({ success: false, message: '未获取到识别结果' });
        }

    } catch (error) {
        console.error('ASR Streaming Error:', error);
        if (ws) {
            try { ws.close(); } catch(e) {}
        }
        if (audioFile && fs.existsSync(audioFile.path)) fs.unlinkSync(audioFile.path);
        
        res.status(500).json({ 
            success: false, 
            message: '语音识别失败', 
            detail: error.message 
        });
    }
});

// ============================================
// MCP 协议端点 (集成在主服务器中)
// ============================================

// MCP 服务器信息
const MCP_SERVER_INFO = {
    name: 'text-to-image-server',
    version: '1.0.0',
    protocolVersion: '2024-11-05',
    capabilities: {
        tools: {}
    }
};

// MCP 工具定义
const MCP_TOOLS = [
    {
        name: 'generate_image',
        description: '根据文本描述生成图片。使用 Replicate 的 Realistic Vision v5.1 模型生成高质量写实风格图像。**重要：prompt 必须使用英文描述。**',
        inputSchema: {
            type: 'object',
            properties: {
                prompt: {
                    type: 'string',
                    description: '图片描述文本（必须使用英文）。详细描述想要生成的图片内容，包括主体、风格、质量等。'
                },
                negative_prompt: {
                    type: 'string',
                    description: '负面提示词（英文），描述不想在图片中出现的内容',
                    default: '(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck'
                },
                width: {
                    type: 'number',
                    description: '图片宽度（像素）',
                    default: 512
                },
                height: {
                    type: 'number',
                    description: '图片高度（像素）',
                    default: 728
                },
                steps: {
                    type: 'number',
                    description: '推理步数，越高质量越好但速度越慢',
                    default: 20
                },
                guidance: {
                    type: 'number',
                    description: '引导强度，控制生成图片与提示词的匹配程度',
                    default: 5
                }
            },
            required: ['prompt']
        }
    }
];

// MCP 协议端点：初始化
app.post('/mcp/initialize', (req, res) => {
    console.log('MCP Initialize request:', req.body);
    res.json({
        protocolVersion: MCP_SERVER_INFO.protocolVersion,
        capabilities: MCP_SERVER_INFO.capabilities,
        serverInfo: {
            name: MCP_SERVER_INFO.name,
            version: MCP_SERVER_INFO.version
        }
    });
});

// MCP 协议端点：列出工具
app.post('/mcp/tools/list', (req, res) => {
    console.log('MCP Tools list request');
    res.json({
        tools: MCP_TOOLS
    });
});

// MCP 协议端点：调用工具
app.post('/mcp/tools/call', async (req, res) => {
    const { name, arguments: args } = req.body.params || req.body;

    console.log('MCP Tool call:', name, args);

    if (name !== 'generate_image') {
        return res.status(400).json({
            error: {
                code: -32601,
                message: `Unknown tool: ${name}`
            }
        });
    }

    try {
        // 从请求头或参数中获取 API Token
        const apiToken = req.headers['x-replicate-token'] || args.api_token;

        if (!apiToken) {
            return res.status(400).json({
                error: {
                    code: -32602,
                    message: 'Missing Replicate API token. Please configure it in settings.'
                }
            });
        }

        // 获取模型配置（MCP 默认使用 realistic-vision-v5.1）
        const mcpModelConfig = IMAGE_MODELS[args.model] || IMAGE_MODELS['realistic-vision-v5.1'];

        // 构建图片生成请求参数
        let mcpImageInput;
        let mcpApiEndpoint;
        let mcpRequestBody;

        if (mcpModelConfig.useAspectRatio) {
            // 新模型使用 aspect_ratio 参数
            mcpImageInput = {
                prompt: args.prompt,
                aspect_ratio: args.aspect_ratio || mcpModelConfig.defaultAspectRatio
            };

            if (mcpModelConfig.creativity !== undefined) {
                mcpImageInput.creativity = args.creativity || mcpModelConfig.creativity;
            }
            if (mcpModelConfig.promptUpsampling !== undefined) {
                mcpImageInput.prompt_upsampling = mcpModelConfig.promptUpsampling;
            }

            if (mcpModelConfig.useModelEndpoint) {
                // p-image 使用 model endpoint
                mcpApiEndpoint = `https://api.replicate.com/v1/models/${mcpModelConfig.modelEndpoint}/predictions`;
                mcpRequestBody = { input: mcpImageInput };
            } else {
                // qwen-image-fast 使用 version endpoint
                mcpApiEndpoint = 'https://api.replicate.com/v1/predictions';
                mcpRequestBody = {
                    version: mcpModelConfig.version,
                    input: mcpImageInput
                };
            }
        } else {
            // 传统模型使用 width/height 参数
            mcpImageInput = {
                seed: Math.floor(Math.random() * 10000),
                steps: args.steps || mcpModelConfig.steps,
                width: args.width || mcpModelConfig.defaultWidth,
                height: args.height || mcpModelConfig.defaultHeight,
                prompt: args.prompt,
                disable_safety_checker: true,
                negative_prompt: args.negative_prompt || '(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck'
            };

            // 添加模型特定参数
            if (mcpModelConfig.guidance) {
                mcpImageInput.guidance = args.guidance || mcpModelConfig.guidance;
            }
            if (mcpModelConfig.scheduler) {
                mcpImageInput.scheduler = mcpModelConfig.scheduler;
            }

            mcpApiEndpoint = 'https://api.replicate.com/v1/predictions';
            mcpRequestBody = {
                version: mcpModelConfig.version,
                input: mcpImageInput
            };
        }

        console.log(`MCP 使用图片模型: ${args.model || 'realistic-vision-v5.1'}`);
        console.log(`MCP API 端点: ${mcpApiEndpoint}`);

        // 调用 Replicate API
        const response = await axios.post(
            mcpApiEndpoint,
            mcpRequestBody,
            {
                headers: {
                    'Authorization': `Bearer ${apiToken}`,
                    'Content-Type': 'application/json',
                    'Prefer': 'wait'
                },
                timeout: 120000
            }
        );

        const result = response.data;
        console.log('MCP Replicate API response:', JSON.stringify(result, null, 2));

        // 处理不同的输出格式
        let imageUrl = null;
        if (Array.isArray(result.output) && result.output.length > 0) {
            imageUrl = result.output[0];
        } else if (typeof result.output === 'string') {
            imageUrl = result.output;
        }

        console.log('MCP Extracted imageUrl:', imageUrl);

        if (!imageUrl) {
            throw new Error('No image URL in response');
        }

        res.json({
            content: [
                {
                    type: 'text',
                    text: `图片生成成功！\n\n提示词: ${args.prompt}\n图片URL: ${imageUrl}`
                },
                {
                    type: 'image',
                    data: imageUrl,
                    mimeType: 'image/png'
                }
            ]
        });

    } catch (error) {
        console.error('Error generating image:', error.response?.data || error.message);
        res.status(500).json({
            error: {
                code: -32603,
                message: `Failed to generate image: ${error.response?.data?.detail || error.message}`
            }
        });
    }
});

// ============================================
// 文生图 REST API 端点
// ============================================

// 文生图接口 (Text-to-Image) - 使用 Replicate API
app.post('/api/text-to-image', async (req, res) => {
    const { apiToken, prompt, negative_prompt, width, height, num_inference_steps, guidance_scale, model, aspect_ratio, creativity } = req.body;

    if (!apiToken || !prompt) {
        return res.status(400).json({ success: false, message: '缺少必要参数：apiToken 和 prompt' });
    }

    try {
        // 获取模型配置
        const restModelConfig = IMAGE_MODELS[model] || IMAGE_MODELS['realistic-vision-v5.1'];

        console.log('Submitting text-to-image request to Replicate...');
        console.log('Model:', model || 'realistic-vision-v5.1');
        console.log('Prompt:', prompt);

        // 构建请求参数
        let restImageInput;
        let restApiEndpoint;
        let restRequestBody;

        if (restModelConfig.useAspectRatio) {
            // 新模型使用 aspect_ratio 参数
            restImageInput = {
                prompt: prompt,
                aspect_ratio: aspect_ratio || restModelConfig.defaultAspectRatio
            };

            if (restModelConfig.creativity !== undefined) {
                restImageInput.creativity = creativity || restModelConfig.creativity;
            }
            if (restModelConfig.promptUpsampling !== undefined) {
                restImageInput.prompt_upsampling = restModelConfig.promptUpsampling;
            }

            if (restModelConfig.useModelEndpoint) {
                // p-image 使用 model endpoint
                restApiEndpoint = `https://api.replicate.com/v1/models/${restModelConfig.modelEndpoint}/predictions`;
                restRequestBody = { input: restImageInput };
            } else {
                // qwen-image-fast 使用 version endpoint
                restApiEndpoint = 'https://api.replicate.com/v1/predictions';
                restRequestBody = {
                    version: restModelConfig.version,
                    input: restImageInput
                };
            }
        } else {
            // 传统模型使用 width/height 参数
            restImageInput = {
                seed: Math.floor(Math.random() * 10000),
                steps: num_inference_steps || restModelConfig.steps,
                width: width || restModelConfig.defaultWidth,
                height: height || restModelConfig.defaultHeight,
                prompt: prompt,
                negative_prompt: negative_prompt || '(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck',
                disable_safety_checker: true
            };

            // 添加模型特定参数
            if (restModelConfig.guidance) {
                restImageInput.guidance = guidance_scale || restModelConfig.guidance;
            }
            if (restModelConfig.scheduler) {
                restImageInput.scheduler = restModelConfig.scheduler;
            }

            restApiEndpoint = 'https://api.replicate.com/v1/predictions';
            restRequestBody = {
                version: restModelConfig.version,
                input: restImageInput
            };
        }

        console.log(`API 端点: ${restApiEndpoint}`);

        const response = await axios.post(
            restApiEndpoint,
            restRequestBody,
            {
                headers: {
                    'Authorization': `Bearer ${apiToken}`,
                    'Content-Type': 'application/json',
                    'Prefer': 'wait'
                },
                timeout: 240000 // 240秒超时
            }
        );

        const result = response.data;
        console.log('Replicate response status:', result.status);

        if (result.status === 'succeeded' && result.output && result.output[0]) {
            res.json({
                success: true,
                imageUrl: result.output[0],
                prompt: prompt
            });
        } else if (result.status === 'processing') {
            // 如果还在处理中，返回预测ID供轮询
            res.json({
                success: false,
                processing: true,
                predictionId: result.id,
                message: '图片生成中，请稍后查询结果'
            });
        } else {
            throw new Error(`Image generation failed with status: ${result.status}`);
        }

    } catch (error) {
        const errorDetail = error.response?.data || error.message;
        console.error('Text-to-Image Error:', errorDetail);
        res.status(500).json({
            success: false,
            message: '图片生成失败',
            detail: typeof errorDetail === 'string' ? errorDetail : JSON.stringify(errorDetail)
        });
    }
});

app.listen(PORT, () => {
    console.log(`服务器运行在 http://0.0.0.0:${PORT}`);
});
