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
const https = require('https');

const app = express();
const upload = multer({ dest: 'uploads/' });
const PORT = 3000;
const HTTPS_PORT = 443;

// 检测文本是否包含中文字符
function containsChinese(text) {
    return /[\u4e00-\u9fa5]/.test(text);
}

// 使用 LLM 将中文提示词翻译成英文
async function translateToEnglish(text, apiKey, model) {
    try {
        const response = await axios.post('https://openrouter.ai/api/v1/chat/completions', {
            model: model || 'google/gemini-2.0-flash-001',
            messages: [{
                role: 'user',
                content: `Translate the following image generation prompt to English. Output ONLY the English translation, nothing else. Keep the same level of detail and artistic style descriptions.

Text to translate:
${text}`
            }]
        }, {
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
            }
        });

        const translatedText = response.data.choices?.[0]?.message?.content;
        if (translatedText && !containsChinese(translatedText)) {
            console.log('Successfully translated prompt to English');
            return translatedText.trim();
        }
        return text; // 如果翻译失败，返回原文
    } catch (error) {
        console.error('Translation error:', error.message);
        return text; // 翻译失败时返回原文
    }
}

// SSL 证书配置
const SSL_KEY_PATH = path.join(__dirname, 'ssl', 'webroleplay.xyz.key');
const SSL_CERT_PATH = path.join(__dirname, 'ssl', 'webroleplay.xyz.pem');

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
                    description: '图片描述文本（必须使用英文）。详细描述想要生成的图片内容，包括主体、风格等。例如："A cute cat sitting on clouds"。如果用户提供中文描述，请先翻译成英文。'
                },
                negative_prompt: {
                    type: 'string',
                    description: '负面提示词（英文），大部分情况下不需要指定。',
                    default: ''
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
    },
    'flux-fast': {
        modelEndpoint: 'prunaai/flux-fast',  // 使用 model endpoint
        defaultImageSize: 1024,
        defaultAspectRatio: '1:1',
        guidance: 3.5,
        speedMode: 'Extra Juiced 🔥 (more speed)',
        outputFormat: 'jpg',
        outputQuality: 80,
        numInferenceSteps: 28,
        useModelEndpoint: true,
        isFluxFast: true  // 标记为 flux-fast 模型
    },
    'z-image-turbo': {
        modelEndpoint: 'prunaai/z-image-turbo',  // 使用 model endpoint
        defaultWidth: 1024,
        defaultHeight: 768,
        guidanceScale: 0,
        outputFormat: 'jpg',
        outputQuality: 80,
        numInferenceSteps: 8,
        useModelEndpoint: true,
        isZImageTurbo: true  // 标记为 z-image-turbo 模型
    },
    'flux-2-pro': {
        modelEndpoint: 'black-forest-labs/flux-2-pro',  // 使用 model endpoint
        defaultResolution: '1 MP',
        defaultAspectRatio: '1:1',
        outputFormat: 'webp',
        outputQuality: 80,
        safetyTolerance: 5,
        useModelEndpoint: true,
        isFlux2Pro: true  // 标记为 flux-2-pro 模型
    },
    'nano-banana-pro': {
        modelEndpoint: 'google/nano-banana-pro',  // 使用 model endpoint
        defaultResolution: '2K',
        defaultAspectRatio: '4:3',
        outputFormat: 'png',
        safetyFilterLevel: 'block_only_high',
        useModelEndpoint: true,
        isNanoBananaPro: true  // 标记为 nano-banana-pro 模型
    },
    'gpt-image-1.5': {
        modelEndpoint: 'openai/gpt-image-1.5',  // 使用 model endpoint
        defaultAspectRatio: '1:1',
        quality: 'high',
        background: 'auto',
        moderation: 'auto',
        outputFormat: 'webp',
        outputCompression: 90,
        inputFidelity: 'low',
        numberOfImages: 1,
        disableSafetyChecker: true,
        useModelEndpoint: true,
        isGptImage15: true  // 标记为 gpt-image-1.5 模型
    },
    'qwen-image': {
        modelEndpoint: 'qwen/qwen-image',  // 使用 model endpoint
        defaultAspectRatio: '16:9',
        goFast: true,
        guidance: 4,
        strength: 0.9,
        imageSize: 'optimize_for_quality',
        loraScale: 1,
        outputFormat: 'webp',
        enhancePrompt: false,
        outputQuality: 80,
        negativePrompt: ' ',
        disableSafetyChecker: true,
        numInferenceSteps: 50,
        useModelEndpoint: true,
        isQwenImage: true  // 标记为 qwen-image 模型
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

                        if (imageModelConfig.isFluxFast) {
                            // Flux Fast 模型
                            imageInput = {
                                seed: -1,
                                prompt: args.prompt,
                                disable_safety_checker: true,
                                guidance: imageModelConfig.guidance,
                                image_size: imageModelConfig.defaultImageSize,
                                speed_mode: imageModelConfig.speedMode,
                                aspect_ratio: args.aspect_ratio || imageModelConfig.defaultAspectRatio,
                                output_format: imageModelConfig.outputFormat,
                                output_quality: imageModelConfig.outputQuality,
                                num_inference_steps: imageModelConfig.numInferenceSteps
                            };
                            apiEndpoint = `https://api.replicate.com/v1/models/${imageModelConfig.modelEndpoint}/predictions`;
                            requestBody = { input: imageInput };
                        } else if (imageModelConfig.isZImageTurbo) {
                            // Z-Image Turbo 模型
                            imageInput = {
                                width: args.width || imageModelConfig.defaultWidth,
                                height: args.height || imageModelConfig.defaultHeight,
                                prompt: args.prompt,
                                disable_safety_checker: true,
                                output_format: imageModelConfig.outputFormat,
                                guidance_scale: imageModelConfig.guidanceScale,
                                output_quality: imageModelConfig.outputQuality,
                                num_inference_steps: imageModelConfig.numInferenceSteps
                            };
                            apiEndpoint = `https://api.replicate.com/v1/models/${imageModelConfig.modelEndpoint}/predictions`;
                            requestBody = { input: imageInput };
                        } else if (imageModelConfig.isFlux2Pro) {
                            // FLUX.2 [pro] 模型
                            imageInput = {
                                prompt: args.prompt,
                                resolution: args.resolution || imageModelConfig.defaultResolution,
                                aspect_ratio: args.aspect_ratio || imageModelConfig.defaultAspectRatio,
                                input_images: [],
                                output_format: imageModelConfig.outputFormat,
                                output_quality: imageModelConfig.outputQuality,
                                safety_tolerance: imageModelConfig.safetyTolerance
                            };
                            apiEndpoint = `https://api.replicate.com/v1/models/${imageModelConfig.modelEndpoint}/predictions`;
                            requestBody = { input: imageInput };
                        } else if (imageModelConfig.isNanoBananaPro) {
                            // Nano Banana Pro 模型
                            imageInput = {
                                prompt: args.prompt,
                                resolution: args.resolution || imageModelConfig.defaultResolution,
                                image_input: [],
                                aspect_ratio: args.aspect_ratio || imageModelConfig.defaultAspectRatio,
                                output_format: imageModelConfig.outputFormat,
                                safety_filter_level: imageModelConfig.safetyFilterLevel
                            };
                            apiEndpoint = `https://api.replicate.com/v1/models/${imageModelConfig.modelEndpoint}/predictions`;
                            requestBody = { input: imageInput };
                        } else if (imageModelConfig.isGptImage15) {
                            // GPT Image 1.5 模型 (OpenAI on Replicate)
                            imageInput = {
                                prompt: args.prompt,
                                quality: imageModelConfig.quality,
                                background: imageModelConfig.background,
                                moderation: imageModelConfig.moderation,
                                aspect_ratio: args.aspect_ratio || imageModelConfig.defaultAspectRatio,
                                output_format: imageModelConfig.outputFormat,
                                input_fidelity: imageModelConfig.inputFidelity,
                                number_of_images: imageModelConfig.numberOfImages,
                                disable_safety_checker: imageModelConfig.disableSafetyChecker ? 1 : 0,
                                output_compression: imageModelConfig.outputCompression
                            };
                            apiEndpoint = `https://api.replicate.com/v1/models/${imageModelConfig.modelEndpoint}/predictions`;
                            requestBody = { input: imageInput };
                        } else if (imageModelConfig.isQwenImage) {
                            // Qwen Image 模型 (通义万象)
                            imageInput = {
                                prompt: args.prompt,
                                go_fast: imageModelConfig.goFast,
                                guidance: imageModelConfig.guidance,
                                strength: imageModelConfig.strength,
                                image_size: imageModelConfig.imageSize,
                                lora_scale: imageModelConfig.loraScale,
                                aspect_ratio: args.aspect_ratio || imageModelConfig.defaultAspectRatio,
                                output_format: imageModelConfig.outputFormat,
                                enhance_prompt: imageModelConfig.enhancePrompt,
                                output_quality: imageModelConfig.outputQuality,
                                negative_prompt: args.negative_prompt || imageModelConfig.negativePrompt,
                                disable_safety_checker: imageModelConfig.disableSafetyChecker,
                                num_inference_steps: imageModelConfig.numInferenceSteps
                            };
                            apiEndpoint = `https://api.replicate.com/v1/models/${imageModelConfig.modelEndpoint}/predictions`;
                            requestBody = { input: imageInput };
                        } else if (imageModelConfig.useAspectRatio) {
                            // 新模型使用 aspect_ratio 参数 (如 qwen-image-fast, p-image)
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

// 火山引擎语音合成接口 (TTS) - 普通模式 (v1 API)
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

        console.log('Submitting TTS request to Volcengine (v1)...');
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

// 火山引擎语音合成接口 (TTS) - 单向流式模式 (v3 API) - SSE 实时推送
app.get('/api/tts-stream', async (req, res) => {
    let { text, appid, access_key, resource_id, voice_type } = req.query;
    if (!text || !appid || !access_key || !resource_id) {
        return res.status(400).json({ success: false, message: '缺少参数: 需要 appid, access_key, resource_id' });
    }

    // URL 解码并去除空格
    text = decodeURIComponent(text);
    appid = appid.trim();
    access_key = access_key.trim();
    resource_id = resource_id.trim();
    if (voice_type) voice_type = voice_type.trim();

    // 设置 SSE 响应头
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.flushHeaders();

    try {
        const tts_url = 'https://openspeech.bytedance.com/api/v3/tts/unidirectional';
        const requestData = {
            user: { uid: 'roleplay_user' },
            req_params: {
                text: text,
                speaker: voice_type || 'zh_female_cancan_mars_bigtts',
                audio_params: {
                    format: 'pcm',      // 使用 PCM 格式便于流式播放
                    sample_rate: 24000,
                    enable_timestamp: false
                },
                additions: JSON.stringify({
                    explicit_language: 'zh',
                    disable_markdown_filter: true
                })
            }
        };

        console.log('Submitting TTS stream request to Volcengine (v3 SSE)...');
        console.log('AppID:', appid);
        console.log('Resource ID:', resource_id);
        console.log('Voice Type:', voice_type || 'zh_female_cancan_mars_bigtts');

        const response = await axios.post(tts_url, requestData, {
            headers: {
                'X-Api-App-Id': appid,
                'X-Api-Access-Key': access_key,
                'X-Api-Resource-Id': resource_id,
                'Content-Type': 'application/json',
                'Connection': 'keep-alive'
            },
            responseType: 'stream'
        });

        console.log('TTS Stream Response Status:', response.status);
        console.log('TTS Stream X-Tt-Logid:', response.headers['x-tt-logid']);

        let lineBuffer = '';
        let chunkIndex = 0;

        response.data.on('data', (chunk) => {
            lineBuffer += chunk.toString('utf-8');
            const lines = lineBuffer.split('\n');
            lineBuffer = lines.pop(); // 保留最后一个不完整的行

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const data = JSON.parse(line);
                    if (data.code === 0 && data.data) {
                        // 实时推送音频块给前端
                        res.write(`data: ${JSON.stringify({ type: 'audio', data: data.data, index: chunkIndex++ })}\n\n`);
                    } else if (data.code === 20000000) {
                        // 合成完成
                        console.log('TTS Stream completed, total chunks:', chunkIndex);
                        if (data.usage) {
                            console.log('TTS Usage:', data.usage);
                        }
                    } else if (data.code > 0 && data.code !== 20000000) {
                        console.error('TTS Stream error:', data);
                        res.write(`data: ${JSON.stringify({ type: 'error', message: data.message || 'TTS error' })}\n\n`);
                    }
                } catch (e) {
                    console.error('Parse line error:', e.message);
                }
            }
        });

        response.data.on('end', () => {
            // 处理最后一行
            if (lineBuffer.trim()) {
                try {
                    const data = JSON.parse(lineBuffer);
                    if (data.code === 0 && data.data) {
                        res.write(`data: ${JSON.stringify({ type: 'audio', data: data.data, index: chunkIndex++ })}\n\n`);
                    }
                } catch (e) {
                    // ignore
                }
            }
            // 发送结束信号
            res.write(`data: ${JSON.stringify({ type: 'end', totalChunks: chunkIndex })}\n\n`);
            res.end();
        });

        response.data.on('error', (err) => {
            console.error('TTS Stream error:', err);
            res.write(`data: ${JSON.stringify({ type: 'error', message: err.message })}\n\n`);
            res.end();
        });

        // 客户端断开连接时清理
        req.on('close', () => {
            console.log('Client closed SSE connection');
            response.data.destroy();
        });

    } catch (error) {
        const errorDetail = error.response?.data || error.message;
        console.error('TTS Stream Error:', errorDetail);
        res.write(`data: ${JSON.stringify({ type: 'error', message: typeof errorDetail === 'string' ? errorDetail : 'TTS 流式请求失败' })}\n\n`);
        res.end();
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
        MESSAGE_FLAGS.NO_SEQUENCE,
        SERIALIZATION.JSON,
        COMPRESSION.GZIP
    );
    const payloadBytes = Buffer.from(JSON.stringify(payload));
    const compressedPayload = zlib.gzipSync(payloadBytes);
    
    // NO sequence buffer for NO_SEQUENCE flag (matches Python demo)
    
    const sizeBuf = Buffer.alloc(4);
    sizeBuf.writeUInt32BE(compressedPayload.length);
    
    return Buffer.concat([header, sizeBuf, compressedPayload]);
}

function constructAudioRequest(seq, audioData, isLast) {
    const header = constructHeader(
        MESSAGE_TYPE.AUDIO_ONLY_REQUEST,
        isLast ? MESSAGE_FLAGS.NEG_SEQUENCE : MESSAGE_FLAGS.NO_SEQUENCE,
        SERIALIZATION.JSON, // Python demo uses JSON serialization flag even for audio
        COMPRESSION.GZIP
    );
    
    // NO sequence buffer for NO_SEQUENCE or NEG_SEQUENCE flag (matches Python demo)
    
    const compressedAudio = zlib.gzipSync(audioData);
    const sizeBuf = Buffer.alloc(4);
    sizeBuf.writeUInt32BE(compressedAudio.length);
    
    return Buffer.concat([header, sizeBuf, compressedAudio]);
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

function convertToPcm(inputPath) {
    return new Promise((resolve, reject) => {
        const args = [
            '-i', inputPath,
            '-acodec', 'pcm_s16le',
            '-ac', '1',
            '-ar', '16000',
            '-f', 's16le',
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
// 按照官方 Python 示例 (streaming_asr_demo.py) 实现
app.post('/api/asr', upload.single('audio'), async (req, res) => {
    const { appid, token, cluster } = req.body;
    const audioFile = req.file;
    const SUCCESS_CODE = 1000;

    if (!audioFile || !appid || !token) {
        return res.status(400).json({ success: false, message: '缺少参数' });
    }

    let ws = null;
    try {
        // 1. Convert audio to required format (PCM s16le, 16k, 1ch)
        const pcmBuffer = await convertToPcm(audioFile.path);

        // 2. Setup WebSocket Connection
        let targetResource = cluster || 'volcengine_streaming_common';
        if (targetResource === 'volc_auc_common') {
            targetResource = 'volcengine_streaming_common';
        }

        const wsUrl = `wss://openspeech.bytedance.com/api/v2/asr`;
        const reqId = crypto.randomUUID();

        const headers = {
            "Authorization": `Bearer; ${token.trim()}`
        };

        ws = new WebSocket(wsUrl, { headers });

        // 辅助函数：等待并解析一条消息
        function waitForMessage(ws, timeoutMs = 10000) {
            return new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('等待服务器响应超时'));
                }, timeoutMs);

                const onMessage = (data) => {
                    clearTimeout(timeout);
                    ws.off('message', onMessage);
                    ws.off('error', onError);
                    const response = parseResponse(data);
                    resolve(response);
                };

                const onError = (err) => {
                    clearTimeout(timeout);
                    ws.off('message', onMessage);
                    ws.off('error', onError);
                    reject(err);
                };

                ws.on('message', onMessage);
                ws.on('error', onError);
            });
        }

        await new Promise((resolve, reject) => {
            ws.on('open', resolve);
            ws.on('error', reject);
        });

        // 3. Send Full Client Request
        const requestPayload = {
            app: {
                appid: appid.trim(),
                cluster: targetResource,
                token: token.trim()
            },
            user: {
                uid: "roleplay_chat_user"
            },
            request: {
                reqid: reqId,
                nbest: 1,
                workflow: "audio_in,resample,partition,vad,fe,decode,itn,nlu_punctuate",
                show_language: false,
                show_utterances: true,
                result_type: "full",
                sequence: 1
            },
            audio: {
                format: "raw",
                rate: 16000,
                language: "zh-CN",
                bits: 16,
                channel: 1,
                codec: "raw"
            }
        };

        const fullRequest = constructFullRequest(1, requestPayload);
        ws.send(fullRequest);

        // 等待 Full Request 的响应
        const fullResponse = await waitForMessage(ws);
        if (fullResponse?.payloadMsg?.code !== SUCCESS_CODE) {
            throw new Error(`Full Request failed: code=${fullResponse?.payloadMsg?.code}, message=${fullResponse?.payloadMsg?.message}`);
        }

        // 4. Send Audio Chunks
        const CHUNK_SIZE = 16000 * 2 * 0.1; // 100ms chunks
        let offset = 0;
        let finalResultText = '';
        let seq = 1;

        // 切分音频数据
        const chunks = [];
        while (offset < pcmBuffer.length) {
            const end = Math.min(offset + CHUNK_SIZE, pcmBuffer.length);
            chunks.push(pcmBuffer.subarray(offset, end));
            offset += CHUNK_SIZE;
        }

        // 发送每个音频块并等待响应
        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            const isLast = (i === chunks.length - 1);

            const audioRequest = constructAudioRequest(seq++, chunk, isLast);
            ws.send(audioRequest);

            const audioResponse = await waitForMessage(ws);

            if (audioResponse?.msgType === MESSAGE_TYPE.SERVER_ERROR_RESPONSE) {
                throw new Error(`ASR Server Error: ${audioResponse.errorCode}`);
            }

            if (audioResponse?.payloadMsg) {
                const result = audioResponse.payloadMsg;

                if (result.code !== SUCCESS_CODE) {
                    throw new Error(`Audio chunk failed: code=${result.code}, message=${result.message}`);
                }

                // 更新最终结果 - result 是数组格式
                if (result.result && Array.isArray(result.result) && result.result.length > 0) {
                    const firstResult = result.result[0];
                    if (firstResult.text) {
                        finalResultText = firstResult.text;
                    }
                } else if (result.result && result.result.text) {
                    finalResultText = result.result.text;
                }

                if (result.text && !finalResultText) {
                    finalResultText = result.text;
                }
            }
        }

        // 等待最终识别结果
        let waitCount = 0;
        const MAX_WAIT = 10;

        while (waitCount < MAX_WAIT) {
            try {
                const finalResponse = await waitForMessage(ws, 5000);
                waitCount++;

                if (finalResponse?.payloadMsg) {
                    const result = finalResponse.payloadMsg;

                    if (result.result && Array.isArray(result.result) && result.result.length > 0) {
                        finalResultText = result.result[0].text || finalResultText;
                    }

                    if (result.sequence < 0) {
                        break;
                    }
                }
            } catch (timeoutErr) {
                break;
            }
        }

        ws.close();
        if (fs.existsSync(audioFile.path)) fs.unlinkSync(audioFile.path);

        if (finalResultText) {
            res.json({ success: true, text: finalResultText });
        } else {
            res.status(200).json({ success: true, text: "(未识别到有效语音)" });
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
    const { apiToken, prompt, negative_prompt, width, height, num_inference_steps, guidance_scale, model, aspect_ratio, creativity, openrouterApiKey } = req.body;

    if (!apiToken || !prompt) {
        return res.status(400).json({ success: false, message: '缺少必要参数：apiToken 和 prompt' });
    }

    try {
        // 检测并翻译中文提示词
        let finalPrompt = prompt;
        if (containsChinese(prompt)) {
            console.log('检测到中文提示词，正在翻译为英文...');
            if (openrouterApiKey) {
                finalPrompt = await translateToEnglish(prompt, openrouterApiKey);
                console.log('翻译后的提示词:', finalPrompt);
            } else {
                console.warn('未提供 OpenRouter API Key，无法翻译中文提示词');
            }
        }

        // 获取模型配置
        const restModelConfig = IMAGE_MODELS[model] || IMAGE_MODELS['realistic-vision-v5.1'];

        console.log('Submitting text-to-image request to Replicate...');
        console.log('Model:', model || 'realistic-vision-v5.1');
        console.log('Prompt:', finalPrompt);

        // 构建请求参数
        let restImageInput;
        let restApiEndpoint;
        let restRequestBody;

        if (restModelConfig.isFluxFast) {
            // Flux Fast 模型
            restImageInput = {
                seed: -1,
                prompt: finalPrompt,
                disable_safety_checker: true,
                guidance: restModelConfig.guidance,
                image_size: restModelConfig.defaultImageSize,
                speed_mode: restModelConfig.speedMode,
                aspect_ratio: aspect_ratio || restModelConfig.defaultAspectRatio,
                output_format: restModelConfig.outputFormat,
                output_quality: restModelConfig.outputQuality,
                num_inference_steps: num_inference_steps || restModelConfig.numInferenceSteps
            };
            restApiEndpoint = `https://api.replicate.com/v1/models/${restModelConfig.modelEndpoint}/predictions`;
            restRequestBody = { input: restImageInput };
        } else if (restModelConfig.isZImageTurbo) {
            // Z-Image Turbo 模型
            restImageInput = {
                width: width || restModelConfig.defaultWidth,
                height: height || restModelConfig.defaultHeight,
                prompt: finalPrompt,
                disable_safety_checker: true,
                output_format: restModelConfig.outputFormat,
                guidance_scale: restModelConfig.guidanceScale,
                output_quality: restModelConfig.outputQuality,
                num_inference_steps: num_inference_steps || restModelConfig.numInferenceSteps
            };
            restApiEndpoint = `https://api.replicate.com/v1/models/${restModelConfig.modelEndpoint}/predictions`;
            restRequestBody = { input: restImageInput };
        } else if (restModelConfig.isFlux2Pro) {
            // FLUX.2 [pro] 模型
            restImageInput = {
                prompt: finalPrompt,
                resolution: req.body.resolution || restModelConfig.defaultResolution,
                aspect_ratio: aspect_ratio || restModelConfig.defaultAspectRatio,
                input_images: [],
                output_format: restModelConfig.outputFormat,
                output_quality: restModelConfig.outputQuality,
                safety_tolerance: restModelConfig.safetyTolerance
            };
            restApiEndpoint = `https://api.replicate.com/v1/models/${restModelConfig.modelEndpoint}/predictions`;
            restRequestBody = { input: restImageInput };
        } else if (restModelConfig.isNanoBananaPro) {
            // Nano Banana Pro 模型
            restImageInput = {
                prompt: finalPrompt,
                resolution: req.body.resolution || restModelConfig.defaultResolution,
                image_input: [],
                aspect_ratio: aspect_ratio || restModelConfig.defaultAspectRatio,
                output_format: restModelConfig.outputFormat,
                safety_filter_level: restModelConfig.safetyFilterLevel
            };
            restApiEndpoint = `https://api.replicate.com/v1/models/${restModelConfig.modelEndpoint}/predictions`;
            restRequestBody = { input: restImageInput };
        } else if (restModelConfig.isGptImage15) {
            // GPT Image 1.5 模型 (OpenAI on Replicate)
            restImageInput = {
                prompt: finalPrompt,
                quality: restModelConfig.quality,
                background: restModelConfig.background,
                moderation: restModelConfig.moderation,
                aspect_ratio: aspect_ratio || restModelConfig.defaultAspectRatio,
                output_format: restModelConfig.outputFormat,
                input_fidelity: restModelConfig.inputFidelity,
                number_of_images: restModelConfig.numberOfImages,
                disable_safety_checker: restModelConfig.disableSafetyChecker ? 1 : 0,
                output_compression: restModelConfig.outputCompression
            };
            restApiEndpoint = `https://api.replicate.com/v1/models/${restModelConfig.modelEndpoint}/predictions`;
            restRequestBody = { input: restImageInput };
        } else if (restModelConfig.isQwenImage) {
            // Qwen Image 模型 (通义万象)
            restImageInput = {
                prompt: finalPrompt,
                go_fast: restModelConfig.goFast,
                guidance: restModelConfig.guidance,
                strength: restModelConfig.strength,
                image_size: restModelConfig.imageSize,
                lora_scale: restModelConfig.loraScale,
                aspect_ratio: aspect_ratio || restModelConfig.defaultAspectRatio,
                output_format: restModelConfig.outputFormat,
                enhance_prompt: restModelConfig.enhancePrompt,
                output_quality: restModelConfig.outputQuality,
                negative_prompt: negative_prompt || restModelConfig.negativePrompt,
                disable_safety_checker: restModelConfig.disableSafetyChecker,
                num_inference_steps: num_inference_steps || restModelConfig.numInferenceSteps
            };
            restApiEndpoint = `https://api.replicate.com/v1/models/${restModelConfig.modelEndpoint}/predictions`;
            restRequestBody = { input: restImageInput };
        } else if (restModelConfig.useAspectRatio) {
            // 新模型使用 aspect_ratio 参数 (如 qwen-image-fast, p-image)
            restImageInput = {
                prompt: finalPrompt,
                disable_safety_checker: true,
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
                prompt: finalPrompt,
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

        if (result.status === 'succeeded' && result.output) {
            // output 可能是数组或单个字符串
            const imageUrl = Array.isArray(result.output) ? result.output[0] : result.output;
            console.log('Extracted imageUrl:', imageUrl);
            res.json({
                success: true,
                imageUrl: imageUrl,
                prompt: finalPrompt,
                originalPrompt: prompt // 保留原始提示词供参考
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

// 启动 HTTP 服务器
app.listen(PORT, () => {
    console.log(`HTTP 服务器运行在 http://localhost:${PORT}`);
});

// 启动 HTTPS 服务器（如果存在 SSL 证书）
if (fs.existsSync(SSL_KEY_PATH) && fs.existsSync(SSL_CERT_PATH)) {
    try {
        const sslOptions = {
            key: fs.readFileSync(SSL_KEY_PATH),
            cert: fs.readFileSync(SSL_CERT_PATH)
        };

        https.createServer(sslOptions, app).listen(HTTPS_PORT, () => {
            console.log(`HTTPS 服务器运行在 https://0.0.0.0:${HTTPS_PORT}`);
        });
    } catch (error) {
        console.error('启动 HTTPS 服务器失败:', error.message);
        console.log('请确保 SSL 证书文件格式正确');
    }
} else {
    console.log('未找到 SSL 证书文件，HTTPS 服务器未启动');
    console.log(`请将证书文件放置到：`);
    console.log(`  - 私钥: ${SSL_KEY_PATH}`);
    console.log(`  - 证书: ${SSL_CERT_PATH}`);
}
