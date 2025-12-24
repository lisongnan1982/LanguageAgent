const express = require('express');
const bodyParser = require('body-parser');
const path = require('path');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const crypto = require('crypto');

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
                    description: '图片描述文本（必须使用英文）。详细描述想要生成的图片内容，包括主体、风格、质量等。例如："A cute cat sitting on clouds, digital art style, high quality, 8k uhd"。如果用户提供中文描述，请先翻译成英文。'
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

// 火山引擎语音识别接口 (ASR) - 使用 v3/auc/bigmodel API
app.post('/api/asr', upload.single('audio'), async (req, res) => {
    const { appid, token, cluster } = req.body;
    const audioFile = req.file;

    if (!audioFile || !appid || !token) {
        return res.status(400).json({ success: false, message: '缺少参数' });
    }

    try {
        const audioData = fs.readFileSync(audioFile.path);
        const task_id = crypto.randomUUID();
        // 兼容旧的 cluster 标识，如果为 volc_auc_common 则自动转换为 volc.bigasr.auc
        let targetResource = cluster || 'volc.bigasr.auc';
        if (targetResource === 'volc_auc_common') {
            targetResource = 'volc.bigasr.auc';
        }
        
        console.log('Submitting ASR task to Volcengine v3/auc/bigmodel...');
        console.log('AppID:', appid.trim());
        console.log('Resource ID:', targetResource);

        // 1. 提交任务
        const submit_url = 'https://openspeech-direct.zijieapi.com/api/v3/auc/bigmodel/submit';
        const headers = {
            "X-Api-App-Key": appid.trim(),
            "X-Api-Access-Key": token.trim(),
            "X-Api-Resource-Id": targetResource,
            "X-Api-Request-Id": task_id,
            "X-Api-Sequence": "-1",
            "Content-Type": "application/json"
        };

        const submitRequest = {
            "user": {
                "uid": "roleplay_chat_user"
            },
            "audio": {
                "data": audioData.toString('base64'),
                "format": "wav"
            },
            "request": {
                "model_name": "bigmodel",
                "enable_channel_split": true, 
                "enable_ddc": true, 
                "enable_speaker_info": true, 
                "enable_punc": true, 
                "enable_itn": true
            }
        };

        const submitResponse = await axios.post(submit_url, submitRequest, { 
            headers: headers,
            timeout: 15000
        });

        const statusCode = submitResponse.headers['x-api-status-code'];
        if (statusCode !== "20000000") {
            throw new Error(`Submit task failed with status: ${statusCode}, message: ${submitResponse.headers['x-api-message']}`);
        }

        const x_tt_logid = submitResponse.headers['x-tt-logid'];
        console.log('Task submitted successfully. Task ID:', task_id, 'LogID:', x_tt_logid);

        // 2. 轮询结果
        const query_url = "https://openspeech-direct.zijieapi.com/api/v3/auc/bigmodel/query";
        let finished = false;
        let resultText = '';
        let attempts = 0;
        const maxAttempts = 30; 

        while (!finished && attempts < maxAttempts) {
            attempts++;
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            console.log(`Polling ASR result (attempt ${attempts})...`);
            
            const queryHeaders = {
                "X-Api-App-Key": appid.trim(),
                "X-Api-Access-Key": token.trim(),
                "X-Api-Resource-Id": targetResource,
                "X-Api-Request-Id": task_id,
                "X-Tt-Logid": x_tt_logid
            };

            const queryResponse = await axios.post(query_url, {}, { headers: queryHeaders });
            const code = queryResponse.headers['x-api-status-code'];

            if (code === '20000000') {  // task finished
                finished = true;
                const respData = queryResponse.data;
                if (respData && respData.result) {
                    resultText = respData.result.text || '';
                    if (!resultText && respData.result.utterances) {
                        resultText = respData.result.utterances.map(u => u.text).join('');
                    }
                }
                console.log('ASR Success:', resultText);
            } else if (code === '20000001' || code === '20000002') {
                // Still processing
                continue; 
            } else {
                throw new Error(`ASR Task failed with code: ${code}, message: ${queryResponse.headers['x-api-message']}`);
            }
        }

        if (fs.existsSync(audioFile.path)) fs.unlinkSync(audioFile.path);

        if (finished) {
            res.json({ success: true, text: resultText });
        } else {
            res.status(500).json({ success: false, message: '语音识别超时' });
        }

    } catch (error) {
        const errorMsg = error.response ? JSON.stringify(error.response.headers) : error.message;
        console.error('ASR Error:', errorMsg);
        
        if (audioFile && fs.existsSync(audioFile.path)) fs.unlinkSync(audioFile.path);
        
        res.status(500).json({ 
            success: false, 
            message: '语音识别请求失败', 
            detail: errorMsg 
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
