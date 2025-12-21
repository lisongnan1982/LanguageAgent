/*
 * @Author: sunnysnli sunnysnli@tencent.com
 * @Date: 2025-12-21 16:02:16
 * @LastEditors: sunnysnli sunnysnli@tencent.com
 * @LastEditTime: 2025-12-21 16:21:26
 * @FilePath: \LanguageAgent\server.js
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
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

// LLM 代理接口 - 解决前端直接调用 OpenRouter 的 CORS 和身份验证问题
app.post('/api/proxy-llm', async (req, res) => {
    const { apiKey, model, messages, response_format } = req.body;
    if (!apiKey || !model || !messages) {
        return res.status(400).json({ success: false, message: '缺少必要参数' });
    }

    try {
        const response = await axios.post('https://openrouter.ai/api/v1/chat/completions', {
            model,
            messages,
            response_format
        }, {
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://github.com/Roo-Code/Roo-Code', // 随便填一个，OpenRouter 喜欢有 referer
                'X-Title': 'RolePlay Chat'
            }
        });

        res.json(response.data);
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

app.listen(PORT, () => {
    console.log(`服务器运行在 http://0.0.0.0:${PORT}`);
});
