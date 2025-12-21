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
const session = require('express-session');
const path = require('path');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');

const app = express();
const upload = multer({ dest: 'uploads/' });
const PORT = 3000;

// 配置访问密码 (您可以根据需要修改)
const ACCESS_PASSWORD = '888';

app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));
app.use(session({
    secret: 'roleplay-chat-secret',
    resave: false,
    saveUninitialized: true,
    cookie: { maxAge: 24 * 60 * 60 * 1000 } // 1天有效期
}));

// 登录接口
app.post('/api/login', (req, res) => {
    const { password } = req.body;
    if (password === ACCESS_PASSWORD) {
        req.session.isLoggedIn = true;
        res.json({ success: true });
    } else {
        res.status(401).json({ success: false, message: '密码错误' });
    }
});

// 检查登录状态的中间件
const checkAuth = (req, res, next) => {
    if (req.session.isLoggedIn) {
        next();
    } else {
        res.status(401).send('Unauthorized');
    }
};

// 服务主页面
app.get('/', (req, res) => {
    // 无论是否登录都先给 index.html，由前端根据状态决定显示登录页还是聊天页
    res.sendFile(path.join(__dirname, 'index.html'));
});

// 检查认证状态接口
app.get('/api/check-auth', (req, res) => {
    res.json({ isLoggedIn: !!req.session.isLoggedIn });
});

// 火山引擎语音识别接口 (ASR)
app.post('/api/asr', upload.single('audio'), async (req, res) => {
    if (!req.session.isLoggedIn) return res.status(401).send('Unauthorized');
    
    const { appid, token } = req.body;
    const audioFile = req.file;

    if (!audioFile || !appid || !token) {
        return res.status(400).json({ success: false, message: '缺少参数' });
    }

    try {
        const audioData = fs.readFileSync(audioFile.path);
        
        // 调用火山引擎 ASR 一句话识别接口 (RESTful)
        // 文档参考: https://www.volcengine.com/docs/6561/80818
        const response = await axios.post(
            `https://openspeech.bytedance.com/api/v1/asr`,
            audioData,
            {
                params: {
                    appid: appid,
                    cluster: 'volc_auc_common', // 常用模型集群
                    workflow: 'audio_common',
                    format: 'wav',
                    resource_id: 'volc.asr.speech_recognition'
                },
                headers: {
                    'Authorization': `Bearer;${token}`,
                    'Content-Type': 'application/octet-stream'
                }
            }
        );

        // 清理临时文件
        fs.unlinkSync(audioFile.path);

        if (response.data && response.data.result) {
            // 返回识别结果中的第一条文本
            res.json({ success: true, text: response.data.result[0].text });
        } else {
            res.status(500).json({ success: false, message: '识别未返回结果', detail: response.data });
        }
    } catch (error) {
        console.error('ASR Error:', error.response ? error.response.data : error.message);
        if (audioFile) fs.unlinkSync(audioFile.path);
        res.status(500).json({ success: false, message: '语音识别请求失败' });
    }
});

app.listen(PORT, () => {
    console.log(`服务器运行在 http://10.30.129.21:${PORT}`);
    console.log(`默认访问密码为: ${ACCESS_PASSWORD}`);
});
