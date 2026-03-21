/**
 * MiMo-V2-TTS 语音合成技能 (Skill)
 *
 * 供大模型 Function Calling 使用的 TTS 工具定义与执行逻辑。
 * 基于小米 MiMo 开放平台 mimo-v2-tts 模型。
 *
 * 文档: https://platform.xiaomimimo.com/#/docs/usage-guide/speech-synthesis
 */

const { OpenAI } = require('openai');
const fs = require('fs');
const path = require('path');

// ═══════════════════════════════════════════════════════════════
//  1. Tool 定义 (OpenAI Function Calling Schema)
// ═══════════════════════════════════════════════════════════════

const TTS_TOOL = {
    type: 'function',
    function: {
        name: 'text_to_speech',
        description: [
            '将文本转换为自然流畅的语音音频。',
            '当用户要求"朗读"、"读出来"、"语音合成"、"TTS"、"转语音"、"说出来"时使用此工具。',
            '支持多种发音风格：情绪（开心/悲伤/生气）、方言（东北话/四川话/粤语）、角色扮演（孙悟空/林黛玉）、唱歌等。',
            '风格通过 style 参数指定，会自动添加 <style> 标签。',
            '也支持在文本中直接使用音频标签进行细粒度控制，如（紧张，深呼吸）、（小声）等。',
        ].join(''),
        parameters: {
            type: 'object',
            properties: {
                text: {
                    type: 'string',
                    description: [
                        '要合成为语音的文本内容。',
                        '可在文本中使用音频标签做细粒度控制，例如：',
                        '"（紧张，深呼吸）呼……冷静，冷静。"、',
                        '"（小声）嘘，别出声。"、',
                        '"如果我当时……（沉默片刻）结果是不是就不一样了？（苦笑）呵。"',
                    ].join('')
                },
                voice: {
                    type: 'string',
                    description: '预置音色。mimo_default=MiMo默认音色，default_zh=中文女声，default_en=英文女声。',
                    enum: ['mimo_default', 'default_zh', 'default_en'],
                    default: 'mimo_default'
                },
                style: {
                    type: 'string',
                    description: [
                        '语音整体风格，可组合多个风格（空格分隔）。',
                        '推荐风格——语速: 变快/变慢；情绪: 开心/悲伤/生气；',
                        '角色扮演: 孙悟空/林黛玉；风格: 悄悄话/夹子音/台湾腔；',
                        '方言: 东北话/四川话/河南话/粤语；特殊: 唱歌。',
                        '也支持使用不在列表中的风格。不需要风格时可不传。',
                    ].join('')
                },
                stream: {
                    type: 'boolean',
                    description: '是否使用流式合成。流式延迟更低，适合实时播放；非流式适合一次性获取完整音频文件。默认 true。',
                    default: true
                },
                context: {
                    type: 'string',
                    description: '上下文信息（填入 user 角色消息），用于辅助调整合成语气与风格。可选参数。例如用户之前说的话，帮助模型理解应该用什么语气回应。'
                },
                output_filename: {
                    type: 'string',
                    description: '输出文件名（不含路径），默认自动生成。例如 "greeting.wav"。',
                }
            },
            required: ['text']
        }
    }
};


// ═══════════════════════════════════════════════════════════════
//  2. Tool 执行逻辑
// ═══════════════════════════════════════════════════════════════

/**
 * 构造标准 WAV 文件头
 */
function buildWavHeader(pcmSize, sampleRate = 24000, channels = 1, bitsPerSample = 16) {
    const byteRate = sampleRate * channels * bitsPerSample / 8;
    const blockAlign = channels * bitsPerSample / 8;
    const header = Buffer.alloc(44);

    header.write('RIFF', 0);                              // ChunkID
    header.writeUInt32LE(36 + pcmSize, 4);                // ChunkSize
    header.write('WAVE', 8);                              // Format
    header.write('fmt ', 12);                             // Subchunk1ID
    header.writeUInt32LE(16, 16);                         // Subchunk1Size (PCM)
    header.writeUInt16LE(1, 20);                          // AudioFormat (PCM=1)
    header.writeUInt16LE(channels, 22);                   // NumChannels
    header.writeUInt32LE(sampleRate, 24);                 // SampleRate
    header.writeUInt32LE(byteRate, 28);                   // ByteRate
    header.writeUInt16LE(blockAlign, 32);                 // BlockAlign
    header.writeUInt16LE(bitsPerSample, 34);              // BitsPerSample
    header.write('data', 36);                             // Subchunk2ID
    header.writeUInt32LE(pcmSize, 40);                    // Subchunk2Size

    return header;
}

/**
 * 执行 TTS 工具调用
 *
 * @param {Object}  args           - 从 function calling 解析出的参数
 * @param {string}  args.text      - 要合成的文本
 * @param {string}  [args.voice]   - 预置音色
 * @param {string}  [args.style]   - 语音风格
 * @param {boolean} [args.stream]  - 是否流式
 * @param {string}  [args.context] - 上下文 (user 消息)
 * @param {string}  [args.output_filename] - 输出文件名
 * @param {Object}  config         - 运行时配置
 * @param {string}  config.apiKey  - MiMo API Key
 * @param {string}  [config.baseUrl]    - API 端点
 * @param {string}  [config.outputDir]  - 输出目录
 * @returns {Object} 工具执行结果
 */
async function executeTextToSpeech(args, config = {}) {
    const {
        text,
        voice = 'mimo_default',
        style = '',
        stream = true,
        context = '',
        output_filename = '',
    } = args;

    const apiKey = config.apiKey || process.env.MIMO_API_KEY;
    const baseUrl = config.baseUrl || 'https://api.xiaomimimo.com/v1';
    const outputDir = config.outputDir || path.join(process.cwd(), 'tmp', 'tts_output');

    // ── 参数校验 ──
    if (!apiKey) {
        return {
            success: false,
            error: '未配置 MiMo API Key。请设置 MIMO_API_KEY 环境变量或在配置中提供 apiKey。',
            help: '获取地址: https://platform.xiaomimimo.com/#/console/api-keys'
        };
    }

    if (!text || text.trim().length === 0) {
        return {
            success: false,
            error: '文本内容为空，请提供要合成的文本。'
        };
    }

    // ── 构造合成文本（添加 style 标签） ──
    let synthesisText = text;
    if (style) {
        synthesisText = `<style>${style}</style>${text}`;
    }

    // ── 构造 messages ──
    // 注意: 语音合成的目标文本必须在 assistant 角色中，user 角色用于提供上下文
    const messages = [];
    if (context) {
        messages.push({ role: 'user', content: context });
    }
    messages.push({ role: 'assistant', content: synthesisText });

    // ── 确定音频格式 ──
    // 流式必须用 pcm16，非流式用 wav
    const audioFormat = stream ? 'pcm16' : 'wav';

    const client = new OpenAI({ apiKey, baseURL: baseUrl });

    try {
        // 确保输出目录存在
        fs.mkdirSync(outputDir, { recursive: true });

        // 生成输出文件名
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const filename = output_filename || `tts_${timestamp}.wav`;
        const outputPath = path.join(outputDir, filename);

        if (stream) {
            // ════════════ 流式调用 ════════════
            const completion = await client.chat.completions.create({
                model: 'mimo-v2-tts',
                messages,
                audio: { format: 'pcm16', voice },
                stream: true,
            });

            // 收集 PCM16 裸数据 (24kHz mono 16bit)
            const pcmChunks = [];
            let audioChunkCount = 0;

            for await (const chunk of completion) {
                if (!chunk.choices || chunk.choices.length === 0) continue;

                const delta = chunk.choices[0].delta;
                const audio = delta?.audio;
                if (!audio) continue;

                // audio 是 dict, audio.data 是 base64 编码的 PCM16 裸数据
                const b64Data = typeof audio === 'object' ? audio.data : null;
                if (!b64Data) continue;

                const pcmBytes = Buffer.from(b64Data, 'base64');
                pcmChunks.push(pcmBytes);
                audioChunkCount++;
            }

            if (pcmChunks.length === 0) {
                return {
                    success: false,
                    error: '流式模式未收到音频数据。请检查 API Key 和文本内容。'
                };
            }

            // 拼接所有 PCM 数据，加 WAV 头写入文件
            const allPcm = Buffer.concat(pcmChunks);
            const wavHeader = buildWavHeader(allPcm.length, 24000, 1, 16);

            fs.writeFileSync(outputPath, Buffer.concat([wavHeader, allPcm]));

            const duration = allPcm.length / (24000 * 1 * 16 / 8);
            const fileSizeKB = (wavHeader.length + allPcm.length) / 1024;

            return {
                success: true,
                file_path: outputPath,
                file_name: filename,
                duration_seconds: Math.round(duration * 10) / 10,
                file_size_kb: Math.round(fileSizeKB * 10) / 10,
                audio_chunks: audioChunkCount,
                format: 'wav (pcm16 stream)',
                sample_rate: 24000,
                voice,
                style: style || '默认',
                message: `语音合成完成，时长 ${Math.round(duration * 10) / 10} 秒，已保存至 ${outputPath}`
            };

        } else {
            // ════════════ 非流式调用 ════════════
            const completion = await client.chat.completions.create({
                model: 'mimo-v2-tts',
                messages,
                audio: { format: 'wav', voice },
            });

            const message = completion.choices?.[0]?.message;
            let audioData = null;

            if (message?.audio) {
                audioData = typeof message.audio === 'object'
                    ? (message.audio.data || null)
                    : null;
            }

            if (!audioData) {
                return {
                    success: false,
                    error: '非流式模式未返回音频数据。请检查 API Key 和文本内容。'
                };
            }

            // 非流式返回完整 WAV，直接写入
            const wavBytes = Buffer.from(audioData, 'base64');
            fs.writeFileSync(outputPath, wavBytes);

            const fileSizeKB = wavBytes.length / 1024;

            return {
                success: true,
                file_path: outputPath,
                file_name: filename,
                file_size_kb: Math.round(fileSizeKB * 10) / 10,
                format: 'wav (non-stream)',
                voice,
                style: style || '默认',
                message: `语音合成完成，已保存至 ${outputPath}`
            };
        }

    } catch (error) {
        return {
            success: false,
            error: `语音合成失败: ${error.message}`,
            details: error.response?.data || error.code || ''
        };
    }
}


// ═══════════════════════════════════════════════════════════════
//  3. 集成接口
// ═══════════════════════════════════════════════════════════════

/**
 * 处理来自 LLM tool_calls 的 TTS 调用
 * 可直接嵌入 server.js 的 tool_calls 处理循环中使用
 *
 * @example
 * // 在 server.js 中集成:
 * const { TTS_TOOL, handleTtsToolCall } = require('./tmp/tts_skill');
 *
 * // 添加到 tools 数组:
 * tools: [TEXT_TO_IMAGE_TOOL, MEMORY_TOOL, TTS_TOOL]
 *
 * // 在 tool_calls 处理循环中:
 * if (toolCall.function.name === 'text_to_speech') {
 *     const result = await handleTtsToolCall(toolCall, { apiKey: MIMO_API_KEY });
 *     toolResults.push(result);
 * }
 */
async function handleTtsToolCall(toolCall, config = {}) {
    try {
        const args = JSON.parse(toolCall.function.arguments);
        const result = await executeTextToSpeech(args, config);

        return {
            tool_call_id: toolCall.id,
            role: 'tool',
            name: 'text_to_speech',
            content: JSON.stringify(result)
        };
    } catch (error) {
        return {
            tool_call_id: toolCall.id,
            role: 'tool',
            name: 'text_to_speech',
            content: JSON.stringify({
                success: false,
                error: `工具调用解析失败: ${error.message}`
            })
        };
    }
}


// ═══════════════════════════════════════════════════════════════
//  4. MCP Tool 定义 (可选，用于 mcp-server.js)
// ═══════════════════════════════════════════════════════════════

const TTS_MCP_TOOL = {
    name: 'text_to_speech',
    description: TTS_TOOL.function.description,
    inputSchema: TTS_TOOL.function.parameters,
};


// ═══════════════════════════════════════════════════════════════
//  导出
// ═══════════════════════════════════════════════════════════════

module.exports = {
    TTS_TOOL,               // OpenAI Function Calling 格式的工具定义
    TTS_MCP_TOOL,           // MCP 格式的工具定义
    executeTextToSpeech,     // 核心执行函数（可独立使用）
    handleTtsToolCall,       // tool_calls 处理器（集成到 server.js）
    buildWavHeader,          // WAV 工具函数（可复用）
};
