# Qwen3-TTS Base x-vector-only 前导静音:修复设计

复现数据见同目录 README.md。本文件写修法、参数、评测协议与备选方案,用于外审与后续验收。

## 机制判断

x-vector-only 请求只有说话人向量,没有参考 codec 前缀,talker 冷启动。复现数据显示起声时间是
双峰分布:要么第 0 帧开口,要么进入 4–7 帧的开场静音,1–3 帧很少。静音段电平约 -60 dB。
由此推断:第 0 帧的 codebook-0 采样落到"静音区"token 时,模型会把静音延续数帧;落到语音
token 时就直接说下去。波动来自 seed(采样),与参考音频基本无关。

## 修法

对 Base x-vector-only 请求,在前 N 个 codec 帧的 codebook-0 采样中,把静音 token 集合 S 的
logit 置为 -inf。等价于在前 N 帧从 p(c0 | c0 ∉ S) 采样,即"以语音开头"为条件的采样;第 N 帧
之后分布不变。

- **S 的定义**:这个 checkpoint 自己的 codec encoder 对听不见的输入给出的 codebook-0 id。引擎
  启动时用 encoder 编码数字静音和 -80、-60 dBFS RMS 的白噪声各 1 秒,收集 codebook-0 id 的并集。
  -60 dBFS 取自复现数据里静音段的实测电平(即 Common Voice 录音底噪)。S 随 checkpoint 推导,
  不写死。
- **帧计数**:用请求已生成的 codec 帧数(output_codes 长度),第 0 帧就是 prefill 那一步采样。
  retraction 重新 prefill 时 output_codes 保留,不会重复屏蔽。
- **作用范围**:只对 x-vector-only 请求。ICL 的起声跟随参考音频末尾停顿(1.7B:参考末尾
  135/170/300 ms 对应起声中位数 105/342/755 ms),是有依据的韵律模仿,不屏蔽。CustomVoice
  已有 #1901 在输出端处理确定出现的 1 帧静音,VoiceDesign 不在本次范围。
- **配置**:引擎级 tts_engine factory 参数 leading_silence_mask_frames(N),N=0 即关闭。
  不加请求级开关。

## 参数 N 的确定

扫 N ∈ {0, 1, 2, 3, 4, 6},两个 checkpoint(1.7B、0.6B Base),全部 x-vector-only:

- 起声:复现用的网格(3 参考 × 12 句 × 10 seed,-34 dBFS 峰值口径),报告中位数、p90、
  超过 1 帧与 2 帧的比例;
- 质量:seed-tts-eval EN(CI 同款 benchmark_tts_seedtts.py,--no-ref-text,固定 seed),
  报告 corpus WER(CI 同款 ASR)与说话人相似度(WavLM),必要时加 UTMOS;
- 默认值取起声曲线变平的最小 N,前提是 WER 与相似度在 N=0 的噪声范围内;若质量有可见
  退化,默认关闭。

我的预判:双峰分布提示 N=1 就能拿到大部分收益;N 过大可能把正常的句首弱起音(清辅音、
气声)也挤掉,需要看 WER 与主观听感。

## 备选方案与不选的理由

1. **输出端裁剪 PCM**(不改采样分布):Base 首帧内容随请求变化(T-PR16 已测),只能按能量
   阈值逐帧判断,阈值本身是启发式,且有削掉轻声起音的风险;4–7 帧静音仍要生成,只省播放
   时间不省计算。
2. **在 prompt 里加一小段参考 codec("mini-ICL")**:改变了条件分布,等于变相 ICL,会带入参考
   末尾停顿的问题,且增加 prefill。
3. **软惩罚(logit 减一个常数)**:多一个无量纲超参,解释不如"条件于语音开头"干净。
4. **调温度/top-k**:影响整段生成,不针对问题。

## 需要外审回答的问题

1. "前 N 帧条件于非静音"作为方法是否站得住?有没有更原则化的做法?
2. S 的推导(codec encoder 对 ≤ -60 dBFS 输入的输出)是否合理?电平上界该怎么论证?
3. 评测协议(起声 + WER + 相似度,两个 checkpoint,N 的 sweep)是否足够?漏了什么风险
   (起音被截断的听感、韵律、长句、非英语)?
4. 默认开启(按"性能优化正确性允许即默认开")是否合适,还是应默认关闭?
