# -*- coding: utf-8 -*-
"""Generate the SGLang-Omni daily PR study report (HTML + PDF)."""
from pygments import highlight
from pygments.lexers import PythonLexer, DiffLexer, BashLexer
from pygments.formatters import HtmlFormatter
import html as _html

DATE = "2026-05-31"

def code(src, lang="python"):
    lexer = {"python": PythonLexer(), "diff": DiffLexer(), "bash": BashLexer()}[lang]
    return highlight(src.strip("\n"), lexer, HtmlFormatter(nowrap=False, cssclass="hl"))

def kp(title, body):
    """Knowledge-point box (三段式)."""
    return f'<div class="kp"><div class="kp-t">🧠 {title}</div>{body}</div>'

# ---------------------------------------------------------------------------
# PR cards. Each is (meta_dict, html_body)
# ---------------------------------------------------------------------------
CARDS = []

# ============================ PR #539 =====================================
CARDS.append(dict(
  num=539, theme="多模态 / 视觉编码器",
  title="Ming vision encoder：用 F.linear 替换 patch_embed 的 Conv3d",
  author="edwingao28", merged="2026-05-31 01:47 UTC", tag="Bugfix / Perf",
  url="https://github.com/sgl-project/sglang-omni/pull/539",
  onesent="把视觉编码器里一个“名义上是 3D 卷积、其实等价于一次矩阵乘”的算子换成真正的矩阵乘，单次调用从 ~3.7 秒降到几乎为 0，整个图像编码阶段从 ~8s/请求 降到 ~0.02s/请求。",
  body=f"""
<h4>2. 它解决了什么问题</h4>
<p>Ming-Omni 的视觉编码器里有一层叫 <code>patch_embed</code>（图像分块嵌入层），它内部包了一个
<code>nn.Conv3d</code>（3D 卷积）。问题在于：这个卷积的 <b>卷积核大小（kernel_size）正好等于输入的空间尺寸</b>，
所以每个 patch 卷积之后只会输出 <b>1×1×1 一个格子</b>——这是一种“退化”的卷积。</p>
<p>cuDNN（NVIDIA 的卷积加速库）对这种退化形状<b>没有高效算法</b>，会回退到一条非常慢的路径：在 H100、bf16 下，
<b>单次调用就要约 3.7 秒</b>，而且即使关掉 <code>cudnn.benchmark</code>、即使形状完全相同重复调用，也还是慢。
对于典型的 MMMU（多模态理解评测）输入，光这一个算子就让编码阶段达到 ~8s/请求，成为整条链路的瓶颈。</p>
<p>关键洞察：当卷积核 = 输入尺寸、输出只有 1 个格子时，这个卷积在数学上<b>完全等价于一次线性投影</b>
（把 <code>C×Tp×Pp×Pp</code> 个数展平后乘一个权重矩阵）。所以可以直接用 <code>F.linear</code>（底层走 cuBLAS 的 GEMM 矩阵乘），
这才是这种形状“本该用”的算法。vLLM 在 <code>Conv3dLayer</code> 里也做了同样的替换。</p>

<h4>3. 具体做了什么改动</h4>
<p>核心文件：<code>sglang_omni/models/ming_omni/components/vision_encoder.py</code>。新增一个辅助函数把卷积“翻译”成矩阵乘：</p>
{code('''
def _linear_patch_embed(patch_embed, pixel_values):
    # 把 Qwen3VLVisionPatchEmbed 的 Conv3d 投影用等价的 Linear 来跑
    patch_dim = (                          # 一个 patch 展平后的长度
        patch_embed.in_channels            # 通道数 C（如 RGB=3）
        * patch_embed.temporal_patch_size  # 时间维 patch 大小 Tp
        * patch_embed.patch_size           # 空间高 Pp
        * patch_embed.patch_size           # 空间宽 Pp
    )
    return F.linear(
        pixel_values.view(-1, patch_dim),               # [N_patch, patch_dim]
        patch_embed.proj.weight.view(patch_embed.embed_dim, -1),  # 把卷积核权重摊平成矩阵
        patch_embed.proj.bias,                          # 偏置照搬
    )
''')}
<p>逐行解释：</p>
<ul>
<li><code>patch_dim</code>：把“通道 × 时间 patch × 高 × 宽”乘起来，得到<b>每个图像块展平后的向量长度</b>。
卷积权重的形状是 <code>[embed_dim, C, Tp, Pp, Pp]</code>，元素总数等于 <code>embed_dim × patch_dim</code>。</li>
<li><code>pixel_values.view(-1, patch_dim)</code>：把输入摊平成 <code>[patch 数量, patch_dim]</code> 的二维矩阵——这正是矩阵乘需要的形状。</li>
<li><code>proj.weight.view(embed_dim, -1)</code>：把 5 维卷积核摊平成 <code>[embed_dim, patch_dim]</code> 的二维权重矩阵。
卷积核里的数字一个没动，只是换了“摆放方式”，所以结果数学上完全相同。</li>
<li><code>F.linear(x, W, b)</code> 计算 <code>x @ W.T + b</code>，走 cuBLAS 的 GEMM——GPU 上最成熟、最快的算子之一。</li>
</ul>
<p>然后 forward 里只把一行调用换掉：</p>
{code('''
- x = self.patch_embed(x)              # 旧：走 Conv3d，落到慢的 cuDNN 路径
+ # Qwen3VLVisionPatchEmbed 包的 Conv3d 每个 patch 只输出一格，
+ # 等价于 Linear，可避开慢的 cuDNN 路径。
+ x = _linear_patch_embed(self.patch_embed, x)   # 新：走 cuBLAS GEMM
''', 'diff')}
<p>同时新增一个单元测试，断言“同样的权重下，Conv3d 和 F.linear 的输出在 bf16 精度内一致”
（相对误差 ≤ 2×0.78%），用合成权重就能测，不需要加载真实 Ming 权重。</p>
<p><b>效果</b>（MMMU 100 样本，H100）：准确率 0.60 基本不变；平均时延 15.6s → 7.7s；编码阶段墙钟时间 ~8s → ~0.02s。</p>

<h4>4. 涉及的知识点</h4>
{kp("ViT / patch embedding（图像分块嵌入）",
   "<b>它是什么</b>：Vision Transformer 不像 CNN 那样滑窗扫描，而是把图片切成一个个固定大小的小块（patch，比如 14×14 像素），"
   "每个 patch 拉直成一个向量，再用一个线性层投影成 Transformer 能吃的「token 向量」。"
   "<b>为什么需要它</b>：Transformer 处理的是「一串 token」，图片不是天然的 token 序列，patch embedding 就是把图片<b>翻译成 token 序列</b>的第一道工序。"
   "<b>本 PR 里</b>：这道工序原本用 Conv3d 实现（卷积核=patch 大小 时，卷积扫一个 patch 恰好等于「展平后做一次线性投影」），本 PR 把它换成显式的 Linear。")}
{kp("Conv3d vs Linear，cuDNN vs cuBLAS / GEMM",
   "<b>它是什么</b>：GEMM=通用矩阵乘（General Matrix Multiply），是线性层的底层运算，由 <b>cuBLAS</b> 库提供，GPU 上极致优化。"
   "卷积由 <b>cuDNN</b> 库提供，它针对「真正的滑窗卷积」做了优化。"
   "<b>为什么需要区分</b>：当卷积退化成「核=输入、输出 1 格」时，它在数学上就是一次矩阵乘，但 cuDNN 不认识这种特例，"
   "会挑一个通用但极慢的算法（这里慢到 3.7 秒/次）。"
   "<b>本 PR 里</b>：识别出这个等价关系，直接调 <code>F.linear</code> 让它走 cuBLAS GEMM——「用对的工具做对的事」。这是推理优化里非常典型的一招：算子等价替换（operator substitution）。")}
{kp("cudnn.benchmark 为什么救不了",
   "<b>它是什么</b>：cuDNN 有个自动调优开关，第一次遇到某形状时会试跑多种算法挑最快的并缓存。"
   "<b>为什么这里没用</b>：对这个退化形状，cuDNN 的「候选算法集合」里压根没有快的，再怎么自动挑也只能在一堆慢算法里选；而且本场景形状重复出现，benchmark 的缓存优势也体现不出来。"
   "<b>启示</b>：性能问题要定位到「算子+形状+后端算法」这一层，单纯调框架开关常常无效。")}

<h4>5. 我能学到什么</h4>
<ul>
<li><b>等价替换思想</b>：当一个算子在特定形状下退化，往往存在数学等价但工程上快得多的实现。先理解算子的数学本质，再选后端。</li>
<li><b>用单测锁住等价性</b>：替换实现时，写一个「新旧输出数值一致」的测试，是保证「优化不改变正确性」的标准做法。</li>
<li><b>延伸</b>：可了解 ViT/Qwen-VL 的视觉编码器结构、cuDNN 算法选择机制、以及 vLLM 的 <code>Conv3dLayer</code> 同款优化。</li>
</ul>
"""))

# ============================ PR #617 =====================================
CARDS.append(dict(
  num=617, theme="Omni / 多阶段语音架构",
  title="Qwen3-Omni talker 默认开启 partial-start，并加 --talker-partial-start 开关",
  author="Hayden727", merged="2026-05-30 18:56 UTC", tag="Perf",
  url="https://github.com/sgl-project/sglang-omni/pull/617",
  onesent="让“说话器”（talker）不必等“思考器”（thinker）把整段文字想完，攒够 5 个文本块就开始生成语音，从而和 thinker 在两张 GPU 上并行，首音延迟（TTFC）最多降 34.5%、吞吐最多升 20.6%；并把这个行为设为默认、同时保留一个运行期开关。",
  body=f"""
<h4>2. 它解决了什么问题</h4>
<p>Qwen3-Omni 的语音生成是<b>多阶段、多 GPU 流水线</b>：thinker（在 GPU0，负责把输入想成文字）→ talker_ar（在 GPU1，把文字转成语音的离散编码）→ code2wav（在 GPU1，把编码还原成波形）。</p>
<p>原来的问题：talker <b>必须等 thinker 把整段文字全部生成完</b>才开始干活。在 H200、SeedTTS 流式场景下，profiling 显示 talker 的「输入→第一个语音编码」延迟主导了首音时间（TTFC，c=8 时约 749ms），因为它在 thinker 跑那 ~400ms 期间一直<b>空闲等待</b>。GPU1 干等着 GPU0，浪费了并行的机会。</p>
<p>partial-start（部分启动）就是让 talker 攒够 <code>partial_start_min_chunks</code>（默认 5）个文本块就先开工，而不是等全文。这样 talker（GPU1）和还在跑的 thinker（GPU0）<b>重叠执行</b>，时延和吞吐双双改善。这个能力在更早的 #475 就实现了，但当时默认关闭，并留了句「等 audio_duration 稳定了再翻默认值」。本 PR 用一组配对实验证明了 partial-start 对音频时长的影响（13.78%）和「同配置重复跑」的天然噪声（12.88%）<b>没有统计差异</b>，于是放心地把默认翻成开。</p>

<h4>3. 具体做了什么改动</h4>
<p><b>(a) 配置默认值翻转</b>，<code>models/qwen3_omni/config.py</code>：把 talker 阶段的 <code>enable_partial_start</code> 由参数注入，
并让「分离式（disaggregated，thinker/talker 各占一张卡）」拓扑默认 <code>True</code>，「共置（colocated，同一张卡）」拓扑默认 <code>False</code>。</p>
{code('''
-def _talker_stage(*, gpu, process):
+def _talker_stage(*, gpu, process, enable_partial_start):
     return StageConfig(
         name="talker_ar",
         ...
-            "enable_partial_start": False,
+            "enable_partial_start": enable_partial_start,
             "partial_start_min_chunks": 5,
''', 'diff')}
<p>为什么共置默认关？因为共置时 thinker 和 talker 抢同一张 GPU，并不存在「GPU1 空等 GPU0」的并行红利，partial-start 反而可能增加调度复杂度，所以保守地默认关。</p>
<p><b>(b) 新增运行期 CLI 开关</b> <code>--talker-partial-start {{default,on,off}}</code>，<code>cli/serve.py</code>。它做了一件很讲究的事：在真正改配置前先<b>校验阶段类型</b>，只有 Qwen3-Omni 的 talker 工厂才允许这个开关，否则 fail fast：</p>
{code('''
def apply_partial_start_cli_overrides(pipeline_config, *, talker_partial_start):
    mode = _normalize_stage_toggle_mode("talker_partial_start", talker_partial_start)
    if mode == "default":
        return pipeline_config        # 用户没指定，保留配置默认值
    stage_name = _resolve_talker_stage(pipeline_config, flag_name="--talker-partial-start")
    matching_stages = _find_matching_stages(pipeline_config, stage_name=stage_name, ...)
    for stage in matching_stages:
        if stage.factory != _QWEN_PARTIAL_START_TALKER_FACTORY:
            raise typer.BadParameter(   # 非 Qwen talker 直接报错，且此时还没改任何东西
                "--talker-partial-start currently supports only Qwen3-Omni talker; ...")
    _apply_stage_factory_args_override(
        pipeline_config, stage_name=stage_name,
        updates={"enable_partial_start": mode == "on"}, ...)
    return pipeline_config
''')}
<p>逐块解释：<br>
• <code>mode == "default"</code>：三态开关里 <code>default</code> 表示「不覆盖」，直接返回，尊重配置文件的默认值——这让 CLI 和配置不打架。<br>
• 先 <code>_find_matching_stages</code> 再循环检查 <code>stage.factory</code>：确保<b>所有匹配阶段都被覆盖</b>且类型正确，<b>校验通过后才动手改</b>。这叫「先校验、后变更」（validate-then-mutate），避免改了一半发现报错、留下半残配置。<br>
• 错误用 <code>typer.BadParameter</code>：CLI 层标准的「用户输入错误」异常，能给出友好提示而不是堆栈。</p>
<p><b>(c) 示例启动脚本</b>把 <code>--enable-partial-start</code> 改成 <code>BooleanOptionalAction</code>（默认 on，可用 <code>--no-enable-partial-start</code> 关），并对共置拓扑自动默认 off。</p>
<p><b>(d) 大量单测</b>覆盖 on/off/default/非法值，以及「默认值翻转」「共置默认关」「非 Qwen 配置报稳定错误信息」等。</p>

<h4>4. 涉及的知识点</h4>
{kp("Thinker–Talker–Code2wav 三段式语音架构",
   "<b>它是什么</b>：Qwen3-Omni 把「听懂/想内容 → 决定怎么说 → 合成声音」拆成三个模块。thinker 是个大语言模型，产出文字（及隐藏状态）；"
   "talker 是个自回归模型（AR），把文字转成一串<b>离散语音编码（codec codes）</b>；code2wav 是声码器，把编码还原成可播放的波形。"
   "<b>为什么需要它</b>：拆成多段可以分别放到不同 GPU、分别优化、分别批处理，像工厂流水线一样并行起来吞吐更高。"
   "<b>本 PR 里</b>：正是利用「thinker 在 GPU0、talker 在 GPU1」这个分离拓扑，让 talker 提前开工与 thinker 并行。")}
{kp("Partial-start（部分启动）/ 流水线重叠",
   "<b>它是什么</b>：下游阶段不等上游完全产出，攒够一小段就开始处理。"
   "<b>类比</b>：自助餐厅你不必等所有菜都端上来才开吃，前几样上来就能动筷子——你「吃」和厨房「继续做」是重叠的。"
   "<b>为什么需要它</b>：减少下游空等，把串行的「等→做」变成并行的「边等边做」，首响应更快、设备利用率更高。"
   "<b>本 PR 里</b>：talker 攒够 5 个 thinker 文本块即开始生成语音，TTFC 最多降 34.5%、吞吐升 20.6%。")}
{kp("TTFC / RTF / ITL 这些语音指标",
   "<b>TTFC</b>（Time To First Chunk，首块/首音延迟）：从收到请求到吐出第一段音频的时间，决定「用户多久听到声音」。"
   "<b>RTF</b>（Real-Time Factor，实时率）：生成 1 秒音频需要多少秒计算，<b>RTF&lt;1 才算比实时快</b>（能边生成边播不卡）。"
   "<b>ITL</b>（Inter-Token/Chunk Latency，块间延迟）：相邻两段音频之间的间隔，决定播放是否流畅。"
   "<b>本 PR 里</b>：TTFC、RTF 都明显下降，ITL 基本持平（符合预期，因为 partial-start 改善的是「开头的等待」而非「中段的吞吐节奏」）。")}
{kp("三态开关（default / on / off）与「校验后变更」",
   "<b>它是什么</b>：CLI 开关不用简单的布尔，而用三态：on/off 显式覆盖、default 表示「不管，听配置的」。"
   "<b>为什么需要它</b>：布尔开关无法区分「用户要关」和「用户没说」。三态让「配置默认值」和「命令行临时覆盖」清晰分层。"
   "<b>本 PR 里</b>：default 直接返回不改；同时遵循「先把所有阶段校验通过、再统一改写」，避免改一半失败留下半残状态。")}
{kp("用配对实验区分「真效应」与「噪声」",
   "<b>它是什么</b>：talker 在温度 0.7 下采样，同一请求每次跑出的音频时长本就会抖动。要判断 partial-start 是否真改变了时长，得先量「不开它、重复跑」的天然抖动（噪声地板 12.88%），再比「开 vs 不开」的差异（13.78%）。"
   "<b>为什么重要</b>：两者几乎相等 → 说明 partial-start 带来的差异淹没在采样噪声里，不是它造成的，于是默认开是安全的。"
   "<b>启示</b>：性能/质量结论要有「对照的噪声基线」，否则容易把随机波动误读成因果。")}

<h4>5. 我能学到什么</h4>
<ul>
<li><b>并行红利来自「打破串行依赖」</b>：找到下游空等上游的窗口，用流式/部分启动把它填满。</li>
<li><b>默认值的翻转要有数据背书</b>：本 PR 不是拍脑袋开默认，而是用配对实验证明无副作用，体现了「以数据驱动决策」的工程文化。</li>
<li><b>good CLI 设计</b>：三态开关、fail-fast 校验、稳定的错误信息、与配置默认值不冲突——都是可复用的接口设计经验。</li>
<li><b>延伸</b>：去了解流水线并行（pipeline parallelism）、disaggregated serving（PD 分离）、以及语音流式合成的评测指标体系。</li>
</ul>
"""))

# ============================ PR #614 =====================================
CARDS.append(dict(
  num=614, theme="Omni / 调度器 / 流式",
  title="新增流式 TTS 调度器（StreamingSimpleScheduler）",
  author="JingwenGu0829", merged="2026-05-30 08:28 UTC", tag="Feature / Perf",
  url="https://github.com/sgl-project/sglang-omni/pull/614",
  onesent="抽象出一个可复用的「流式简单调度器」基类，让声码器这类需要「分块流式输入」的阶段，既能边收边算（流式），又保留原来的整批一次算（非流式）路径；Higgs/Fish-S2-Pro/Qwen-Code2Wav 三个 TTS 都接上后，流式模式下 RTF 大幅下降、吞吐翻倍。",
  body=f"""
<h4>2. 它解决了什么问题</h4>
<p>原来的 <code>SimpleScheduler</code>（简单调度器）只会做「一个请求来了，整个算完再返回」的<b>非流式</b>处理。但声码器（vocoder，把语音编码变波形的最后一段）天然适合<b>流式</b>：上游 talker 是一个 chunk 一个 chunk 地吐编码的，声码器完全可以收到一块就合成一块音频先发出去，让用户更早听到声音。</p>
<p>痛点：要给每个 TTS 模型（Higgs、Fish-Audio S2-Pro、Qwen3-Omni Code2Wav）单独写一套「接收分块 / 处理分块 / 收尾」的流式逻辑，会大量重复且容易出错（中止清理、内存泄漏、批处理与流式混在一起）。本 PR 把这套「流式请求的生命周期管理」抽成一个共享基类。</p>

<h4>3. 具体做了什么改动</h4>
<p>核心新文件 <code>sglang_omni/scheduling/streaming_simple_scheduler.py</code>，定义 <code>StreamingSimpleScheduler</code>。它保留 SimpleScheduler 的 <b>inbox/outbox（收件箱/发件箱队列）契约</b>，额外支持四种消息生命周期：</p>
{code('''
class StreamingSimpleScheduler:
    # 带流式输入的简单阶段调度器基类。
    # 子类实现流式钩子；非流式请求仍走 compute_fn / batch_compute_fn，与 SimpleScheduler 一致。

    # —— 留给子类实现的「钩子」（模板方法模式）——
    def is_streaming_payload(self, payload):      ...  # 判断这个请求是不是流式
    def on_streaming_new_request(self, rid, p):   ...  # 流式请求初始化
    def on_stream_chunk(self, rid, item):  return []   # 来了一个 chunk 怎么处理→产出消息
    def on_stream_done(self, rid):         return []   # 上游说「发完了」怎么收尾
    def clear_stream_state(self, rid):            ...  # 清理该请求的状态
''')}
<p>调度主循环根据消息类型分发——这是它的“心脏”：</p>
{code('''
def _handle_message(self, msg, loop):
    if msg.type == "new_request":   # 新请求：流式做初始化，非流式直接算（可批）
        self._handle_new_request_batch(self._collect_new_request_batch(msg), loop)
    elif msg.type == "stream_chunk":     # 来了一块流式数据
        self._on_chunk(msg.request_id, msg.data)
    elif msg.type == "stream_done":      # 上游声明该请求数据发完
        self._on_done(msg.request_id)
    else:
        raise ValueError(f"Unsupported streaming scheduler message type: {msg.type}")
''')}
<p>它还认真处理了两件分布式系统里最容易出 bug 的事：</p>
<ul>
<li><b>中止（abort）与清理</b>：用一个有上限的「已中止请求 id 集合」记录被取消的请求，主循环每次取到消息先查它是否已中止，是就直接丢弃，避免给已经放弃的请求白算。集合超过上限（10000）会裁剪到 5000，<b>防止内存无限增长</b>。</li>
<li><code>stream_done</code> <b>可能先于最后一个数据块到达</b>（注释里专门点出 “may arrive before the terminal payload”），所以用 <code>_pending_done</code> 集合把「完成信号」暂存，等数据真到齐再收尾。这是流式系统里典型的「乱序事件」处理。</li>
</ul>
<p>非流式请求依然可以攒批（<code>max_batch_size</code> / <code>max_batch_cost</code>），流式请求则被排除在批处理之外单独走流式路径——<b>两条路径在同一个调度器里共存、互不干扰</b>。</p>
<p>然后把三个声码器接上：Higgs 新增 <code>HiggsStreamingVocoderScheduler</code>、Fish S2-Pro 的 <code>streaming_vocoder.py</code>、Qwen 的 <code>code2wav_scheduler.py</code>，并统一用「紧凑波形负载（compact waveform payload）」让流式响应体积更小。</p>
<p><b>效果</b>（节选）：Higgs Audio V3 TTS，非流式 RTF 0.76 / 吞吐 4.62 req/s → 流式 RTF 0.30 / 吞吐 11.17 req/s，且首音仅 0.87s；WER（字错率）不升反略降。</p>

<h4>4. 涉及的知识点</h4>
{kp("Scheduler（调度器）在 SGLang 里是什么",
   "<b>它是什么</b>：调度器是每个「阶段（stage）」的大脑，决定「收到的请求/数据块按什么顺序、攒多大批、何时送进模型计算」。"
   "它通过 inbox（收件箱队列）拿输入、outbox（发件箱队列）发输出，与上下游解耦。"
   "<b>为什么需要它</b>：推理服务要同时服务很多请求，调度器负责把它们高效地组织起来（批处理、流式、优先级、中止）。"
   "<b>本 PR 里</b>：新增的流式调度器基类，专门管「流式分块输入」的请求生命周期，同时兼容老的非流式批处理。")}
{kp("流式 vs 非流式 / 声码器（vocoder）",
   "<b>声码器</b>：语音合成的最后一段，把模型产出的「离散语音编码」还原成耳朵能听的波形（waveform）。"
   "<b>非流式</b>：整段编码到齐后一次性合成全部音频——简单，但用户要等全部算完才听到声音。"
   "<b>流式</b>：收到一块编码就合成一小段音频先发出去——首音快、体验好，但要管好「分块、收尾、乱序、中止」。"
   "<b>本 PR 里</b>：让声码器从只能非流式，升级为「流式/非流式双模」。")}
{kp("模板方法模式（Template Method）/ 钩子方法",
   "<b>它是什么</b>：基类把「主流程骨架」写死（这里是消息分发主循环、中止清理、批处理），把「会变化的步骤」留成空方法（钩子，如 on_stream_chunk）让子类填。"
   "<b>类比</b>：报销流程是固定的（提交→审核→打款），但「审核怎么审」各部门自己定。"
   "<b>为什么需要它</b>：三个声码器共享同一套复杂的生命周期/中止/内存管理，只各自实现「一块数据来了怎么合成」，避免重复造轮子。"
   "<b>本 PR 里</b>：StreamingSimpleScheduler 是模板，三个 vocoder 调度器是子类。")}
{kp("乱序事件 & 幂等清理（流式系统的两个坑）",
   "<b>乱序</b>：在并发/异步系统里，「数据发完」的信号可能比最后一块数据先到。代码用 _pending_done 暂存完成信号，等数据齐了再真正收尾。"
   "<b>有界清理</b>：被中止的请求 id 要记下来好丢弃后续数据，但不能无限记——所以设上限并定期裁剪，防止内存泄漏。"
   "<b>启示</b>：写流式/异步代码，永远要问「事件会不会乱序？状态会不会无限增长？请求中途取消了怎么办？」")}

<h4>5. 我能学到什么</h4>
<ul>
<li><b>抽象共性、隔离差异</b>：当多个模块有相同的「复杂骨架 + 少量差异」，提一个模板基类是教科书级的复用手法。</li>
<li><b>双模共存</b>：新功能（流式）不破坏旧路径（非流式批处理），降低迁移风险。</li>
<li><b>流式的工程素养</b>：中止清理、乱序处理、内存有界，是做实时推理服务的硬功夫。</li>
<li><b>延伸</b>：可学习生产者-消费者队列、asyncio 事件循环、以及 TTS 声码器（如 HiFi-GAN、DAC）的原理。</li>
</ul>
"""))

# ============================ PR #605 =====================================
CARDS.append(dict(
  num=605, theme="Omni / TTS / 缓存优化",
  title="优化 Higgs TTS 的参考音频编码缓存",
  author="BBuf", merged="2026-05-30 10:54 UTC", tag="Perf",
  url="https://github.com/sgl-project/sglang-omni/pull/605",
  onesent="给 Higgs TTS 加一个可选的「参考音频缓存」（环境变量开启）：相同的参考音频不再每次都重新加载、重采样和编码，靠稳定且能自动失效的缓存键避免重复劳动；默认行为不变。",
  body=f"""
<h4>2. 它解决了什么问题</h4>
<p>Higgs 是「声音克隆」式 TTS：你给一段<b>参考音频（reference audio）</b>，它学这段声音的音色，再用这个音色念目标文本。处理参考音频要做三件重活：① 加载并重采样到 24kHz；② 用 codec 编码成「参考语音编码」；③ 套上 delay pattern。</p>
<p>痛点：实际服务里，很多请求会反复用<b>同一段参考音频</b>（比如某个固定发音人）。原来每个请求都把这三步从头跑一遍，纯属重复劳动，浪费 GPU 和时间。本 PR 加缓存：同一段参考音频只算一次，后续直接复用。难点不在「缓存」本身，而在于<b>缓存键（cache key）要既稳定又能正确失效</b>——同一文件命中缓存，文件被替换了就不能再用旧结果。</p>

<h4>3. 具体做了什么改动</h4>
<p>核心文件 <code>models/higgs_tts/stages.py</code>、<code>payload_types.py</code>。整个设计由环境变量 <code>SGLANG_OMNI_HIGGS_REF_CODE_CACHE=1</code> 控制，<b>不开就完全是老行为</b>（安全的灰度策略）。</p>
<p><b>(a) 为不同输入形态生成稳定缓存键</b>：</p>
{code('''
def _reference_audio_cache_key(reference_audio):
    # 为一个参考音频输入生成稳定的缓存键
    if isinstance(reference_audio, (str, Path)):
        return _reference_path_cache_key(reference_audio)        # 本地路径
    ...
    if "bytes" in reference_audio:                               # 原始字节
        data = reference_audio["bytes"]
        return hash_media_item(data)
    encoded = reference_audio.get("base64") or reference_audio.get("data")  # base64/data URL
    raw = base64.b64decode(encoded) if isinstance(encoded, str) else bytes(encoded)
    return hash_media_item(raw)
''')}
<p>对路径、原始字节、base64 三种输入分别算键；注意 <code>media_type</code>（如 audio/wav vs audio/mpeg）只是解码提示、不影响内容，所以<b>不进键</b>（单测专门验证「同内容不同 media_type 必须同键」）。</p>
<p><b>(b) 本地文件键：既要不重复读、又要能感知文件被替换</b>。这是最精妙的部分：</p>
{code('''
def _reference_path_cache_key(path_like):
    # 用 (路径, 大小, mtime_ns, ctime_ns) 当 memo key：稳定文件不必反复整文件读哈希，
    # 但普通替换/同大小快速替换都能让键失效。
    path = Path(str(path_like)).expanduser()
    memo = _reference_path_hash_memo_key(path)         # = (路径:size:mtime:ctime, size)
    if memo is None:  return None                      # 不是文件（如 URL）就不缓存
    memo_key, file_size = memo
    sentinel = _reference_path_sentinel(path, file_size)   # 读 头/中/尾 各 8KB 当「指纹」
    digest = _get_reference_path_hash(memo_key, sentinel)  # 命中 memo 直接拿全文哈希
    if digest is not None:
        return f"file:{digest}"
    digest = hash_bytes(path.read_bytes())             # 没命中才真正读全文算哈希（慢，但只一次）
    if _reference_path_hash_memo_key(path) == memo:
        _put_reference_path_hash(memo_key, sentinel, digest)   # 存起来，下次免读
    return f"file:{digest}"
''')}
<p>逐层理解这套「<b>三级防失效</b>」设计：<br>
1）<b>memo key</b> = 路径 + 文件大小 + 修改时间(mtime) + 创建/变更时间(ctime)。文件没动时，这串完全一样，可以跳过「读全文算哈希」。<br>
2）<b>sentinel（哨兵指纹）</b>：读文件的头、中、尾各 8KB 拼起来再哈希。用来对付「文件大小、mtime 都没变，但内容被改了」这种刁钻情况——只要头/中/尾任一处变了，sentinel 就变，旧缓存作废。单测里专门构造「同大小、同头尾、只改中段」来验证不会误命中。<br>
3）只有前两级都判定「可能是同一文件」时，才信任缓存的<b>全文哈希</b>当最终键；否则老老实实读全文。这样既快（稳定文件零额外读）又稳（改了一定失效）。<br>
memo 用一个带锁的有界 <code>OrderedDict</code>（LRU），超出 1024 条就淘汰最旧的。</p>
<p><b>(c) 两级内容缓存</b>：<br>
• 预处理阶段缓存「加载+重采样后的波形」（<code>StageOutputCache</code>，按条数 256、字节 512MB 双上限），多线程跑所以加锁；取出时 <code>.clone()</code> 一份，<b>避免多个请求共享同一张可变 tensor</b>。<br>
• 音频编码阶段缓存「加了 delay pattern 的参考编码」，存成 CPU 的 int32 tensor（不是 list，<b>方便按字节计大小</b>来做容量上限）；这一阶段是单线程的 SimpleScheduler，所以不用锁。</p>
{code('''
cached_delayed = reference_code_cache.get(state.reference_cache_key)
if cached_delayed is not None:
    delayed_rows = cached_delayed.tolist()          # 命中：直接用，跳过 codec.encode_reference
else:
    ref_codes_TN = codec.encode_reference(waveform, sample_rate=24000).to(torch.long)  # 未命中才编码
    ...
    delayed = apply_delay_pattern(ref_codes_TN)
    delayed_rows = delayed.tolist()
    reference_code_cache.put(state.reference_cache_key, delayed.to("cpu", torch.int32))
''')}

<h4>4. 涉及的知识点</h4>
{kp("缓存键（cache key）的「稳定」与「失效」",
   "<b>它是什么</b>：缓存的本质是「输入相同→直接返回上次结果」。键就是用来判断「输入是否相同」的指纹。"
   "<b>两难</b>：键太松→文件改了还命中旧结果（脏数据）；键太严→每次都重算（缓存没用）。"
   "<b>本 PR 里</b>：用「stat 元信息 memo（快速放行稳定文件）+ 头中尾哨兵（抓同大小篡改）+ 全文哈希（最终裁决）」三级，兼顾速度与正确性。"
   "<b>类比</b>：判断两份文件是否相同，先看名字+大小+日期（快），可疑再抽查几页（哨兵），仍可疑才逐字比对（全文哈希）。")}
{kp("mtime / ctime / 文件 stat 元信息",
   "<b>它是什么</b>：操作系统给每个文件记录大小、最后修改时间 mtime、状态变更时间 ctime 等元数据，读取很快（不用打开文件内容）。"
   "<b>为什么用它</b>：绝大多数「同一文件重复请求」场景下，这些元信息没变，就能<b>零成本</b>确认「还是那个文件」，跳过昂贵的全文哈希。"
   "<b>本 PR 里</b>：作为缓存键的第一级快速通道。")}
{kp("LRU 缓存 / 有界容量（按条数 + 按字节）",
   "<b>它是什么</b>：LRU=最近最少使用淘汰。缓存不能无限长，满了就踢掉最久没用的。"
   "<b>为什么按字节也限</b>：音频波形/编码是大对象，只按「条数」限可能内存爆掉，所以同时设「最大字节数」双保险。"
   "<b>本 PR 里</b>：path-hash memo 限 1024 条；波形缓存 256 条/512MB；编码缓存 256 条/256MB。把编码存成 int32 tensor 就是为了能精确按字节计量。")}
{kp("可变对象的别名 bug 与 .clone()",
   "<b>它是什么</b>：tensor 是可变的、按引用传递。如果缓存直接把同一张 tensor 发给多个请求，一个请求改了它，别人也跟着被改——隐蔽的 bug。"
   "<b>本 PR 里</b>：存入和取出都 <code>.clone()</code> 复制一份；单测甚至断言两个请求拿到的波形 <code>data_ptr()</code>（底层内存地址）不同，确保是各自独立的副本。"
   "<b>启示</b>：缓存可变对象时，务必想清楚「共享还是复制」。")}
{kp("灰度开关（feature flag）",
   "<b>它是什么</b>：新功能用环境变量/开关控制，默认关，确认无碍再放开。"
   "<b>本 PR 里</b>：<code>SGLANG_OMNI_HIGGS_REF_CODE_CACHE=1</code> 才启用，默认行为完全不变——上线风险可控。")}

<h4>5. 我能学到什么</h4>
<ul>
<li><b>缓存的难点是「键」不是「存」</b>：稳定命中 + 正确失效，是缓存正确性的核心。这套三级键设计值得收藏。</li>
<li><b>分层防御</b>：快通道（stat）→ 抽检（哨兵）→ 终判（全文哈希），每级都更慢更准，按需升级，是性能与正确性的经典折中。</li>
<li><b>缓存可变对象要复制</b>，容量要双上限（条数+字节）。</li>
<li><b>延伸</b>：了解 functools.lru_cache、内容寻址存储（content-addressable storage）、以及 TTS 的声音克隆/参考编码机制。</li>
</ul>
"""))

# ============================ PR #612 =====================================
CARDS.append(dict(
  num=612, theme="Omni / 音频编码器 / 编译优化",
  title="用 torch.compile 编译 Higgs 的 DAC 音频编码器",
  author="yxs", merged="2026-05-30 05:25 UTC", tag="Perf",
  url="https://github.com/sgl-project/sglang-omni/pull/612",
  onesent="对 Higgs 的 DAC 声学编码器调用 torch.compile（动态形状），并在加载时做一次 warm-up 触发编译，换来约 -11.8% 的 RTF（更快）和 +8.5% 的 QPS，代价是音色相似度仅降 0.55%、WER 尾部略升。",
  body=f"""
<h4>2. 它解决了什么问题</h4>
<p>Higgs 处理参考音频时要用一个 <b>DAC 声学编码器（acoustic encoder）</b>把波形编码成离散 token。它默认是 PyTorch 的 <b>eager（即时执行）</b>模式——每个算子单独发射到 GPU，调度开销大、无法跨算子融合，速度有提升空间。</p>
<p><code>torch.compile</code> 能把这段网络<b>整体编译</b>成优化过的内核（算子融合、减少 Python/调度开销），从而提速。本 PR 就是给这个编码器套上 <code>torch.compile</code>，并解决「编译要在第一次真正跑时才发生、会拖慢第一个请求」的问题——办法是加载时先用一段静音 warm-up 一下，把编译成本提前付掉。</p>

<h4>3. 具体做了什么改动</h4>
<p>改动极小但很典型，<code>models/higgs_tts/stages.py</code> 的 <code>create_audio_encoder_executor</code>：</p>
{code('''
codec = get_or_load_codec(checkpoint_dir, device, dtype)
+codec.model.acoustic_encoder = torch.compile(
+    codec.model.acoustic_encoder, mode="default", dynamic=True
+)
+codec.encode_reference(                       # 加载期 warm-up：用 1 秒静音先跑一次
+    torch.zeros(codec.SAMPLE_RATE), sample_rate=codec.SAMPLE_RATE
+)
''', 'diff')}
<p>逐行解释：<br>
• <code>torch.compile(module, mode="default", dynamic=True)</code>：把 <code>acoustic_encoder</code> 包成「会被即时编译（JIT）」的版本。<code>dynamic=True</code> 表示<b>允许输入形状变化</b>——参考音频长度各不相同，如果按固定形状编译，每来一个新长度都要重新编译，反而更慢；动态形状让一份编译产物适配多种长度。<br>
• <code>mode="default"</code>：平衡编译时间与运行速度的默认策略（相对 <code>max-autotune</code> 编译更快、调优更少）。<br>
• 紧接着用 <code>torch.zeros(SAMPLE_RATE)</code>（1 秒静音）调一次 <code>encode_reference</code>：这一步是<b>触发编译</b>。torch.compile 是「懒编译」——不真正跑就不编译。提前用假数据跑一次，编译开销就落在<b>服务启动时</b>而非<b>第一个真实用户请求</b>上。</p>
<p>性能（n=6 对照，整套 1088 条 SeedTTS-EN，c=16，单卡 H200）：RTF 0.462→0.407（<b>-11.8%</b>，更快）；QPS 8.11→8.80（<b>+8.5%</b>）；音色相似度 66.42→66.06（-0.55%）；WER 中位基本不变，p95 尾部 +0.6pp。<b>这是一次明确的「速度 vs 质量」权衡</b>，PR 把代价量化得很清楚，方便决策。</p>

<h4>4. 涉及的知识点</h4>
{kp("torch.compile / eager 模式 / JIT 编译",
   "<b>eager（即时执行）</b>：PyTorch 默认模式，写一行算一行，灵活好调试，但每个算子单独发射到 GPU，开销大。"
   "<b>torch.compile</b>：把一段网络<b>整体捕获成计算图</b>再编译成融合内核，减少 Python 与 kernel 启动开销、能算子融合，通常更快。"
   "<b>类比</b>：eager 像口译（说一句翻一句，灵活但慢），compile 像先把整段稿子拿去专业排版印刷（前期要排版，之后印得又快又好）。"
   "<b>本 PR 里</b>：只编译最吃算力的 acoustic_encoder，换来 ~12% 提速。")}
{kp("动态形状（dynamic shapes）与重编译",
   "<b>它是什么</b>：torch.compile 默认会针对「具体输入形状」编译。形状一变（这里是音频长度），就触发<b>重新编译</b>，很贵。"
   "<b>dynamic=True</b>：告诉编译器「形状会变，请生成一份能适配多种长度的通用产物」，避免反复重编译。"
   "<b>本 PR 里</b>：参考音频长度天然多变，所以必须开 dynamic，否则缓存命中率低、可能比 eager 还慢。")}
{kp("懒编译 与 warm-up（预热）",
   "<b>它是什么</b>：torch.compile 是「第一次真正执行时才编译」（lazy/JIT）。编译本身可能耗时几秒，会拖慢撞上它的那个请求。"
   "<b>warm-up</b>：服务启动时先用假数据（这里 1 秒静音）跑一遍，把编译成本提前在启动阶段付掉，真实用户的第一个请求就已经是编译好的快版本。"
   "<b>本 PR 里</b>：编译后立刻 <code>encode_reference(torch.zeros(...))</code> 就是预热。这是部署 compile/CUDA graph 类优化的标准配套动作。")}
{kp("性能与质量的量化权衡",
   "<b>它是什么</b>：很多加速手段会轻微改变数值（融合、近似），可能影响模型质量。负责任的做法是<b>同时报速度收益和质量代价</b>。"
   "<b>本 PR 里</b>：-11.8% RTF 的收益，对应 -0.55% 音色相似度、+0.6pp 的 WER 尾部——明牌摆出，让维护者判断是否值得。"
   "<b>启示</b>：优化 PR 不能只晒提速，必须给出质量回归数据。")}

<h4>5. 我能学到什么</h4>
<ul>
<li><b>只编译热点</b>：不必整模型 compile，挑最吃算力的子模块（acoustic_encoder）性价比最高。</li>
<li><b>compile 必配 warm-up</b>，且对变长输入要开 dynamic，否则可能适得其反。</li>
<li><b>优化要带质量账单</b>：速度和质量一起报，是工程诚信也是决策依据。</li>
<li><b>延伸</b>：去了解 torch.compile 的 TorchDynamo/Inductor、CUDA graph、以及 DAC/RVQ 这类神经音频编解码器。</li>
</ul>
"""))

# ============================ PR #602 =====================================
CARDS.append(dict(
  num=602, theme="Omni / 多阶段 / 健壮性",
  title="修复 Ming-Omni talker 的音色预设加载，并加生成时长护栏",
  author="edwingao28", merged="2026-05-30 07:56 UTC", tag="Bugfix",
  url="https://github.com/sgl-project/sglang-omni/pull/602",
  onesent="原来音色清单文件缺失/损坏时服务会「带病运行」、生成出跑偏的声音；本 PR 改成加载时就 fail-fast 报清楚的错，请求时按优先级解析发音人，并用一个「按文本长度估算时长」的护栏给 CFM 解码步数封顶，避免无谓地长跑。",
  body=f"""
<h4>2. 它解决了什么问题</h4>
<p>Ming-Omni 的 talker 用<b>音色预设（voice preset）</b>来决定「用谁的嗓子说话」：一个 <code>voice_name.json</code> 清单把音色名（如默认的 "DB30"）映射到一段<b>提示音频（prompt wav）</b>。CFM 声码器需要这段 prompt wav 作为<b>说话人锚点（speaker anchor）</b>，否则声音会逐渐跑偏。</p>
<p>痛点：原来如果 <code>voice_name.json</code> 没挂载上，代码只打一行 warning 然后<b>继续服务</b>，<code>voice_json_dict</code> 是空的。于是默认音色 "DB30" 永远解析不到 prompt wav，CFM 在 <code>prompt_wav_lat=None</code>（没有说话人锚点）下运行，输出几秒内就<b>漂移</b>得不像 DB30。这是典型的「<b>静默失败（silent failure）</b>」——服务看起来活着，结果却是错的，比直接崩溃更难排查。</p>
<p>第二个问题：CFM 解码默认最多跑 1000 步，但短文本根本不需要那么多步，白白浪费算力、还可能拖出多余的尾音。</p>

<h4>3. 具体做了什么改动</h4>
<p><b>(a) 加载时 fail-fast 校验</b>，<code>talker_executor.py</code> / <code>streaming_talker.py</code> 各加一个 <code>_validate_voice_presets</code>：</p>
{code('''
def _validate_voice_presets(self, voice_dict, manifest_path, talker_dir):
    # 解析相对的 prompt-wav 路径并校验清单；就地把每条改成绝对路径
    if self._voice is not None and self._voice not in voice_dict:
        raise ValueError(f"default voice {self._voice!r} not found in {manifest_path}; "
                         f"available presets: {sorted(voice_dict.keys())}")   # 默认音色不在清单→报错
    for name, entry in voice_dict.items():
        rel_path = entry.get("prompt_wav_path")
        if rel_path is None:
            raise ValueError(f"voice preset {name!r} ... is missing prompt_wav_path")  # 缺字段→ValueError(而非裸 KeyError)
        resolved = os.path.join(talker_dir, rel_path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"voice preset {name!r} references missing prompt wav {resolved}")  # wav 不存在→报错
        entry["prompt_wav_path"] = resolved      # 相对路径→绝对路径
''')}
<p>并把加载逻辑改成：清单缺失但<b>配了默认音色</b>→直接 <code>FileNotFoundError</code>（不再「警告后带病运行」）；没配默认音色才降级为 info 日志「预设关闭」。四种失败（清单缺失 / 默认音色不在清单 / 条目缺 prompt_wav_path / wav 文件不存在）都在加载期、用清晰的异常类型暴露，而不是等到推理时炸一个看不懂的 <code>KeyError</code>。</p>
<p><b>(b) 请求时按优先级解析发音人</b>，并<b>把解析移出 CUDA 流上下文</b>，让坏请求快速失败、不占 CUDA 资源：</p>
{code('''
# 在抢 CUDA stream 之前解析，坏请求 fail-fast 且不占 CUDA 资源
if prompt_wav_path is not None:
    pass                                   # 1) 显式 prompt_wav_path 优先（单次请求覆盖预设）
elif voice_name is not None and voice_name in self.voice_json_dict:
    prompt_text = self.voice_json_dict[voice_name]["prompt_text"]   # 2) 用预设里的发音人
    prompt_wav_path = self.voice_json_dict[voice_name]["prompt_wav_path"]
elif voice_name is not None:               # 3) 给了 voice_name 但查无此人→明确报错并解释后果
    raise ValueError(f"voice_name={voice_name!r} not found ... the talker would run "
                     f"without a speaker anchor and produce drifting voice output. ...")
else:
    raise ValueError("omni_audio_generation requires either voice_name ... or prompt_wav_path. Both are None.")
''')}
<p>优先级：①请求显式给的 prompt_wav_path（单次覆盖）＞ ②voice_name 查预设 ＞ ③都没有就<b>明确报错</b>（而不是悄悄用空锚点）。错误信息还顺带解释「为什么不能放行：会没有说话人锚点、声音会漂移」，对运维极友好。</p>
<p><b>(c) 时长护栏</b>：新增 <code>duration_capped_steps</code>，用「每字约 0.36 秒、最低 2 秒」的启发式估算这段文本最多需要多长音频，换算成 CFM 解码步数上限，给 <code>generate()</code> 封顶：</p>
{code('''
def duration_capped_steps(self, text_len, audio_detokenizer, requested_max_steps):
    # 按音频时长启发式给 CFM 解码步数封顶：约 0.36s/字，下限 2.0s
    ...
    seconds_per_step = (self.patch_size * vae_patch_size * hop_size) / sample_rate
    max_duration_s = max(2.0, text_len * (5818.0 / 16000.0))      # 文本越长允许越久，但有 2s 下限
    max_steps_by_duration = max(1, int(max_duration_s / seconds_per_step))
    return min(requested_max_steps, max_steps_by_duration)        # 取「请求上限」与「时长上限」的更小者
''')}
<p>同时修正 <code>generate()</code> 的收尾逻辑：无论是「命中停止 token」还是「撞到步数上限」退出，<b>都要 yield 一个 <code>last_chunk=True</code></b>，这样下游流式 VAE 才会把最后的尾音 flush 出来——否则按步数上限退出时会丢掉最后一块音频。单测专门覆盖了这两条退出路径。</p>

<h4>4. 涉及的知识点</h4>
{kp("Fail-fast vs 静默失败（silent failure）",
   "<b>静默失败</b>：出错了却不报，继续以错误状态运行（这里：音色清单缺失仍开服，输出跑偏的声音）。它最坑，因为问题被掩盖、排查极难。"
   "<b>Fail-fast</b>：一发现前置条件不满足就<b>立刻、清楚地</b>报错。"
   "<b>为什么更好</b>：在「加载期」就炸，比在「半夜服务跑偏」才发现要好得多。"
   "<b>本 PR 里</b>：把「warning 后带病运行」改成加载期抛 FileNotFoundError/ValueError，并在请求期对查无此人的发音人明确拒绝。")}
{kp("说话人锚点 / prompt wav / CFM 声码器",
   "<b>它是什么</b>：要让 TTS 用某个特定嗓音，得给一段该嗓音的<b>提示音频（prompt wav）</b>作为「锚点」，模型据此条件化生成。CFM（Conditional Flow Matching，条件流匹配）是一类用于语音合成的生成模型，需要这个锚点来稳定音色。"
   "<b>为什么缺它会漂移</b>：没有锚点，模型没有「该像谁」的约束，生成几秒后音色就自由漂移了。"
   "<b>本 PR 里</b>：正是因为锚点（prompt_wav_lat）变成 None 才导致 DB30 走样，所以坚决不允许无锚点放行。")}
{kp("异常类型要表意（ValueError vs 裸 KeyError）",
   "<b>它是什么</b>：用 <code>dict[key]</code> 直接取缺失的键会抛裸 <code>KeyError</code>，信息含糊、出错位置也不对。"
   "<b>更好做法</b>：在正确的层用正确的异常类型（缺字段→ValueError、文件不存在→FileNotFoundError）并带可读消息。"
   "<b>本 PR 里</b>：<code>_validate_voice_presets</code> 显式检查并抛带上下文的异常，让「清单写错了」在第一时间被人看懂。")}
{kp("启发式护栏（heuristic guard）/ 解码步数封顶",
   "<b>它是什么</b>：用一个简单经验公式（这里：每字约 0.36s、下限 2s）估算合理上限，给可能失控的循环封顶。"
   "<b>为什么需要</b>：短文本不该跑满 1000 步——既浪费算力又可能拖出多余尾音。封顶把「最坏情况」约束住。"
   "<b>本 PR 里</b>：把文本长度换算成最大音频时长，再换算成 CFM 步数上限，与请求上限取更小者。")}
{kp("资源获取前先校验（不要拿着锁/CUDA 资源失败）",
   "<b>它是什么</b>：把「可能失败的参数解析」放在<b>申请昂贵资源（CUDA stream）之前</b>，坏请求就不会先占资源再报错。"
   "<b>本 PR 里</b>：发音人解析被「hoist（上提）」到 <code>torch.cuda.stream(...)</code> 之外。这和 #617 的「先校验后变更」是同一种纪律。")}

<h4>5. 我能学到什么</h4>
<ul>
<li><b>宁可崩溃，不要带病运行</b>：把静默失败改成 fail-fast，是健壮服务的基本功。</li>
<li><b>错误信息要解释后果与出路</b>：本 PR 的报错不仅说「找不到」，还说「会导致音色漂移、请这样修」，极大降低排查成本。</li>
<li><b>循环要有护栏</b>，资源申请前先校验，收尾路径（flush 尾块）每条都要测到。</li>
<li><b>延伸</b>：了解 CFM/flow-matching TTS、声音克隆中的 speaker conditioning、以及「输入校验应放在哪一层」的设计原则。</li>
</ul>
"""))

# ---------------------------------------------------------------------------
# Minor / CI / Docs cluster (briefly)
# ---------------------------------------------------------------------------
MINORS = [
 (621,"[CI] 新增 Whisper ASR parity CI","zhaochenyang20","2026-05-30 22:57",
  "给语音识别（ASR）加一条 CI：用 OpenAI Whisper 作为参照，校验本仓库 ASR 输出与其「对齐/一致（parity）」，防止改动悄悄拉低识别质量。"),
 (623,"[CI] 把 Higgs 接入 CI","zhaochenyang20","2026-05-31 06:33",
  "把 Higgs TTS 纳入持续集成，让每次提交都自动跑 Higgs 的相关测试。"),
 (629,"[CI] Higgs 的 CI 阈值","zhaochenyang20","2026-05-31 07:53",
  "为上一条接入的 Higgs CI 标定/设置质量阈值（如 WER、相似度的可接受范围），超标即判失败。"),
 (630,"修复 Qwen3-Omni MMMU CI 的纯文本输出配置","Hayden727","2026-05-31 10:48",
  "修正 MMMU（多模态理解评测）在 CI 中「文本输出」相关的配置错误，让评测跑对。"),
 (633,"[CI] 放宽 TTS WER 的最大失败容忍","zhaochenyang20","2026-05-31 18:47",
  "调整 TTS 字错率（WER）CI 的最大允许失败数，降低因随机抖动导致的误报（flaky）。"),
 (634,"[CI] 把 ASR CI 移到 Stage 1","zhaochenyang20","2026-05-31 20:35",
  "把 ASR 的 CI 检查前移到流水线第一阶段，让快速、关键的检查更早跑、更早反馈。"),
 (632,"修复被追踪的 CI venv 符号链接","Ratish1","2026-05-31 18:37",
  "修复被 git 误追踪的 CI 虚拟环境 symlink，避免环境不一致。"),
 (622,"[Docs] 重写 README 并切换到 Apache-2.0 许可","xinlij","2026-05-30 22:56",
  "重写项目 README，并把许可证切换为 Apache-2.0。"),
 (631,"[Docs] 使用逐字的 Apache-2.0 许可文本","yxs","2026-05-31 17:56",
  "把许可证文件替换为标准、逐字的 Apache-2.0 全文。"),
 (616,"更新文档 logo 与仓库 URL","xinlij","2026-05-30 17:27",
  "更新文档里的 logo 图片与仓库链接。"),
]

# ---------------------------------------------------------------------------
# Render HTML
# ---------------------------------------------------------------------------
pyg_css = HtmlFormatter().get_style_defs('.hl')

theme_counts = {}
for c in CARDS:
    theme_counts[c["theme"]] = theme_counts.get(c["theme"],0)+1

def card_html(c):
    return f"""
<details class="card" open>
  <summary>
    <span class="badge">#{c['num']}</span>
    <span class="ctitle">{_html.escape(c['title'])}</span>
    <span class="tag">{c['tag']}</span>
  </summary>
  <div class="cbody">
    <div class="meta">
      <span>👤 {c['author']}</span>
      <span>✅ merged · {c['merged']}</span>
      <span>🏷️ {c['theme']}</span>
      <a href="{c['url']}" target="_blank">🔗 PR #{c['num']}</a>
    </div>
    <div class="onesent"><b>一句话：</b>{c['onesent']}</div>
    {c['body']}
  </div>
</details>
"""

minor_rows = "\n".join(
  f"<tr><td><a href='https://github.com/sgl-project/sglang-omni/pull/{n}' target='_blank'>#{n}</a></td>"
  f"<td>{_html.escape(t)}</td><td>{a}</td><td>{m}</td><td>{_html.escape(d)}</td></tr>"
  for (n,t,a,m,d) in MINORS)

overview_themes = "".join(f"<li><b>{t}</b> · {n} 个</li>" for t,n in theme_counts.items())

HTML_DOC = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SGLang-Omni 每日 PR 学习日报 · {DATE}</title>
<style>
:root {{ --bg:#0f1220; --card:#171a2b; --ink:#e8eaf2; --mut:#9aa0b5; --acc:#7c9cff; --acc2:#39d3a3; --line:#2a2f48; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font-family:"Noto Sans CJK SC","WenQuanYi Zen Hei","PingFang SC","Microsoft YaHei",system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  line-height:1.75; font-size:15.5px; }}
.wrap {{ max-width:980px; margin:0 auto; padding:28px 20px 80px; }}
header.hero {{ background:linear-gradient(135deg,#1b2140,#10233a 60%,#0f2e2a);
  border:1px solid var(--line); border-radius:18px; padding:28px 30px; margin-bottom:22px; }}
header.hero h1 {{ margin:0 0 6px; font-size:26px; }}
header.hero .sub {{ color:var(--mut); font-size:14px; }}
.stats {{ display:flex; flex-wrap:wrap; gap:14px; margin-top:18px; }}
.stat {{ background:#0e1426aa; border:1px solid var(--line); border-radius:12px; padding:12px 16px; min-width:120px; }}
.stat .n {{ font-size:24px; font-weight:700; color:var(--acc2); }}
.stat .l {{ font-size:12px; color:var(--mut); }}
.overview {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px 20px; margin-bottom:24px; }}
.overview h2 {{ margin:.2em 0 .4em; font-size:18px; color:var(--acc); }}
.overview ul {{ margin:.3em 0; padding-left:1.2em; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:14px; margin:16px 0; overflow:hidden; }}
.card > summary {{ cursor:pointer; list-style:none; padding:16px 20px; display:flex; align-items:center; gap:12px;
  background:#141831; border-bottom:1px solid transparent; }}
.card[open] > summary {{ border-bottom:1px solid var(--line); }}
.card > summary::-webkit-details-marker {{ display:none; }}
.badge {{ background:var(--acc); color:#0b1020; font-weight:700; border-radius:8px; padding:2px 9px; font-size:13px; }}
.ctitle {{ font-weight:700; font-size:16px; flex:1; }}
.tag {{ font-size:11.5px; color:var(--acc2); border:1px solid #2e5b4e; border-radius:20px; padding:2px 10px; white-space:nowrap; }}
.cbody {{ padding:8px 22px 22px; }}
.meta {{ display:flex; flex-wrap:wrap; gap:16px; color:var(--mut); font-size:13px; margin:10px 0 14px; }}
.meta a {{ color:var(--acc); text-decoration:none; }}
.onesent {{ background:#0e1730; border-left:3px solid var(--acc); padding:10px 14px; border-radius:8px; margin:6px 0 16px; }}
h4 {{ color:var(--acc); margin:22px 0 8px; font-size:15.5px; border-bottom:1px dashed var(--line); padding-bottom:5px; }}
code {{ background:#0c1024; border:1px solid var(--line); border-radius:5px; padding:1px 6px; font-size:13px;
  font-family:"DejaVu Sans Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; color:#ffd9a8; }}
.hl {{ background:#0b0e1d !important; border:1px solid var(--line); border-radius:10px; padding:14px 16px;
  overflow-x:auto; font-size:12.5px; line-height:1.55; margin:10px 0; }}
.hl pre {{ margin:0; }}
.kp {{ background:#101a2e; border:1px solid #284067; border-radius:10px; padding:12px 16px; margin:12px 0; }}
.kp-t {{ font-weight:700; color:#9fc0ff; margin-bottom:4px; }}
ul,ol {{ padding-left:1.4em; }}
table {{ width:100%; border-collapse:collapse; margin:12px 0; font-size:13px; }}
th,td {{ border:1px solid var(--line); padding:7px 10px; text-align:left; }}
th {{ background:#141831; color:var(--acc); }}
td a {{ color:var(--acc); text-decoration:none; }}
footer {{ color:var(--mut); font-size:12.5px; text-align:center; margin-top:36px; border-top:1px solid var(--line); padding-top:16px; }}
{pyg_css}
.hl .hll {{ background:#1e2540; }}
</style></head>
<body><div class="wrap">
<header class="hero">
  <h1>🛰️ SGLang-Omni 每日 PR 学习日报</h1>
  <div class="sub">日期：{DATE}（UTC） · 仓库：sgl-project/sglang-omni · 范围：过去约 24 小时内合并（merged）的 PR</div>
  <div class="stats">
    <div class="stat"><div class="n">{len(CARDS)+len(MINORS)}</div><div class="l">合并 PR 总数</div></div>
    <div class="stat"><div class="n">{len(CARDS)}</div><div class="l">深度讲解（Omni/多模态/多阶段）</div></div>
    <div class="stat"><div class="n">{len(MINORS)}</div><div class="l">CI / 文档类（简述）</div></div>
  </div>
</header>

<div class="overview">
  <h2>📌 当日概览</h2>
  <p>今天合并的 PR 大致分两类：一类是<b>围绕 Omni 多模态 / 多阶段语音架构的核心改动</b>（性能、健壮性、流式能力），
  另一类是<b>CI 与文档</b>的工程维护。本报告对前者逐一讲透，对后者列表简述。</p>
  <p><b>深度讲解的主题分类：</b></p>
  <ul>{overview_themes}</ul>
  <p><b>一条主线串起来看：</b>Qwen3-Omni 的语音链路是 <code>thinker → talker → code2wav</code> 三段式；
  Higgs / Fish-S2-Pro 是 TTS 声音克隆链路。今天的改动几乎都落在这两条多阶段流水线上——
  让视觉编码更快（#539）、让 talker 提前并行开工（#617）、让声码器支持流式（#614）、
  给参考音频加缓存（#605）、编译音频编码器（#612）、修音色加载的静默失败（#602）。
  它们共同体现了推理基础设施的几个永恒主题：<b>并行与重叠、缓存复用、算子/编译优化、以及 fail-fast 的健壮性</b>。</p>
</div>

<h2 style="color:var(--acc2);font-size:20px;border-bottom:2px solid var(--line);padding-bottom:6px;">🔬 深度讲解</h2>
{''.join(card_html(c) for c in CARDS)}

<h2 style="color:var(--acc2);font-size:20px;border-bottom:2px solid var(--line);padding-bottom:6px;margin-top:30px;">🧰 其余 PR（CI / 文档，简述）</h2>
<table>
<tr><th>PR</th><th>标题</th><th>作者</th><th>合并时间(UTC)</th><th>一句话</th></tr>
{minor_rows}
</table>

<div class="overview" style="margin-top:26px;">
  <h2>🎓 给小白的「该再去补哪些前置知识」清单</h2>
  <ul>
   <li><b>SGLang 架构组件</b>：scheduler（调度器）、tokenizer manager、detokenizer、model runner、attention backend、KV cache、CUDA graph、continuous batching、radix attention、tensor parallelism。</li>
   <li><b>多模态/Omni</b>：ViT 与 patch embedding、视觉/音频 encoder 如何对接 LLM、thinker–talker–code2wav 三段式、TTS 声音克隆与 speaker conditioning。</li>
   <li><b>性能工程</b>：算子等价替换、cuBLAS/cuDNN、torch.compile（Dynamo/Inductor）、动态形状、warm-up、流水线重叠与 PD 分离、缓存键设计与 LRU。</li>
   <li><b>健壮性/工程纪律</b>：fail-fast、validate-then-mutate、资源获取前校验、灰度开关（feature flag）、用单测锁住等价性与边界。</li>
  </ul>
</div>

<footer>
  本日报由自动化流程抓取 GitHub 合并 PR 并生成 · 数据截至 {DATE} · 仅供学习参考<br>
  讲解深度优先于覆盖广度：6 个 Omni 相关 PR 深讲，10 个 CI/文档 PR 简述。
</footer>
</div></body></html>"""

OUT_HTML = "/home/user/sglang-omni/sglang_omni_pr_report_2026-05-31.html"
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(HTML_DOC)
print("HTML written:", OUT_HTML, len(HTML_DOC), "bytes")

# PDF
from weasyprint import HTML as WHTML
OUT_PDF = "/home/user/sglang-omni/sglang_omni_pr_report_2026-05-31.pdf"
WHTML(string=HTML_DOC).write_pdf(OUT_PDF)
import os
print("PDF written:", OUT_PDF, os.path.getsize(OUT_PDF), "bytes")
