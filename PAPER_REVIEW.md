# 论文与实现逐项核对（第二轮）

核对来源：本目录 `2405.03218v3.pdf`，第 III–V 节，式 (8)–(33)，实验设置与参数研究；以及数据自带 `Readme.txt`。以下区分论文明确给出的内容、实现补充的选择、已修正的问题。论文未公开的内容不以“原版”名义推断。

## 本轮直接修正

|位置|原问题|当前处理与影响|
|---|---|---|
|式 (16) 类内方差|各非单例类别的方差等权平均，未体现公式中对每个样本的求和。类别不平衡时改变 w2 的有效意义。|默认 `sum_c (n_c/B) * mean_feature Var_batch(zc_c)`，总体方差采用 `unbiased=False`。单例类别方差为零。保留 `class` 旧模式用于对照。|
|式 (24) 对抗项|CE 仅除以梯样本数；其他逐样本项除以整个 batch 大小。不同 batch 梯样本占比变化时，对抗项相对论文的统一归约被放大。|默认将梯样本 CE 求和除以 B；保留 `conveyor` 旧模式。论文未说明实际代码 reduction，本修改采用全式统一除以 B 的约定，不能保证作者采用相同约定。|
|式 (21) 首磁差分|没有前一个采样，却人为填零进入首窗口。|默认删除差分首点，实际切窗起点后移一个采样（100 Hz 时为 10 ms），保留完整 200 点窗口。`--diff-start zero` 仅用于旧实现对照。|
|近似重力对齐|低通重力估计跨越缺测区间传播旧状态。|遇到无效采样或缺口后的首点重置估计。缺口仍通过有效性掩码剔除。|
|可复查性|checkpoint 未包含测试路径，单独复制 checkpoint 无法恢复 split。|新 checkpoint 内嵌录制路径和实际损失参数；评估兼容旧 checkpoint 的旁置 config。|
|可调性|许多未知宽度写死，特征维度也不能从训练入口调整。|通过 `--model-config` / `--loss-config` JSON 配置，解析后完整保存到运行 config 与 checkpoint。未知键报错；卷积核必须为正奇数。|

`optimization_loss` 是带 GRL 的优化标量，其中行为 CE 为正，不能直接当作式 (33) 的数值。新增 `encoder_objective` 记录带负行为 CE 的式 (33) 数值；两者的 encoder 梯度一致，而判别器需用正 CE 更新。

## 数据预处理核对

|项目|论文或数据说明|本实现与未确定之处|影响与调节|
|---|---|---|---|
|采样与窗口|100 Hz；窗口、步长均 2 秒。|按传感器共同时间区间插值，200 点。|默认一致；`--hz --seconds --stride` 可调。|
|时钟|CSV 同时有系统时钟、回调时刻、采样时刻。论文未给转换代码。|优先 imuTimestamp，用 wall-clock 与 elapsedRealtime 的中位差连接 speak 的 epoch 时间；回调时间不能用于逐点插值。|设备时钟异常/漂移可能影响标注边界；未做额外漂移拟合。|
|标签|三分类。原始轨迹 0/1/3/4。|轨迹 0/1→neither，3→elevator，4→escalator；两个标签文件按 tick 合并，冲突未知。|映射有数据说明支持；不是将 walking 作为第四类。|
|轨迹时间锚|readme 未说明 relative_time 锚点和每个 tick 的覆盖语义。|以原始加速度首采样为零；左闭右开，按相邻 tick 间隔限长；末 tick 延长一个中位间隔。|高影响假设。`--trajectory-offset-ms` 可移动标签起点；正值使标签在设备时间中更晚开始。尚无独立真值证据确认锚点。|
|分段事件|readme 描述一对上下梯时刻；文件实际包含多对事件及 still。|按事件类型分别逐对配对，still→neither，cross-layer→路径梯种。拒绝重叠、未知名称、缺标注。|多对事件的配对是数据驱动解释；区间外默认未知，可 `--boundary neither` 对照。|
|缺测与过渡|论文未给处理策略。|大于 100 ms 的插值桥无效；差分前后任一点无效则该差分无效；默认全窗口同标签。|会减少有效窗口；`--max-gap-ms --purity` 可调。降低 purity 仍不接受未知标签。|
|坐标与标定|论文仅说依照 RoNIN 的预处理，未给所用步骤/姿态来源。|数据无同等姿态四元数和标定数据。默认设备坐标，无传感器偏置估计；另提供低通重力近似。|**主要未闭合缺口**。近似对齐不能替代 RoNIN，不恢复航向；强动态污染重力估计，参考轴切换还可能造成水平轴不连续。`--frame gravity --gravity-tau` 用于对照。|
|单位和尺度|CSV 为 m/s²、rad/s、uT；论文未给标准化细节。|加速度先除 9.80665，不去重力；再用训练集逐通道标准化。重建目标也在标准化空间。|标准化改变各运动通道在重建误差中的权重，因此 w1 不保证与作者等效。`--acc-scale` 可控制预处理单位；`train --normalization none` 关闭标准化。开启标准化时 acc-scale 的常数缩放基本被抵消。|
|磁差分|式 (21)：相邻磁场模长之差。|三轴先插值、再求模长、再差分；不乘 fs，也不是三轴各自差分。|公式形式一致。先求模再插值是另一种未实现的处理顺序，作者没有披露插值顺序。|
|扰动伪标签|1.5 rad/s 阈值，未说轴、模长或时间聚合方式。|使用窗口中角速度模长均值；在标准化前计算。|高影响假设。`--behavior-threshold`、`--behavior-reducer mean/rms/max` 均可调；mean、RMS、max 产生不同扰动比例。|

新默认预处理为 pipeline v3，缓存示例使用 `cache/review_v3`。旧 `cache/full` 保留为 pipeline v2，以便原 checkpoint 对照。两者不能当作同一套实验混用；评估会核对预处理配置。

## 网络结构核对

论文明确：运动分支和磁场行为过滤分支均为“两层 ConvLSTM + 两层 FC + ReLU”；zc 与 zb 各 128 维；生成器三层 FC；模拟干预分类器两层 FC；证据头三层 FC 且输出 ReLU。

默认实现保留上述层数和激活，但以下均为补充假设：

|选择|当前默认|可调整位置|
|---|---|---|
|ConvLSTM 解释|两层真正带卷积门的 1D LSTM；将 200 点分为 20 个 10 点 frame，以 frame 为递归时间轴，frame 内位置为卷积维。|`backbone`, `frame_samples`；论文没有指定 frame 划分。|
|替代骨干|`cnn_lstm`：两层时间卷积 + 两层普通 LSTM，按每个采样递归。|`backbone: "cnn_lstm"`；这只是另一种合理解释，**不是确认过的论文骨干**，额外卷积使层组织也不同。|
|隐通道、卷积核|32、3，same padding；最后时刻状态展平。|`hidden`, `kernel_size`；frame_samples 必须整除窗口点数，仅适用于 convlstm。|
|编码器 FC|隐藏宽度 256；运动输出 256 拆为 zc/zp 各 128，磁场输出 128。|`encoder_width`, `feature_dim`；zp 与 zc 相加要求同维，属于从公式推导的约束。|
|生成器|128→256→512→1200，隐藏层 ReLU，输出线性。|`generator_widths`；线性输出是为重建有符号信号补充的选择。|
|模拟干预分类器|128→128→3，末层 logits。|`intervention_width`。|
|行为判别器|128→64→2；论文未说明该判别器结构。|`adversary_width`。|
|证据头|256→128→64→3，所有 FC 后 ReLU。|`evidence_widths`。|
|噪声与初始化|latent σ∼N(0,0.05²I)；证据末层偏置初始化为 1。|`noise_std`, `evidence_bias`；论文只明确 Gaussian，未给方差或初始化。|

没有将网络尺寸硬凑为论文的约 9 MB。默认约 119 万参数；配置改变后实际参数量记录于 config。FC 数量、ReLU 和特征维度对齐不代表内部结构已唯一确定。当前 `features motion/magnetic` 是证据头输入消融，仍计算辅助分支；`no-causal` 是删除辅助损失，不能解释成删除了整个运动分支。`model e2e` 为本实现的监督对照，不是论文外部基线的官方复现。

## 损失函数逐式核对

|公式|实现|判断|
|---|---|---|
|(11)|MSE(g(zc+zp+noise), normalized_motion)，对每个样本的时间/通道取均值，再对 batch 平均。|形式一致；目标空间、noise 方差为假设。|
|(14)|CE(k(zc+zp), y)+CE(k(zp), neither)，同一个 k 共享参数。|一致；未错误地将 zp 监督为原标签。|
|(15)–(16)|仅约束 zc 的类内方差，不额外添加 Var(zc)<Var(zp) 的 hinge。|符合作者最终使用式 (16) 的描述；Var 的特征维归约未公开，这里取均值。默认 sample reduction 更贴近统一逐样本平均，batch 估计不等于全数据方差。|
|(17)|Lsim+0.6Lrec+0.3Lcon。|权重一致；不同归约和标准化仍会改变有效尺度。|
|(23)–(24)|仅 y!=neither 样本的行为 CE；正向 GRL 恒等，反向乘 -1。|“s=1”为简化二元记号，包含 elevator 和 escalator。判别器最小化 CE，磁场编码器最大化 CE；梯度方向已测试。没有额外按梯种分开判别器，作者未指定这种组织。|
|(25)–(29)|e≥0，C=e/(sum(e)+3)，u=3/(sum(e)+3)，argmax(C)，严格 max(C)>τ 才接受。|一致；没有把 Dirichlet 均值 (e+1)/S 当作置信度。|
|(30)|-log((e_y+eps)/(sum(e)+3eps))。|概率上的交叉熵；没有再次 softmax，也未替换为常见但不同的 digamma EDL 损失。eps 与极小值裁剪是数值补充，全零证据用均匀概率，置信度仍为零。|
|(31)–(32)|alpha=e+1，求和 alpha_i(S-alpha_i)/(S²(S+1))，加入分类损失。|按公式实现，没有擅自加入 Dirichlet KL。`variance_weight=1` 默认一致，可做对照。|
|(33)|Lels+0.4 Lmag+1 Lcal，联合端到端。|用 GRL 实现 min-max；优化标量与 encoder 数值分开记录，避免符号误读。|

## 结果可比性与剩余 GAP

- 文中 36,420 个实例不等于本地按严格策略可用的窗口数。有效样本变化、区间外标签解释、多来源合并与重复数据都会改变类别分布和难度；不能只调模型来消除数据数量差异。
- 五折按完整录制和完整捕获指纹分组，内部另划 10% 剩余录制作验证；这不同于论文不够明确的“shuffle the signal sequences”。不做 window 随机切分。没有人员 ID，无法确认跨人测试；部分重叠片段尚未自动识别。
- UD 后的样本分母论文不够清晰，继续同时报告全体样本 F1 与仅接受样本的 selective F1，连同 coverage。主 F1 仅平均 elevator/escalator，不是三类宏平均。
- Adam 明确；学习率、batch、早停、梯度裁剪和训练时长是工程默认。没有正式五折长训练，也没有手机部署/功耗验证。
- 旧实验结果不能作为新版本成绩：首差分起点、损失归约和可选标准化会影响训练。必须使用独立输出目录重训。

## 调整与运行

在 `configs/model.json` 中调整结构，在 `configs/loss.json` 中调整归约和权重。JSON 中的同名字段覆盖基础 CLI 参数；最终生效内容写入每折 `config.json` 的 `model_options/loss_options`。`samples` 始终由缓存决定，禁止在 JSON 中覆盖。权重文件没有写入 causal/adversarial/uncertainty 时，CLI 的 `--no-*` 仍然生效。

```powershell
.venv/Scripts/python.exe -m eleson prepare --output cache/review_v3
.venv/Scripts/python.exe -m eleson train --cache cache/review_v3 --output runs/review_v3 --fold 0 --model-config configs/model.json --loss-config configs/loss.json

# 更换骨干：复制 model.json，改 backbone 为 cnn_lstm，使用独立输出目录
# 比较旧归约：--loss-config configs/loss_legacy.json
# 比较标准化：train ... --normalization none
# 比较扰动聚合：prepare ... --behavior-reducer rms --output cache/rms_v3
.venv/Scripts/python.exe -m unittest discover -s tests -v
```
