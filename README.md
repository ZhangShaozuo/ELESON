# ELESON：从原始 CSV 到训练与评估

第二轮逐式核对、修正和可调配置说明见 [PAPER_REVIEW.md](PAPER_REVIEW.md)。新默认缓存为 pipeline v3；旧缓存与 checkpoint 保留供对照。

本目录实现所附论文《Elevator, Escalator, or Neither? Classifying Conveyor State Using Smartphone under Arbitrary Pedestrian Behavior》（arXiv:2405.03218v3）的可运行复现尝试。保留三个原始数据目录，通过独立缓存连接数据清洗、标注解析、网络、损失、训练和评估。

**这是依据论文补足细节的实现，不是作者源码，也不承诺得到论文数字。** 完整训练尚未执行；`runs/smoke` 仅验证程序能训练、保存、重载和评估，不代表实验成绩。

## 快速运行

在本目录打开 PowerShell。当前已创建 `.venv` 并安装 CPU 版 PyTorch 和 NumPy；可直接运行：

```powershell
# 全量解析三个目录，生成缓存和数据审计
.venv/Scripts/python.exe -m eleson prepare --root . --output cache/review_v3

# 先跑一折
.venv/Scripts/python.exe -m eleson train --cache cache/review_v3 --output runs/eleson --fold 0 --epochs 100

# 正式五折；每折独立初始化，使用该折训练集的标准化参数
.venv/Scripts/python.exe -m eleson train --cache cache/review_v3 --output runs/eleson_5fold --fold all --epochs 100

# 重载最佳模型，在它原先的测试录制上评估
.venv/Scripts/python.exe -m eleson evaluate --cache cache/review_v3 --checkpoint runs/eleson/fold_0/best.pt --output runs/evaluation

# 关键逻辑测试
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

迁移到其他环境时，用 Python 3.10+ 创建虚拟环境并 `python -m pip install -r requirements.txt`。在已支持 CUDA 的 PyTorch 环境中，训练/评估传 `--device cuda`。这里安装的环境是 CPU 版，不能通过单独修改该参数获得 GPU 支持。

`requirements-lock.txt` 记录本机此次验证的精确依赖版本。`IMPLEMENTATION_CHECKS.json` 保存测试与运行检查摘要，包括 17 项测试、五折录制隔离、商场隔离、默认网络训练与 checkpoint 重载，以及 12 个真实窗口的固定批过拟合诊断。后者仅确认优化链路能学习，不是泛化结果。

快速检查可以限制录制数、批次数和网络宽度；不要把这种配置的分数用于对照论文：

```powershell
.venv/Scripts/python.exe -m eleson prepare --output cache/smoke --limit 40
.venv/Scripts/python.exe -m eleson train --cache cache/smoke --output runs/smoke_new --fold 0 --epochs 1 --max-steps 3 --batch-size 16 --hidden 8 --threads 2
```

## 数据到输入的 GAP 如何补齐

内部标签统一为 `0=neither, 1=elevator, 2=escalator`，与轨迹文件的原始标签不同。

|数据来源|处理方式|
|---|---|
|Single_Label_Data|读取路径中的 Elevator / Escalator / Neither，整段赋值。|
|Segmented_Data|读取 `speak.csv`；分别将每种事件的第 1/2、3/4…个时间戳配为区间。`cross-layer label` 采用路径中的梯种；`still label` 为 neither。支持同一录制多次上下梯。|
|Trajectory_Data|将两个 `Label_file_*.csv` 的 `traj_id / relative_time / label` 三行块合并。0、1 映射为 neither；3 映射为 elevator；4 映射为 escalator。重复且相同的标注去重，冲突 tick 标为未知。|

1. 通过列名识别时钟，优先使用 `imuTimestamp`。旧格式 `elapsedRealtime` 是回调时间，可能连续多行相同，不能代替采样时刻；新格式只有 `elapsedRealtime`，使用该列。
2. 旧格式通过 `median(currentTimeMillis - elapsedRealtime)` 连接设备启动时钟与 epoch 毫秒，以对齐 `speak.csv`。三个传感器使用共同设备时钟，而非分别将起点归零。
3. 去除非有限数值，按时间排序、去重，在三路传感器交集上做 100 Hz 线性插值。默认超过 100 ms 的缺口无效；禁止跨大缺口生成训练样本。
4. 加速度除以 9.80665 转为 g 尺度，角速度保留 rad/s。磁场先取三维模长，再做相邻采样差分（不是乘采样率的导数），对应式 (21)。差分在整段录制计算，避免每个窗口首点被错误置零。
5. 输入为 `[N,200,7]`：加速度 3 轴、陀螺仪 3 轴、磁强度差分 1 维。默认窗口和步长均为 2 秒。窗口中任何未知/缺测采样都剔除；默认仅保留标签一致的窗口；`--purity 0.9` 可允许少量标签过渡。
6. 行为扰动伪标签用窗口内原始角速度模长的均值是否超过 1.5 rad/s；该阈值来自论文，窗口聚合方式是本实现补充。
7. 仅从训练录制拟合逐通道均值和标准差，验证、测试、重载模型复用同一组参数；不按窗口去均值，以免抹去有意义的运动信号。

轨迹标签采用左闭右开区间，并假定标签相对时间从原始加速度流首采样开始；不能从三传感器交集起点开始，否则会引入时间偏移。相邻标签通常间隔 1280 ms，末标签仅延续一个该长度区间，不无限外推。这个锚点/区间语义是公开 readme 未明确给出的可审计假设。

部分中文目录发生了 UTF-8 被当作 CP850 解码的乱码。代码只在内存中恢复路径语义并识别“扶梯/电梯”，不重命名原文件。同目录多次采集按最接近的文件名时间配对传感器，并再次检查实际时钟交集。列数不完整的传感器行被剔除并写入审计，保留采样时间以继续检查缺口。

分段数据区间外的标签默认未知：公开说明只保证标注区间的梯状态，不能无条件假设整个背景都是 neither。要比较另一种解释，可独立生成缓存：

```powershell
.venv/Scripts/python.exe -m eleson prepare --output cache/background_neither --boundary neither
```

`cache/review_v3/audit.json` 保存跳过文件、原因、修复警告、有效录制和类别统计；缺失 `speak.csv`、缺失传感器、损坏事件名称不会被悄悄猜标签。公开目录与论文报告的 36,420 个窗口不保证一致，应以审计统计为准。

## 网络和论文公式

|模块|本实现|
|---|---|
|运动编码器|6 通道 → 两层 1D ConvLSTM → FC(256)+ReLU → FC(256)+ReLU，拆为 `zc(128), zp(128)`。|
|磁场编码器|强度差分 1 通道 → 两层 1D ConvLSTM → FC(256)+ReLU → FC(128)+ReLU，得到 `zb`。|
|ConvLSTM 具体化|每 10 个连续采样组成一个 frame，共 20 个递归时间步；卷积沿 frame 内采样位置执行。两层隐通道默认 32，核大小 3，最后状态展平。是卷积门控 LSTM，不是将整个窗口当单时间步。|
|信号生成器|`zc+zp+GaussianNoise` → FC(256)+ReLU → FC(512)+ReLU → FC(1200)，还原标准化运动信号，式 (11)。|
|模拟干预分类器|FC(128)+ReLU → FC(3)，分别监督 `zc+zp` 为原标签、`zp` 为 neither，式 (14)。|
|类内稳定约束|每类 `zc` 在 batch 样本维度的方差，再平均，式 (16) 的 mini-batch 实现。单样本类不贡献方差，避免 NaN。|
|行为判别器|FC(64)+ReLU → FC(2)，仅作用于 elevator/escalator 样本；梯度反转实现式 (24) 的 min-max。|
|证据分类器|拼接 `[zc,zb]` → FC(128)+ReLU → FC(64)+ReLU → FC(3)+ReLU。|

采用论文给出的 `w1=0.6, w2=0.3, w3=0.4, w4=1`。总损失和梯度对应式 (17)、(24)、(30)–(33)。行为判别器本身最小化 CE，磁场编码器最大化 CE，不能简单把负 CE 同时用于更新二者。

式 (30) 实现为 `-log(e_y / sum(e))`，不是把概率再次送入 CrossEntropy 的 softmax。全零证据采用 epsilon 归一化保证数值有限。证据末层使用正偏置初始化，以降低 ReLU 初始化即全部死亡的风险。

式 (31) 直接计算 Dirichlet 方差，`alpha=e+1`、`A=sum(alpha)`。推理置信度严格采用式 (26) 的 **`C=e/(sum(e)+3)`**，不混用 `(e+1)/sum(e+1)`。不确定度 `u=3/(sum(e)+3)`，保证 `sum(C)+u=1`；最大置信度不大于阈值时输出 UD。

## 论文没有公开、这里补充的设置

- ConvLSTM 内部组织、卷积核、隐层宽度、信号生成器中间宽度、行为判别器宽度、Gaussian 噪声标准差均未完整披露；上述是明确的工程选择。128 维特征、两层时序模块、FC 层数、ReLU、损失权重按论文。
- 优化器为论文的 Adam；学习率 1e-3、batch size 128、最多 100 epoch、验证 F1 早停 20 epoch、梯度裁剪 5、噪声标准差 0.05 是本实现默认值，可通过 CLI 修改。
- 论文引用 RoNIN 做预处理。RoNIN 的坐标归一化依赖姿态信息，公开 CSV 未提供相同的姿态四元数/标定信息，**不能宣称精确重建 RoNIN 处理**。默认 `--frame device` 保留设备坐标；`--frame gravity` 用 0.5 秒低通估计重力并近似对齐竖直轴，没有可靠的航向校正。强动态动作下估计会受污染，应作为预处理对照而非论文原版。
- 论文五折描述为打乱采集的信号序列，具体分组和随机种子不公开。本实现按完整录制分组做尽量平衡的五折，避免同录制窗口进入不同 split；原始加速度起止时间与首尾片段指纹完全相同的重复捕获合并分组。只能识别这种完整重复，不能保证识别任意部分重叠片段；公开数据也不足以保证 subject-independent 划分。
- 每折外层测试约 20% 录制，剩余录制随机取 10% 做内部验证，因此训练/验证/测试大致 72%/8%/20%，按录制数而非窗口数。最佳 checkpoint 仅由验证集的阈值 0 conveyor F1 选取，最终再评估测试集。
- 模型参数量写入各折 config，不强行调成论文的 9 MB；训练结果与效率均需独立测量。

RoNIN 原作者资料：[项目页](https://ronin.cs.sfu.ca/index.html)、[原作者代码](https://github.com/Sachini/ronin)。具体论文约束请对照本目录 PDF 第 III–V 节。

## 输出与指标解释

每折输出 `config.json`（参数、录制划分、训练 scaler）、`history.json`（所有损失分量、验证指标）、`best.pt`、`metrics.json`、`predictions.csv`。运行根目录 `summary.json` 包含折间均值/标准差和合并测试样本指标。

- 主要 F1 为 elevator 和 escalator 的 F1 均值，neither 的指标另报，不用三类 macro-F1 冒充论文指标。
- 默认同时输出阈值 0 和 0.5、UD 比例、coverage、每类 precision/recall/F1，以及 3×4 混淆矩阵（真实三类 × 预测三类加 UD）。
- `conveyor_f1` 保留所有测试样本，UD 的真实正类仍计为漏检；`selective_conveyor_f1` 仅在接受的样本上计算。论文未完全明确丢弃 UD 后的评估分母，故同时给出两种，防止通过丢弃难样本虚增成绩。
- `confidence_auroc` 以“预测正确”为正样本，置信度为分值，评估正确/错误预测的区分度；仅一种正确性标签时返回 null，不伪造 AUC。并列置信度使用平均秩。
- CSV 保存原始录制路径、窗口编号、起始设备时刻、真值、预测、置信度、不确定度、来源与行为级别，便于之后画轨迹和定位误差。

## 已支持的对照实验

```powershell
# 同骨干的纯监督 softmax 基线（不是其他论文作者原版）
.venv/Scripts/python.exe -m eleson train --model e2e --output runs/e2e --fold all

# 删除因果辅助损失 / 磁场对抗 / 不确定度项
.venv/Scripts/python.exe -m eleson train --no-causal --output runs/no_causal
.venv/Scripts/python.exe -m eleson train --no-adversarial --output runs/no_adversarial
.venv/Scripts/python.exe -m eleson train --no-uncertainty --output runs/no_uncertainty

# 只向证据头输入运动特征 / 磁场特征；辅助分支仍保留
.venv/Scripts/python.exe -m eleson train --features motion --output runs/motion
.venv/Scripts/python.exe -m eleson train --features magnetic --output runs/magnetic

# 未见商场测试：只用有商场标签的分段数据，整座 Mall7 留作测试
.venv/Scripts/python.exe -m eleson train --protocol mall --holdout-mall Mall7 --sources Segmented_Data --fold 0 --output runs/mall7

# 比较近似重力对齐；需生成独立缓存并重新训练
.venv/Scripts/python.exe -m eleson prepare --frame gravity --output cache/gravity
.venv/Scripts/python.exe -m eleson train --cache cache/gravity --output runs/gravity
```

没有实现论文所有外部基线、手机端部署和功耗测量。当前优先完成从这份公开数据到 ELESON 网络及核心评估的闭环。

## 当前数据审计快照

严格默认配置审计了 821 个加速度录制入口，保留 726 段录制、16,184 个窗口：

|来源|有效窗口|
|---|---:|
|Segmented_Data|12,725|
|Single_Label_Data|1,198|
|Trajectory_Data|2,261|

类别计数为 neither 7,896、elevator 4,377、escalator 3,911。另有 35,970 个候选窗口因未知标签、过渡或缺测被排除；95 段录制被跳过：87 段缺少上下梯标注或时钟连接、6 段路径使用尚未确认语义的 `straightladder`、1 段事件名称损坏、1 段缺少陀螺仪。不直接将 `straightladder` 猜成某一梯种。

这些数量描述当前公开文件和本实现的严格策略，不等同于论文实验集规模。重跑 `prepare` 后以生成的 `audit.json` 为准。

## 文件职责

`eleson/data.py` 负责原始解析、时钟对齐、标注和缓存；`model.py` 负责模型与论文公式；`engine.py` 负责划分、训练、保存和预测；`metrics.py` 负责 F1/UD/AUROC；`__main__.py` 提供 CLI；`tests/test_pipeline.py` 验证关键数据与梯度不变量。
