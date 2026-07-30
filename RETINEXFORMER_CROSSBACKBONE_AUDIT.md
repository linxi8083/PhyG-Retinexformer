# Retinexformer 跨骨干验证：静态审计与执行计划

审计日期：2026-07-30  
审计范围：只读代码、配置、Git 元数据和本机既有 `PhyG-PEFT` 协议文件；未读取或运行任何
官方 Test/Eval 数据，未下载或加载权重，未训练，未修改训练/网络代码。

## 1. 结论摘要与当前停止点

1. 当前仓库位于开发分支 `phyg-crossbackbone`，HEAD 为
   `e8086f9d6e4badf0eebf48825264a785d3aa541e`。本次审计开始时工作区干净。
   本地 `author-baseline`、远端跟踪分支 `origin/author-baseline`、
   `origin/phyg-crossbackbone` 和冻结标签 `retinexformer-official-frozen` 均指向该提交。
   `git diff retinexformer-official-frozen...HEAD` 与
   `git diff author-baseline..HEAD` 开始时均为空，故受 Git 跟踪的官方代码无差异。
2. 冻结标签存在：
   `refs/tags/retinexformer-official-frozen ->
   e8086f9d6e4badf0eebf48825264a785d3aa541e`。本报告只新增于开发分支工作区，
   不移动标签、不改写 `author-baseline`。
3. 存在一个**实施前必须修复的仓库完整性阻断项**：`.gitignore:2` 的 `data/`
   会匹配任意层级同名目录，因此本机官方源码目录 `basicsr/data/` 全部被忽略且未被
   HEAD/冻结标签跟踪。`git check-ignore -v basicsr/data/paired_image_dataset.py`
   明确命中 `.gitignore:2`；`git ls-files basicsr/data` 为空。当前机器之所以能看到
   数据集类，是因为它们是 ignored 本地文件，干净 clone 会缺失这些源码。不能在该状态下
   声称冻结标签完整包含官方训练实现。
4. Physical 正式参考可以访问，无需猜参数：本机
   `C:/Users/hxy/Desktop/PhyG-PEFT/phyg_peft/physical_degradation.py:1-148`
   是标记为 frozen protocol-v1 的实现；固定 810/90 manifest 也存在。必须做“逐字复用
   或有 SHA-256 约束的移植”，不能重新设计。
5. 最小接入方式是不改 `RetinexFormer_arch.py`，而在新增的独立训练入口中，于官方成对
   crop/flip 后、送入 `net_g` 前，仅变换 `lq`。Identity、Gamma、Physical 三个训练臂
   必须经过同一入口、同一 manifest、初始化权重、优化器、scheduler、loss、step 数、
   validation 和 checkpoint 选择规则；唯一实验因子是 replay 变换。
6. 官方 LOLv2-Synthetic 配置把 `val` 指向官方 Synthetic Test
   （`Options/RetinexFormer_LOL_v2_synthetic.yml:43-49`），因此 PhyG 配置**禁止继承该
   val 段**。方法开发只能构造 manifest 中 90 对 Development Validation。
7. 官方 checkpoint/resume 不满足本项目要求；需由新增入口完整实现。PyTorch 2.6 加载
   本项目自己生成且可信的完整 checkpoint 时必须显式 `torch.load(...,
   weights_only=False)`，且不能沿用官方隐式加载。
8. 本报告完成后停止；没有确认不得实现、预检数据、加载权重、运行验证或训练。

## 2. Git、远端、分支与冻结状态

### 2.1 审计命令和结果

静态命令：

```text
git status --short --branch
git rev-parse HEAD
git branch -vv
git remote -v
git show-ref --tags
git diff --name-status retinexformer-official-frozen...HEAD
git diff --name-status author-baseline..HEAD
```

结果：

```text
branch: phyg-crossbackbone
HEAD: e8086f9d6e4badf0eebf48825264a785d3aa541e
author-baseline: e8086f9 (tracks origin/author-baseline)
phyg-crossbackbone: e8086f9 (tracks origin/phyg-crossbackbone)
origin fetch/push: https://github.com/linxi8083/PhyG-Retinexformer.git
tag retinexformer-official-frozen: e8086f9d6e4badf0eebf48825264a785d3aa541e
initial tracked diff against tag: empty
initial tracked diff against author-baseline: empty
initial porcelain status: empty
```

保护结论：开发只能继续发生在 `phyg-crossbackbone`。后续每次写入前应强制检查
`git branch --show-current` 等于 `phyg-crossbackbone`，并检查标签目标仍为上述 SHA。
禁止 checkout 后在 `author-baseline` 上写入，禁止 force-update 标签，禁止将 CIDNet 网络、
loss 或 HVI/PEFT 代码合入本仓库。

### 2.2 冻结标签的完整性缺口

- `.gitignore:1-3` 用 `data/`、`datasets/` 排除数据。
- 但 Git ignore 的 `data/` 同时排除了 `basicsr/data/`；该目录包含创建 DataLoader、
  paired dataset、sampler、crop/augmentation 等官方训练必需源码。
- `basicsr/train.py:12-14` 直接导入 `basicsr.data`、`EnlargedSampler` 和 prefetcher；
  干净 clone 缺少 ignored 源码将无法训练。
- 本地具体实现见 `basicsr/data/__init__.py:16-27,31-54,57-127` 和
  `basicsr/data/paired_image_dataset.py:18-133`，但二者当前都不在
  `git ls-tree -r retinexformer-official-frozen` 中。

建议实施第一步只在 `phyg-crossbackbone`：

1. 把 ignore 规则锚定到仓库根数据目录（例如 `/data/`、`/datasets/`），保留
   `basicsr/data/` 可跟踪；
2. 从可验证的官方来源/当前已审计副本恢复 `basicsr/data/`，记录文件清单及 SHA-256；
3. 单独提交“repository completeness fix”，不得移动冻结标签；报告中永久注明：
   冻结标签的 tracked tree 缺失 `basicsr/data/`，开发分支只做源码完整性修复。

## 3. 官方 Retinexformer 实现定位

### 3.1 网络定义

- 配置选择 `RetinexFormer`，参数为 RGB 3→3、`n_feat=40`、`stage=1`、
  `num_blocks=[1,2,2]`：`Options/RetinexFormer_LOL_v2_synthetic.yml:51-58`。
- 网络动态发现与实例化：`basicsr/models/archs/__init__.py:7-18,23-48`。
- 主类：`basicsr/models/archs/RetinexFormer_arch.py:342-359`；其 forward 只是
  `self.body(x)`，没有必要为 Physical 修改。
- 单 stage 的照明估计及恢复路径：
  `basicsr/models/archs/RetinexFormer_arch.py:323-339`。
- Denoiser encoder/bottleneck/decoder/residual mapping：
  `basicsr/models/archs/RetinexFormer_arch.py:234-320`。

结论：跨骨干验证不得改上述网络。新增入口用原 `define_network` 构造，严格加载相同官方
Synthetic 权重的 `params`。

### 3.2 LOLv2-Synthetic 官方配置

文件：`Options/RetinexFormer_LOL_v2_synthetic.yml`。

- 训练根为 `data/LOLv2/Synthetic/Train/{Normal,Low}`：
  `Options/RetinexFormer_LOL_v2_synthetic.yml:9-19`。
- shuffle、8 workers、batch 8：同文件 `:21-24`。
- 固定 128 patch，dataset 内阶段预算写为 300000：
  同文件 `:33-40`。
- 顶层训练总步数却写为 150000：同文件 `:67-80`。注释声称 300k/92k/208k，
  实际 `total_iter` 与 scheduler periods `[46000,104000]` 合计 150k；实施时以解析值
  为准，不能以注释为准。
- mixup 开启且 identity 开启：同文件 `:82-85`。
- Adam，lr `2e-4`，betas `[0.9,0.999]`：同文件 `:87-91`。
- L1 mean、weight 1：同文件 `:93-97`。
- 官方 val 指向 Synthetic 官方 Test：同文件 `:43-49`，开发阶段严禁使用。
- val 每 1000 iter、PSNR、无 crop、非 Y 通道：同文件 `:99-117`。

### 3.3 数据集类与 DataLoader

- `Dataset_PairedImage` 定义在
  `basicsr/data/paired_image_dataset.py:18-133`。
- folder/meta-info/lmdb 三种路径构建分支在同文件 `:61-74`。这意味着可以用 manifest
  派生的显式路径列表/专用 dataset，无需重新随机拆分。
- 读取结果是 `[0,1]` float32 BGR，训练阶段成对 pad/crop/几何增强，再转 RGB tensor：
  同文件 `:79-129`。
- 动态数据集注册：`basicsr/data/__init__.py:16-54`。
- 训练 loader 使用 sampler、`drop_last=True`，worker 只设置 NumPy/Python seed，
  没有显式 DataLoader generator：`basicsr/data/__init__.py:57-127`。
- 官方 `EnlargedSampler` 被训练入口构造：
  `basicsr/train.py:96-145`，且每 epoch 调 `set_epoch`：
  `basicsr/train.py:249-254`。

结论：为保证 exact resume 和固定文件名 split，优先新增独立
`DevelopmentPairDataset`/sampler，不改官方类。其输出保持官方键
`lq/gt/lq_path/gt_path`，训练 crop/flip 语义需预先冻结。DataLoader 必须显式接受并保存
`torch.Generator`，protocol-v1 继续采用 `num_workers=0`，否则 prefetch 会破坏中断时的
精确 batch/RNG 轨迹。

### 3.4 训练入口

- 入口参数、配置解析和 seed：`basicsr/train.py:28-62`。
- train/val loader 构建：`basicsr/train.py:96-145`。
- 官方自动扫描 `experiments/<name>/training_states` 并加载最大 state：
  `basicsr/train.py:148-174`。
- 建模与 resume：`basicsr/train.py:187-210`。
- progressive/fixed patch、batch 选择、crop、feed/optimize：
  `basicsr/train.py:235-300`。
- 保存、validation、best PSNR：`basicsr/train.py:311-340`。
- 训练结束再次 validation：`basicsr/train.py:350-357`。

风险：官方 `while current_iter <= total_iters` 和自动选择目录中最大 state 的行为不适合
严格实验编排；新增入口应要求显式 `--resume`，并以 `global_step` 为唯一预算计数。

### 3.5 loss、optimizer、scheduler

- `ImageCleanModel` 根据配置动态构造 pixel loss：
  `basicsr/models/image_restoration_model.py:122-133`。
- Adam/AdamW 构造：同文件 `:135-156`。
- 前向、L1、AMP backward、global norm clip=0.01、optimizer step：
  同文件 `:171-202`。
- `L1Loss` 公式及 reduction：`basicsr/models/losses/losses.py:11-12,24-51`。
- scheduler 类型分发：`basicsr/models/base_model.py:87-133`。
- `CosineAnnealingRestartCyclicLR` 实现：
  `basicsr/models/lr_scheduler.py:186-232`；scheduler 在每个 optimizer step 前更新，
  `basicsr/train.py:259-264` 与 `basicsr/models/base_model.py:183-205`。

公平性建议：Retinexformer 三个训练臂都保留同一个 Retinexformer L1、Adam、clip 和
scheduler 协议，不复制 CIDNet/HVI loss。Identity/Gamma/Physical 只能改变 LQ replay。
60 epoch 微调时必须重新给出与总 step 匹配的 scheduler 配置；不能直接沿用 150k periods。
该选择在实现前需冻结成一个共享 base config。

### 3.6 checkpoint 保存与恢复

- 网络权重另存 `{params: state_dict}`：
  `basicsr/models/base_model.py:213-244`。
- training state 仅含 epoch、iter、optimizer、scheduler、kwargs/best metric 和可选 AMP：
  `basicsr/models/base_model.py:311-342`。
- resume 只恢复 optimizer、scheduler、可选 AMP：
  `basicsr/models/base_model.py:344-364`。
- `ImageCleanModel.save` 分开保存网络和 training state：
  `basicsr/models/image_restoration_model.py:347-355`。
- best 文件只含网络，且会删除旧 `best_*`：
  同文件 `:357-386`。
- 官方完整 state 加载位于 `basicsr/train.py:168-172`，没有 PyTorch 2.6 所需的显式
  `weights_only=False`。网络加载位于 `basicsr/models/base_model.py:281-309`，
  其中 `torch.load` 在 `:295`。

缺口：官方 state 没有保存 Python RNG、NumPy RNG、Torch CPU/CUDA RNG、DataLoader
generator、augmentation RNG、完整 config、split hash、initialization SHA-256，
也没有把 model 与训练状态放在同一个原子 checkpoint。必须由新增入口实现，不能只给
官方 `save_training_state` 加几个字段后宣称满足 exact resume。

### 3.7 Validation 与 Test 入口

- 训练内 validation 触发及以 PSNR 选 best：
  `basicsr/train.py:316-340`。
- validation 逐图 pad/inference、可选保存和 metric：
  `basicsr/models/image_restoration_model.py:243-326`。
- 单独 BasicSR Test 入口创建配置中的所有 dataset 后调用同一 validation：
  `basicsr/test.py:13-58`。
- 作者发布的独立评测入口加载 `pretrained_weights/*.pth`：
  `Enhancement/test_from_dataset.py:53-116`；其中 `--GT_mean` 是可选参数
  `:69`，实际校正发生在 `:184-189` 和 `:262-267`，本研究必须始终禁用。
- 该入口直接 glob 配置 val 目录、推理并计算 PSNR/SSIM：
  `Enhancement/test_from_dataset.py:204-281`。开发阶段不得运行。
- 官方配置的 val 就是 Synthetic Test，见
  `Options/RetinexFormer_LOL_v2_synthetic.yml:43-49`，是当前最重要的误触风险。

Development Validation 应使用新增独立入口/函数，只接受 manifest 的 90 个 Train
文件名，并在路径审计层拒绝任何 `Test`/`Eval` 根。正式 Test runner 应在方法冻结后另行
启用，并默认拒绝运行；本阶段不实现、不运行。

### 3.8 官方预训练权重预期路径

- README 要求模型放入 `pretrained_weights`：
  `README.md:418-420`。
- LOLv2-Synthetic 命令期望
  `pretrained_weights/LOL_v2_synthetic.pth`：
  `README.md:422-433`。
- 独立入口默认/参数加载方式见
  `Enhancement/test_from_dataset.py:63-65,100-111`。
- 官方训练 YAML 的 `pretrain_network_g` 当前为空：
  `Options/RetinexFormer_LOL_v2_synthetic.yml:61-65`；PhyG config 必须显式填入同一权重，
  strict-load `params`，并在所有臂启动前记录文件 SHA-256。

本审计未下载、未寻找或读取权重内容，因此初始化文件 SHA-256 尚未知，属于训练前必填项。

## 4. 冻结 Physical 协议复用判断

参考来源：

- 正式实现：
  `C:/Users/hxy/Desktop/PhyG-PEFT/phyg_peft/physical_degradation.py:1-148`。
- 冻结配置：
  `C:/Users/hxy/Desktop/PhyG-PEFT/configs/cidnet_physical_full_seed1234.json:31-85`。
- 实施报告：
  `C:/Users/hxy/Desktop/PhyG-PEFT/PHYSICAL_MODULE_IMPLEMENTATION.md:3-20`。

不可调整的事实：

- sRGB↔linear 公式：正式实现 `:15-24`。
- identity matrix、warm/cool frozen matrix：`:9-12`。
- identity probability=0.5、四个 family 等概率选择、strength `[0.5,1.0]`：
  `:50-56` 与冻结配置 `:31-45`。
- 四 family 的 exposure、shot/read noise、white balance、matrix 参数：
  正式实现 `:57-76`，配置公式 `:46-82`。
- `exposure`、`exposure_noise`、`exposure_color`、`full` 过滤规则：
  正式实现 `:50-52,77-85`。
- 处理顺序为 linearization→exposure→Poisson→read noise→WB→matrix→clip→sRGB→
  8-bit quantization：正式实现 `:88-111`。
- 独立 Python 参数 RNG 与 Torch noise generator，可保存/恢复：
  正式实现 `:114-148`。

接入点：

```text
manifest pair load
  -> paired crop/flip
  -> RGB float tensor [0,1]
  -> replay adapter applied to LQ only
     identity: unchanged
     gamma: frozen Gamma implementation
     physical: exact frozen Physical implementation
  -> unchanged RetinexFormer net_g
  -> unchanged shared L1 / optimizer / scheduler / validation
```

不能把 Physical 放进网络，也不能改变 GT。应优先复制为 `phyg/physical_degradation.py`
并同时记录来源文件 SHA-256，或以明确的外部依赖方式导入；为了服务器可复现和避免跨仓库
运行时耦合，建议在本仓库 `phyg/` 内做逐字复制并保留来源注释、哈希和等价性测试。
不得复制 CIDNet 网络、HVI loss 或 PEFT 代码。

Gamma 也必须从既有项目正式实现中逐字定位/复用；当前冻结配置只给出范围
`start_percent=60,end_percent=120`
（`C:/Users/hxy/Desktop/PhyG-PEFT/configs/cidnet_gamma_replay_seed1234.json:26-30`），
实现阶段仍需审计其具体采样/应用代码，不能仅凭这两个数字重写。

## 5. 固定 810/90 Development 划分

参考 manifest：

```text
C:/Users/hxy/Desktop/PhyG-PEFT/protocols/lolv2_synthetic_dev_split_seed20260726.txt
```

- 结构及 canonical hash 算法：
  `C:/Users/hxy/Desktop/PhyG-PEFT/phyg_peft/dataset_wrappers.py:13-29`。
- 强制数据根只能是 `LOLv2/Synthetic/Train`、Low/Normal 文件名相等、810/90、
  无交集且并集完整、hash 相等：
  同文件 `:32-46`。
- dataset 严格按 manifest 文件名加载：
  同文件 `:49-76`。
- frozen canonical split SHA-256：
  `e3e73abce5405d7fa4cbb7d1f16470dc924abf97c77208ba049da1613d0a944d`，
  见 `C:/Users/hxy/Desktop/PhyG-PEFT/configs/cidnet_physical_full_seed1234.json:84-85`。
- 本次只读计算得到 manifest **原始文件字节** SHA-256：
  `f80b8d5cebd3b8f6529b0670e91037320bf939dba6f16d8a5402861465f543d6`。
  原始 hash 与 canonical hash 用途不同；两者均应记录，不能混用。
- 文件共 902 行（2 个 section header + 810 + 90）。

复用方案：将 manifest 原样复制到 `configs/phyg/protocols/` 或 `phyg/protocols/`，
运行时同时验证 raw SHA 与 canonical SHA；数据集按文件名列表构造，严禁调用随机 split。
Development Train 只使用 `[train]` 810 名；Validation 只使用 `[validation]` 90 名。
路径守卫必须在枚举文件前拒绝 `Test`/`Eval`。本审计没有枚举或读取当前仓库
`datasets/` 内容。

## 6. `.gitignore` 审计

当前 `.gitignore:1-24`：

- 已排除 `data/`、`datasets/`；
- 已排除 `*.pth`、`*.pt`、`*.ckpt`、`pretrained_weights/`、`weights/`；
- 已排除 `experiments/`、`results/`、`runs/`、`wandb/`；
- 已排除 cache、IDE 和 `*.log`。

优点：数据、权重、主要结果与 checkpoint 类型已有覆盖。  
缺陷：

1. `data/` 未锚定，误伤 `basicsr/data/` 官方源码，是阻断项；
2. 建议补充根锚定 `/data/`、`/datasets/`，并显式允许/跟踪 `basicsr/data/**`；
3. 建议补充 `tb_logger/`（官方实际写入路径见 `basicsr/train.py:90-93`）；
4. 建议补充 `*.state`（官方 state 后缀见 `basicsr/models/base_model.py:339-342`）；
5. 建议补充 PhyG 自身输出根，如 `/outputs/`（若最终采用）；
6. 结果 metadata/config JSON 不能一刀切全部忽略：小型、去敏、无数据/权重内容的
   protocol metadata 应允许提交；大图和 checkpoint 必须忽略。

## 7. 建议的最小实现文件清单（尚未创建）

优先全部新增，不改网络：

```text
phyg/__init__.py
phyg/physical_degradation.py
phyg/gamma_replay.py
phyg/development_dataset.py
phyg/checkpoint.py
phyg/metrics.py
phyg/provenance.py
configs/phyg/protocols/lolv2_synthetic_dev_split_seed20260726.txt
configs/phyg/base_seed1234.yml
configs/phyg/identity_seed1234.yml
configs/phyg/gamma_replay_seed1234.yml
configs/phyg/physical_exposure_seed1234.yml
configs/phyg/physical_exposure_noise_seed1234.yml
configs/phyg/physical_exposure_color_seed1234.yml
configs/phyg/physical_full_seed1234.yml
scripts/phyg/train.py
scripts/phyg/validate_development.py
scripts/phyg/preflight.py
scripts/phyg/verify_budget_and_init.py
scripts/phyg/test_resume_equivalence.py
scripts/phyg/run_seed1234.sh
```

另有两项受控修改：

- `.gitignore`：修复源码误伤并补齐输出类型；
- 把缺失的官方 `basicsr/data/**` 加入开发分支跟踪。此项是仓库完整性修复，不修改其内容
  语义；应单独提交并记录来源/哈希。

不建议修改 `basicsr/train.py`、`image_restoration_model.py` 或
`RetinexFormer_arch.py`。独立入口可复用 `define_network`、官方 L1 与 metric primitives，
同时满足完整 checkpoint 和路径守卫。若后续证明独立入口无法保持某项作者语义，再提出
最小补丁并单列审计，未经确认不做。

## 8. 完整 checkpoint 合同

每个可信完整 checkpoint 至少包含：

```text
format_version
model
optimizer
scheduler
amp_scaler (如启用)
epoch
global_step
batch_in_epoch / sampler position
python_rng_state
numpy_rng_state
torch_cpu_rng_state
torch_cuda_rng_state_all
dataloader_generator_state
augmentation_rng_state
physical_parameter_rng_state
physical_noise_generator_state
config (解析后的完整副本)
config_sha256
split_raw_sha256
split_canonical_sha256
initialization_checkpoint_sha256
git_commit
best_validation_metric / best_epoch / best_global_step
test_access: NONE
```

保存需临时文件 + 原子替换，避免半写 checkpoint。resume 必须在创建下一 batch 前恢复所有
状态，并以 integration test 验证“连续 N step”与“中断后恢复”模型、optimizer、
scheduler、batch 文件名、augmentation 参数逐步一致。

PyTorch 2.6：

- 对本项目自己保存并经 SHA-256 验证的完整 checkpoint：
  `torch.load(path, map_location=..., weights_only=False)`；
- 对只含 tensor state dict 的官方初始化权重可优先使用安全 weights-only 路径，但需要先
  静态检查其实际顶层格式；本阶段未加载；
- 不能用裸 `except:` 猜 checkpoint 格式，官方评测入口当前在
  `Enhancement/test_from_dataset.py:103-111` 这样做，新增入口不可复用该模式。

## 9. 第一阶段实验协议与执行顺序

### 9.1 实施/预检顺序

1. 修复开发分支仓库完整性：跟踪 `basicsr/data/`，修正 `.gitignore`；验证冻结分支/标签
   未移动。
2. 冻结并复制 manifest、Physical、Gamma；记录源与目标 SHA；写纯 CPU 单元等价测试。
3. 冻结共享 Retinexformer base config：官方 Synthetic 初始化、810/90、seed=1234、
   crop/batch、optimizer、scheduler、loss、clip、validation、step 数、checkpoint 规则。
4. 实现路径守卫、完整 checkpoint、PyTorch 2.6 加载和 exact-resume 测试。
5. 只做不访问数据的静态 config/budget/init-path 检查；随后经授权才可对 Synthetic Train
   做 preflight。任何 Test/Eval 路径匹配均失败退出。
6. 经授权后，先记录官方初始化 SHA，运行 Frozen Official 的 **Development Validation
   only**（无训练）。
7. Identity Replay seed=1234。
8. Gamma Replay seed=1234。
9. Physical 四种模式 seed=1234。核心比较预注册为 `full` 对 Gamma；三个消融只解释机制，
   不用于根据真实 Test 调参。
10. 用同一冻结 90 对选择 checkpoint，并先判断 Development 证据是否支持进入正式 Test。
    方法、checkpoint 规则、真实域 Test 清单全部冻结后，才允许一次正式 Test；不使用
    GT mean。
11. 只有 Physical 相对 Gamma 在预注册真实域指标获得可信改善后，才新增 seed 2024、3407。

注意：用户给出的科学问题要求所有微调臂从同一个**官方 LOLv2-Synthetic 权重**初始化。
这与旧 CIDNet 报告中“Gamma replay/Physical 从 Gamma best 初始化”的历史做法
（`C:/Users/hxy/Desktop/PhyG-PEFT/PHYSICAL_MODULE_IMPLEMENTATION.md:39-50`）不同。
Retinexformer 项目必须服从本次新协议：Identity/Gamma/Physical 都 strict model-only
加载完全相同的官方 `LOL_v2_synthetic.pth`，各自新建相同 optimizer/scheduler。

### 9.2 预算建议与待冻结点

旧冻结 Physical 微调预算是 60 epochs、lr `1e-5`、batch 8、crop 256、每 10 epoch
validation：`C:/Users/hxy/Desktop/PhyG-PEFT/configs/cidnet_physical_full_seed1234.json:6-18,86-101`。
但 Retinexformer 官方 Synthetic 使用 crop 128、Adam lr `2e-4` 和按 step 的 cyclic cosine
（`Options/RetinexFormer_LOL_v2_synthetic.yml:33-37,76-97`）。

建议遵循“跨骨干而非复制 CIDNet”的原则：

- 网络、loss、optimizer 类型与官方 Retinexformer 保持一致；
- 初始化统一为官方 Synthetic；
- Replay 微调预算统一为 60 epochs；
- lr、crop size、scheduler 曲线必须在运行任何真实 Test 前一次性冻结，且三臂完全相同；
- 不应因某个 Retinexformer Test 结果再调整。

这里存在需要项目负责人确认的协议选择：是采用旧 Physical 微调的 `lr=1e-5/crop=256`
以强化跨骨干统一，还是采用 Retinexformer 官方 `crop=128` 并为 60 epoch 预注册一个共享
微调 lr/scheduler。静态审计不能科学地替用户决定；在确认前不得训练。

若采用 810、batch 8、`drop_last=True`、60 epochs，则每臂约：

```text
floor(810 / 8) = 101 optimizer steps/epoch
101 * 60 = 6,060 optimizer steps
Identity + Gamma + Physical-full = 18,180 optimizer steps
Frozen Official = 0 optimizer steps
```

四个 Physical 消融全跑则总训练臂为 Identity + Gamma + 4 Physical，即 36,360 steps。
必须把 `global_step` 而非“epoch 文案”作为预算断言；各臂最终 step 必须完全相同。

### 9.3 预计训练时间

当前审计机不是目标 Ubuntu 4090，且未获准训练/benchmark，因此不能给出经过测量的绝对
时间。建议服务器正式训练前只在 Synthetic Development Train 做经授权的 100-step
短 benchmark（不做 validation、不读 Test），按以下公式写入 preflight 报告：

```text
训练臂预计时间 =
  median_seconds_per_step * 6,060
  + 6 次 Development Validation 实测时间
```

在 4090、单卡、batch 8、128/256 patch 的常见量级下，仅作为资源排期而非承诺，预估
每个 60-epoch 训练臂约 0.5–2 小时；核心三训练臂约 1.5–6 小时；含四 Physical 消融约
3–12 小时。Physical 逐样本 Poisson 可能成为额外开销，必须用同一 100-step benchmark
分别测 Gamma/Physical 后更新估计。没有实测前，不得把该区间写成实验耗时结论。

## 10. 明确风险与阻断条件

| 级别 | 风险 | 证据 | 处置 |
|---|---|---|---|
| 阻断 | 冻结提交未跟踪 `basicsr/data/` | `.gitignore:2`；`basicsr/train.py:12-14` | 仅开发分支修复完整性，标签不动 |
| 阻断 | 官方 YAML 的 val 是 Synthetic Test | `Options/RetinexFormer_LOL_v2_synthetic.yml:43-49` | PhyG config 不继承 val，路径守卫拒绝 Test/Eval |
| 阻断 | 官方 resume 不保存完整 RNG/provenance | `basicsr/models/base_model.py:311-364` | 新增完整原子 checkpoint 与 exact-resume test |
| 高 | PyTorch 2.6 完整 checkpoint 默认加载语义变化 | `basicsr/train.py:168-172` 未显式参数 | 可信完整 checkpoint 显式 `weights_only=False` |
| 高 | 官方 Synthetic config 的 300k 注释/段预算与 150k total 不一致 | YAML `:26-38,67-80` | 共享 config 以显式 global step 为真值 |
| 高 | 旧 CIDNet 初始化规则与本次同官方初始化规则不同 | 旧报告 `:39-50` | 本项目三臂统一 strict-load 官方权重 |
| 高 | Gamma 具体实现尚未在本报告逐行定位 | 目前只定位配置 `:26-30` | 实现前继续审计正式 Gamma 源码，禁止重写猜测 |
| 高 | crop/lr/scheduler 跨项目选择尚未冻结 | 官方 YAML 与旧 Physical config 不同 | 负责人确认一次后锁配置 hash |
| 中 | 官方 mixup 自带 identity 随机性会与 replay identity 混淆 | `image_restoration_model.py:30-58,158-165` | 明确预注册是否三臂共同保留；不可只对某臂开关 |
| 中 | cuDNN benchmark=True 且 deterministic=False | `basicsr/train.py:152-153` | 明确确定性策略并写 metadata；exact resume 做实测 |
| 中 | 官方 best 只按单一 current metric/PSNR且删旧文件 | `train.py:329-334`; model `:357-386` | 新增不可变 checkpoint 与预注册多指标字典序 |
| 中 | 官方独立评测允许 GT mean | `Enhancement/test_from_dataset.py:69,184-189,262-267` | PhyG 入口不提供该选项并记录 false |
| 中 | 官方 DataLoader 无显式 generator，worker prefetch 难 exact resume | `basicsr/data/__init__.py:88-127` | protocol-v1 workers=0、显式 generator 和 sampler position |
| 中 | 初始化权重 SHA 未知 | README `:418-433`，本阶段未加载 | 训练前计算并与所有臂配置/metadata绑定 |

## 11. 下一步建议（需确认后才执行）

建议下一阶段仅实现基础设施，不训练：

1. 在 `phyg-crossbackbone` 修复 `.gitignore` 与 `basicsr/data/` 跟踪完整性；
2. 移植并哈希锁定 manifest、Physical、Gamma；
3. 新增独立入口、共享 base config、完整 checkpoint、路径守卫与 resume 等价测试；
4. 冻结 Retinexformer 微调的 crop/lr/scheduler 与 mixup 策略；
5. 提交一份 implementation audit 和所有静态/CPU 测试结果后再次停下；
6. 只有再次授权，才对 Synthetic Train 做 preflight/短 benchmark；仍不访问 Test/Eval。

本报告不是训练授权。当前停止点：等待用户确认上述仓库完整性修复方式和共享微调超参数
冻结原则。
