# Retinexformer 跨骨干验证基础设施实施报告

实施日期：2026-07-30  
分支：`phyg-crossbackbone`  
状态：基础设施与静态/CPU 测试完成；未训练、未读取数据图像、未加载官方权重、
未访问任何官方 Test/Eval。  
`official_test_participation=NONE`

## 1. Git 与官方源码完整性修复

初始冻结提交、`author-baseline` 与 `retinexformer-official-frozen` 均为：

```text
e8086f9d6e4badf0eebf48825264a785d3aa541e
```

冻结标签初始 `.gitignore` 中未锚定的 `data/` 错误忽略了官方源码目录
`basicsr/data/`，导致冻结标签 tracked tree 遗漏该目录。开发分支只进行了源码完整性修复：

- `data/` → `/data/`；
- `datasets/` → `/datasets/`；
- 增加 `/tb_logger/`、`/outputs/`、`*.state`；
- 将本机已有 25 个 `basicsr/data/**` 文件原样加入 Git；
- 没有编辑、格式化或清理其中任何源码内容。

该修复已作为独立 commit：

```text
e69ca52 chore: restore ignored official data sources
```

25 个文件的逐文件 SHA-256 固化于：

```text
configs/phyg/protocols/basicsr_data_sha256.txt
```

冻结标签未移动、未重建；`author-baseline` 未修改。以下禁止文件相对冻结标签仍无 diff：

```text
basicsr/models/archs/RetinexFormer_arch.py
Enhancement/test_from_dataset.py
basicsr/test.py
```

## 2. 修改/新增文件

文档：

```text
RETINEXFORMER_CROSSBACKBONE_AUDIT.md
RETINEXFORMER_CROSSBACKBONE_IMPLEMENTATION.md
```

协议模块：

```text
phyg/__init__.py
phyg/checkpoint.py
phyg/development_dataset.py
phyg/gamma_replay.py
phyg/metrics.py
phyg/physical_degradation.py
phyg/protocol.py
phyg/provenance.py
```

配置与 provenance：

```text
configs/phyg/base_seed1234.yml
configs/phyg/identity_seed1234.yml
configs/phyg/gamma_replay_seed1234.yml
configs/phyg/physical_exposure_seed1234.yml
configs/phyg/physical_exposure_noise_seed1234.yml
configs/phyg/physical_exposure_color_seed1234.yml
configs/phyg/physical_full_seed1234.yml
configs/phyg/protocols/lolv2_synthetic_dev_split_seed20260726.txt
configs/phyg/protocols/basicsr_data_sha256.txt
configs/phyg/protocols/source_sha256.txt
```

入口与测试：

```text
scripts/phyg/train.py
scripts/phyg/validate_development.py
scripts/phyg/preflight.py
scripts/phyg/preflight_static.py
scripts/phyg/verify_budget_and_init.py
scripts/phyg/test_resume_equivalence.py
scripts/phyg/run_seed1234.sh
```

没有修改 Retinexformer 网络结构和官方 Test/Eval 入口。

## 3. 既有协议复制、SHA-256 与差异

| 内容 | 原文件 SHA-256 | 目标文件 SHA-256 | 逐字一致 |
|---|---|---|---|
| Physical | `133fda95e645390eaebda636d0db9b3ca01967b8163db30bc20ad75b415a5a2b` | `133fda95e645390eaebda636d0db9b3ca01967b8163db30bc20ad75b415a5a2b` | 是 |
| Development dataset | `45f055d27988b6c68891383ebbb6e31202c2e5d97b7451adcc72989f701574f9` | `45f055d27988b6c68891383ebbb6e31202c2e5d97b7451adcc72989f701574f9` | 是 |
| 统一 metrics | `8e35a8d85b2f0ed0a83d0ad7706845b95f3d2b0f3e41073eb555f85f12eebaf0` | `8e35a8d85b2f0ed0a83d0ad7706845b95f3d2b0f3e41073eb555f85f12eebaf0` | 是 |
| 810/90 manifest | `f80b8d5cebd3b8f6529b0670e91037320bf939dba6f16d8a5402861465f543d6` | `f80b8d5cebd3b8f6529b0670e91037320bf939dba6f16d8a5402861465f543d6` | 是 |

源文件完整列表及 hash 固化于
`configs/phyg/protocols/source_sha256.txt`。

Manifest canonical SHA-256：

```text
e3e73abce5405d7fa4cbb7d1f16470dc924abf97c77208ba049da1613d0a944d
```

Gamma 的正式来源不是独立模块，而是
`C:/Users/hxy/Desktop/PhyG-PEFT/train.py:52-55`：

```text
gamma = random.randint(start_gamma, end_gamma) / 100.0
model(im1 ** gamma)
```

源 `train.py` SHA-256：

```text
ed9fd092fe9a11c429b15675b6bbf081670d4d36654b8ca97562674ee6fc3eee
```

目标 `phyg/gamma_replay.py` SHA-256（报告生成前）：

```text
980a2f9b7ddcf18ea146f084cad8ce7979b4fa2e629e28aa9bca0c0a1ba06337
```

Gamma 不能逐文件复制，因为来源逻辑嵌在 CIDNet 训练入口中。必要差异只有：

1. 将两行正式语义封装成 `GammaReplay`；
2. Python 参数 RNG 改为独立 `random.Random`；
3. 增加 `state_dict/load_state_dict` 以支持 exact resume；
4. 保持每 batch 一次整数闭区间 `[60,120]` 采样及 `LQ ** gamma` 完全不变。

`scripts/phyg/preflight.py` 使用相同 seed 的参考 `random.Random` 逐次比较 gamma 值和
输出 tensor，数值等价测试通过。

## 4. 冻结共同训练协议

共享 base config 为 `configs/phyg/base_seed1234.yml`：

| 项目 | 冻结值 |
|---|---|
| initialization | `pretrained_weights/LOL_v2_synthetic.pth` |
| load | strict model-only，官方只读加载使用 `weights_only=True` |
| seed | 1234 |
| split | 固定 810/90 manifest |
| batch/crop | 8 / 128 |
| epochs | 60 |
| steps | 101/epoch，6060 total |
| loader | `drop_last=True`，`num_workers=0` |
| optimizer | Adam，betas `[0.9,0.999]` |
| lr | `1e-5` |
| scheduler | 单周期 `CosineAnnealingLR(T_max=6060, eta_min=1e-7)`，每 optimizer step 更新 |
| loss | 官方 Retinexformer `L1Loss(weight=1, reduction=mean)` |
| gradient clip | global norm 0.01，backward/unscale 后 |
| mixup | 所有训练臂共同启用，beta=1.2，含 identity |
| validation | 每 10 epoch |
| metrics | 8-bit RGB、PSNR、RGB SSIM 11×11/sigma=1.5、AlexNet LPIPS |
| GT mean | 禁止 |
| best | max PSNR → max SSIM → min LPIPS |

六个训练配置为 Identity、Gamma 和 Physical 的 exposure、exposure_noise、
exposure_color、full。共享协议检查会移除 `experiment/replay` 后逐字段比较，其余字段
必须完全一致。送入 `net_g` 前的处理顺序固定为：

```text
manifest pair -> paired crop/flip -> shared mixup -> LQ-only replay -> unchanged net_g
```

Identity/Gamma/Physical 唯一变化是 LQ-only replay。

## 5. Development 与路径隔离

- Manifest 解析结果为 Train 810、Validation 90；
- 两集合无交集；
- raw/canonical SHA 均强制核验；
- `audit_pairs` 只接受父目录名为 `Synthetic`、自身名为 `Train` 的根；
- `phyg.provenance.assert_development_root` 额外拒绝路径组件中的 `Test` 或 `Eval`；
- Validation 只从 manifest 90 文件名构建；
- 没有继承官方 Synthetic YAML 中指向 Test 的 val 段；
- 独立 validation 入口不提供 GT mean 选项；
- Frozen Official Development Validation 可显式选择
  `--official-initialization`，但本阶段未执行。

本阶段测试只解析 manifest 文本，没有枚举或读取任何数据目录。

## 6. Checkpoint 与 exact resume

`phyg/checkpoint.py` 采用临时文件加 `os.replace` 原子保存。完整 checkpoint 包含：

- model、optimizer、scheduler、AMP scaler；
- epoch、batch_in_epoch、global_step；
- epoch order、sampler position；
- Python、NumPy、Torch CPU、CUDA RNG；
- DataLoader generator、augmentation RNG；
- Gamma/Physical replay RNG；
- Physical noise generator；
- 完整解析配置及 config SHA-256；
- split raw/canonical SHA-256；
- initialization checkpoint SHA-256；
- Git commit；
- best metric、epoch、step；
- `official_test_participation=NONE`。

训练入口每 100 step 保存可中途恢复的 `latest.pth`，并在 epoch/validation 节点保存。
恢复前强制核验 config、两个 split hash、初始化 checkpoint hash 和 Test participation。

本项目自己生成且可信的完整 checkpoint 通过：

```python
torch.load(path, map_location=..., weights_only=False)
```

显式加载。官方 model-only 初始化走单独的安全路径，不与完整 resume loader 混用。

Exact-resume CPU 测试覆盖：

- tiny model；
- Adam 参数和内部状态；
- step-level CosineAnnealingLR；
- L1 backward、clip=0.01；
- Python/NumPy/Torch RNG；
- Gamma 独立参数 RNG；
- DataLoader generator state；
- model/optimizer/scheduler；
- checkpoint 必需字段；
- PyTorch 2.6 `weights_only=False` 可信完整加载。

比较结果：

```text
连续 7 step == 连续 4 step + 保存/重建/恢复 + 3 step
```

逐 step loss、gamma、输入、target，以及最终 model、optimizer、scheduler 状态完全相等。
正式 Retinexformer/CUDA/DataLoader 图像训练的恢复等价仍需在后续获准的服务器 preflight
阶段验证；本阶段没有运行正式模型训练。

## 7. 测试环境、命令与结果

本机初始没有 Python 训练环境。测试依赖只安装到系统临时目录，不进入仓库：

```text
Python 3.12.13
PyTorch 2.6.0+cpu
torchvision 0.21.0+cpu
PyYAML 6.x
```

没有安装 CUDA，没有加载权重或数据。

### 7.1 Compile

```text
python -m compileall -q phyg scripts/phyg
PASS
```

### 7.2 无 PyTorch 的标准库静态检查

```text
python scripts/phyg/preflight_static.py
PASS: frozen refs and forbidden-file integrity
PASS: 25 official basicsr/data files are tracked
PASS: four exact source copies
PASS: frozen split 810/90, disjoint, raw/canonical hashes
PASS: shared base budget/init literals and six arm configs
official_test_participation=NONE
```

### 7.3 PyTorch 2.6 CPU 协议测试

```text
python scripts/phyg/preflight.py
PASS: author integrity
PASS: repository completeness
PASS: exact source copies and Physical/Gamma CPU equivalence
PASS: split count/nesting/hashes and Test/Eval path guard
official_test_participation=NONE
```

### 7.4 Budget/init

```text
python scripts/phyg/verify_budget_and_init.py
PASS: 6 arms share initialization and 6060-step budget
Only experiment/replay fields differ
official_test_participation=NONE
```

该检查只验证初始化**路径字符串**一致，未打开权重文件。

### 7.5 Exact resume

```text
python scripts/phyg/test_resume_equivalence.py
PASS: continuous 7 steps == 4 steps + trusted restore + 3 steps
PASS: required checkpoint fields and weights_only=False load path
official_test_participation=NONE
```

### 7.6 Import

```text
import scripts.phyg.train
PASS project imports with 2.6.0+cpu
```

## 8. 尚未完成事项与下一停止点

以下事项有意未执行：

1. 官方 `pretrained_weights/LOL_v2_synthetic.pth` 是否存在、实际顶层格式及 SHA-256；
2. Synthetic Train 的 900 对实际文件名/嵌套核验；
3. Retinexformer 官方权重 strict model-only load；
4. 真实 Retinexformer 单 step、100-step benchmark；
5. 正式模型 CUDA exact-resume；
6. Frozen Official 的 Development Validation；
7. 任何 Identity/Gamma/Physical 正式训练；
8. 任何官方 Test/Eval；
9. 正式 Test checkpoint 选择与一次性评测。

这些都需要下一次明确授权。当前只完成基础设施、静态检查和无数据 tiny-CPU 测试。

最终声明：

```text
official_test_participation=NONE
official_weight_loaded=NO
dataset_images_read=NO
formal_training_started=NO
```

## 9. 2026-07-31 可移植预检、真实栈恢复与 benchmark 增补

### 9.1 仓库自包含 SHA 校验

`preflight.py` 和 `preflight_static.py` 已去除
`C:/Users/hxy/Desktop/PhyG-PEFT` 硬编码。两者只读取仓库内
`configs/phyg/protocols/source_sha256.txt`，校验：

```text
phyg/physical_degradation.py
phyg/development_dataset.py
phyg/metrics.py
configs/phyg/protocols/lolv2_synthetic_dev_split_seed20260726.txt
```

清单固定四个 SHA-256，文件缺失或内容变化均立即失败。`/data/` 与 `/datasets/`
继续被 `.gitignore` 忽略；权重、checkpoint、日志和 `/outputs/` 忽略规则保持。

### 9.2 官方初始化验证

文件：

```text
pretrained_weights/LOL_v2_synthetic.pth
```

本机检查结果：

```text
exists=true
SHA-256=2f793853c6b1ba0b4a3ff12b7b10f0f2c851eaefda1006437eb3f35229838c4a
top-level keys={'params'}
parameter_tensors=122
parameter_values=1605701
strict=True
renamed=false
dropped=false
repaired=false
```

`phyg/initialization.py` 在加载前验证文件 SHA，要求顶层严格为
`{'params': state_dict}`，逐项比较 key、shape 和 dtype，最后调用
`model.load_state_dict(state, strict=True)`。不包含自动丢弃、添加 `module.`、重命名、
shape 修补或 fallback 分支。

### 9.3 DataLoader RNG 修复

PyTorch 在 `iter(DataLoader)` 时会生成 base seed；未提供 generator 会额外消耗全局
Torch RNG。新增 `isolated_loader_iterator`：

1. 使用独立 `iteration_generator`；
2. 在创建 iterator 前保存 generator state；
3. 创建 iterator 后立即恢复该 state；
4. checkpoint 独立保存/恢复 epoch-order generator 和 iterator generator。

`test_dataloader_rng_isolation.py` 验证重建 iterator 前后全局 Torch RNG 与独立
generator 均逐字相等，且 sample order 不变。

### 9.4 真实训练栈断点等价测试

新增 `scripts/phyg/test_real_stack_resume.py`，覆盖：

- 真实 `RetinexFormer`；
- 官方 strict initialization；
- 实际 Development Train 810；
- 128 crop、flip、shared Mixup；
- Gamma Replay 与 Physical Full；
- Adam、L1、clip=0.01、step-level cosine scheduler；
- epoch 中途完整 checkpoint；
- model、optimizer、scheduler、global_step；
- 下一批文件名、crop/flip/Mixup 后 tensor 和下一批 Gamma/Physical 参数。

本机 CPU 补充执行结果：

```text
Gamma: continuous 2 steps == 1 step + restore + 1 step: PASS
Physical Full: continuous 2 steps == 1 step + restore + 1 step: PASS
next Gamma=0.67: identical
next Physical per-sample parameter list: identical
official_test_participation=NONE
```

CUDA 执行尝试使用 PyTorch `2.6.0+cu124`，但本机 GPU 是 RTX 5060 Laptop、
compute capability `sm_120`；PyTorch 2.6 wheel 只包含至 `sm_90`，在
`model.to("cuda")` 时失败：

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

这是本机 GPU/PyTorch 2.6 架构不兼容，不是 OOM、NaN、数据或 checkpoint 失败。
目标 Ubuntu RTX 4090 为 `sm_89`，在 PyTorch 2.6 支持范围内。CUDA exact-resume
仍必须在目标 4090 服务器执行后才可标记 PASS。

### 9.5 100-step benchmark

新增 `scripts/phyg/benchmark_100_steps.py`。它依次从同一官方权重初始化并运行：

```text
Identity 100 steps
Gamma Replay 100 steps
Physical Full 100 steps
```

输出每臂：

- completed steps；
- peak CUDA memory bytes/GiB；
- average synchronized step seconds；
- mean/final loss；
- NaN/OOM；
- initialization SHA。

并以：

```text
(Identity + Gamma + 4 × Physical Full) × 6060 steps
```

估算六组 60-epoch 训练步耗时，明确不包含每组六次 Development Validation。
结果写入 ignored 的 `outputs/phyg_benchmark_seed1234.json`。

由于上述 `sm_120`/PyTorch 2.6 阻断，本机未产生伪造或跨版本 benchmark 数字。
benchmark 必须在目标 4090 服务器运行。

`scripts/phyg/run_seed1234.sh` 现在只串联全部 preflight、权重 strict check、
RNG test、tiny resume、真实栈 CUDA resume 和三臂 benchmark；不会调用
`scripts/phyg/train.py`，因此不会自动启动六组正式训练。

本增补阶段最终状态：

```text
official_test_participation=NONE
official_weight_loaded_for_strict_check=YES
official_weight_used_for_formal_training=NO
development_train_read_for_resume_test=YES
official_test_or_eval_read=NO
formal_training_started=NO
cuda_exact_resume_on_target_4090=PENDING
benchmark_on_target_4090=PENDING
```

## 10. 2026-07-31 CUDA checkpoint RNG state 兼容修复

问题：对完整 checkpoint 使用 `map_location="cuda"` 时，checkpoint 内原本属于 CPU
generator 的 RNG tensors 也会被映射到 CUDA。直接调用
`torch.set_rng_state(cuda_tensor)` 会报告：

```text
TypeError: RNG state must be a torch.ByteTensor
```

修复严格限制在 RNG state，不改变完整 checkpoint 的加载位置：

- `torch_cpu_rng` 在恢复前显式转为 contiguous CPU `torch.uint8`；
- `torch_cuda_rng` 的每个 state 在传给 `torch.cuda.set_rng_state_all` 前做相同转换；
- DataLoader epoch-order generator state 做相同转换；
- DataLoader iterator generator state 做相同转换；
- Physical `noise_generator` state 在
  `PhysicalBatchAugmenter.load_state_dict` 内做相同转换。

model、optimizer、scheduler 和其他 tensor 仍按调用者的 `map_location` 加载到训练 GPU；
没有把整个 checkpoint 改成 CPU 加载。

`phyg/physical_degradation.py` 仅在 `load_state_dict` 增加 RNG tensor 的
device/dtype normalization，退化采样、参数、公式和 forward 数值路径均未改变。更新后的
仓库固定 SHA-256：

```text
e68b90cd55f94b99089da580c7076313c2348e08f1034314b55c06c203d32c13
```

测试增强：

- `test_resume_equivalence.py` 明确断言两个 DataLoader generator state；
- `test_real_stack_resume.py` 使用 `map_location=device` 加载完整 checkpoint；
- CUDA 模式断言 checkpoint RNG tensors 确实先被映射到 CUDA；
- 随后断言 CPU RNG、每卡 CUDA RNG、两个 DataLoader generator 和 Physical noise
  generator 均恢复为预期 CPU ByteTensor state；
- Gamma 与 Physical Full 都比较连续 4 step 和 2 step + restore + 2 step；
- 继续比较 model、optimizer、scheduler、global_step 和下一批退化参数。

目标 4090 CUDA 测试命令：

```text
python scripts/phyg/test_real_stack_resume.py --steps 4 --interrupt-step 2
```

本机使用同一真实网络、实际 Development Train 和完整 checkpoint 的 CPU 回归结果：

```text
Gamma: continuous 4 steps == 2 steps + restore + 2 steps: PASS
Physical Full: continuous 4 steps == 2 steps + restore + 2 steps: PASS
model/optimizer/scheduler/global_step/next degradation: bitwise identical
```

CUDA `map_location=cuda` 专属断言已经实现，但因本机 `sm_120` 不受 PyTorch 2.6
支持，仍需在目标 RTX 4090 执行后确认 CUDA RNG restore PASS。

本修复阶段不运行 benchmark、不启动正式训练、不访问官方 Test/Eval。

### 10.1 cuBLAS 确定性前置条件

目标 4090 首次执行真实栈测试时，cuBLAS GEMM 在
`torch.use_deterministic_algorithms(True)` 下要求进程启动前设置 workspace：

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

`test_real_stack_resume.py` 现已在导入 PyTorch 前使用 `os.environ.setdefault` 设置该值；
`run_seed1234.sh` 也在启动任何 Python 进程前显式 `export`。这只影响确定性测试的
cuBLAS workspace，不修改网络、优化器、退化协议或正式训练配置。
