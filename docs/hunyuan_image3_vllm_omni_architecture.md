# HunyuanImage-3.0 在 vLLM-Omni 中的架构与推理流程

本文最初基于 `pr-3297` 整理，当前已按 `main` 分支的 deploy 体系更新。重点覆盖：

- HunyuanImage-3.0 原始模型结构在代码中的组成。
- 适配 vLLM-Omni 后的 stage 化架构。
- 文生图、图生图、图生文、文生文的推理流程。
- 为什么模型里同时有 VAE 和 ViT。
- 现有 HunyuanImage3 配置的含义与差异。

当前 `main` 路径里，离线/部署优先使用：

- `vllm_omni/deploy/hunyuan_image_3_moe.yaml`：AR + DiT
- `vllm_omni/deploy/hunyuan_image3_ar.yaml`：AR only
- `vllm_omni/deploy/hunyuan_image3_dit.yaml`：DiT only

文中对 `model_executor/stage_configs/hunyuan_image3*.yaml` 的说明保留为实现背景，实际使用时应优先看 deploy YAML。

相关实现入口：

- AR/vLLM 侧模型：`vllm_omni/model_executor/models/hunyuan_image3/hunyuan_image3.py`
- Diffusion/DiT 侧 pipeline：`vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py`
- DiT 主体与 Diffusers pipeline：`vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py`
- AR -> Diffusion 衔接：`vllm_omni/model_executor/stage_input_processors/hunyuan_image3.py`
- stage config：`vllm_omni/model_executor/stage_configs/hunyuan_image3*.yaml`
- 离线示例：`examples/offline_inference/hunyuan_image3/end2end.py`

## 1. HunyuanImage-3.0 的模型结构

HunyuanImage-3.0-Instruct 在这个适配里不是一个单纯的 DiT 文生图模型，而是一个统一的多模态自回归模型加扩散生成模型：

```text
文本 token / 特殊图像 token
        |
        v
Hunyuan AR/MM Decoder
        |
        +--> 文本输出：T2T / I2T
        |
        +--> 图像生成控制序列、CoT、recaption、KV cache
                 |
                 v
          HunyuanImage3 Diffusion Pipeline
                 |
                 v
        FlowMatch Euler denoising + HunyuanImage3Model/DiT
                 |
                 v
              VAE decode
                 |
                 v
                Image
```

代码里可以分成几类组件。

### 1.1 AR/MM Decoder

AR 侧入口是 `HunyuanImage3ForConditionalGeneration`。它包装了：

- `HunyuanModel`：vLLM 侧的 causal decoder 主干。
- `lm_head`：用于普通 token 预测。
- 自定义 sampler：在不同任务阶段强制或限制特殊 token。
- 自定义 interleaved 2D RoPE：替换 vLLM 标准 mRoPE，以匹配原始 HunyuanImage3 的二维位置编码布局。
- 多模态输入 embedding 逻辑：把输入图片转成 VAE token embedding 和 ViT token embedding 后填入 `<img>` 占位区域。

AR 侧主要负责：

- 文本理解和文本生成。
- 读入条件图像。
- 在图像生成任务中生成 `<think>`、`<recaption>`、`<answer>`、`<boi>`、`<img_size_*>`、`<img_ratio_*>` 等结构化 token。
- 为下游 DiT 提供条件上下文，部分配置还会通过 `omni_kv_config` 发送 KV cache。

### 1.2 Diffusion/DiT 侧

Diffusion 侧入口是 `HunyuanImage3Pipeline`。初始化时会加载：

- `HunyuanImage3Model`：用于 denoising 的 transformer 主体。
- `AutoencoderKLConv3D`：VAE，负责 image <-> latent。
- `TokenizerWrapper`：构造 HunyuanImage3 的 pretrain/chat 模板。
- `Siglip2VisionModel` 和 `LightProjector`：用于条件图像的 ViT 语义编码。
- `UNetDown`：把 VAE latent patch 化并投影到 hidden size。
- `UNetUp`：把 transformer 输出还原成 VAE latent 通道。
- `TimestepEmbedder`：把扩散时间步注入图像 token。
- `FlowMatchEulerDiscreteScheduler`：扩散采样调度器，`shift` 来自 `generation_config.flow_shift`。

虽然文件名里有 `Text2ImagePipeline`，实际实现也支持条件图像路径：当请求带有条件图像时，会同时走 VAE encode 和 ViT encode，然后把两类图像 embedding 写入 transformer 输入序列。

### 1.3 VAE

VAE 负责像素空间和 latent 空间之间的压缩/还原：

- 条件图像输入时，VAE encode 得到低维 latent，再经过 `UNetDown` 变成 transformer token embedding。
- 文生图/图生图输出时，denoising loop 得到最终 latents，然后按 `scaling_factor` / `shift_factor` 反变换，再 VAE decode 成 PIL image。
- 配置里的 `vae_downsample_factor` 和 `vae["latent_channels"]` 决定 latent 尺寸和通道数。

### 1.4 ViT

代码里有 ViT 是因为 HunyuanImage3 对“输入条件图像”使用双路表示：

```text
条件图像
   |
   +--> VAE encoder -> latent tokens
   |
   +--> SigLIP2 ViT -> semantic/visual tokens -> LightProjector
```

两路用途不同：

- VAE tokens 保留空间、颜色、结构等可重建信息，适合图像编辑和图像条件生成。
- ViT tokens 提供更抽象的视觉语义信息，适合理解图片内容、指令跟随和与文本对齐。

在 AR 侧，`HunyuanImage3Processor` 会把单个 `<img>` 替换成一段结构化 token：

```text
<boi>
<img_size_*>
<img_ratio_*>
<img> * timestep_token_num
<img> * vae_token_num
<joint_img_sep>
<img> * vit_token_num
<eoi>
```

随后 `embed_multimodal()` 会把这些 `<img>` 位置替换成真实 embedding：先 timestep embedding，再 VAE token embedding，再 ViT token embedding。Diffusion 侧也有类似逻辑，`instantiate_vae_image_tokens()` 和 `instantiate_vit_image_tokens()` 分别写入 VAE/ViT 条件图像 token。

因此，ViT 不是用来生成最终图像的 decoder；它是条件图像的视觉编码器。真正输出图像的是 denoising transformer 预测 latent，再由 VAE decode。

### 1.5 AR 侧 ViT 独立 DP

本仓库当前任务里确认的是 AR 侧 ViT，而不是 diffusion pipeline 里的 ViT。HunyuanImage3 AR 实现已经接入 vLLM 的现有 ViT data-parallel 开关，不需要新增 HunyuanImage3 专用并行字段：

- `vllm_omni/model_executor/models/hunyuan_image3/siglip2.py` 引入 `vllm.model_executor.models.vision.is_vit_use_data_parallel`。
- `Siglip2Attention` 和 `Siglip2MLP` 在 `is_vit_use_data_parallel()` 为 true 时，对 `QKVParallelLinear`、`RowParallelLinear`、`ColumnParallelLinear` 传入 `disable_tp=True`。
- 同时 `Siglip2Attention.tp_size` 会设为 `1`，表示 ViT 不再跟随 AR decoder 的 tensor parallel 切权重。

因此推荐的配置方式是在包含 AR stage 的 HunyuanImage3 YAML 里使用现有 vLLM 字段：

```yaml
engine_args:
  model_stage: AR
  tensor_parallel_size: 4
  mm_encoder_tp_mode: data
```

这个语义是：

- AR decoder 仍然使用 `tensor_parallel_size`、`pipeline_parallel_size` 等原有并行方式。
- ViT/multimodal encoder 使用 vLLM 现有的 data-parallel 路径。
- ViT DP 与 AR 主干 TP 隔离，不新增新的 HunyuanImage3 配置字段。
- DiT/diffusion stage 的 `parallel_config` 不受影响，仍可独立配置 TP、EP、SP、CFG parallel 等。

## 2. 适配 vLLM-Omni 后的架构

vLLM-Omni 把 HunyuanImage3 拆成 stage。不同任务可以只跑 AR，也可以跑 AR + Diffusion。

### 2.1 模型注册

同一个 `model_arch` 名称会在两个 registry 中指向不同运行时：

- `vllm_omni/model_executor/models/registry.py`
  - `HunyuanImage3ForCausalMM` -> `HunyuanImage3ForConditionalGeneration`
  - 用于 AR/LLM stage。
- `vllm_omni/diffusion/registry.py`
  - `HunyuanImage3ForCausalMM` -> `HunyuanImage3Pipeline`
  - 用于 diffusion stage。

因此 YAML 里可以在 AR 和 diffusion stage 使用同一个 `model_arch: HunyuanImage3ForCausalMM`，实际加载哪个类由 stage 类型和 engine 决定。

### 2.2 Stage 拆分

常见两类拓扑：

```text
T2T / I2T:
    Stage 0: llm / AR -> text

T2I / IT2I:
    Stage 0: llm / AR -> latent/control tokens 或 KV cache
    Stage 1: diffusion / DiT + VAE -> image
```

AR stage 使用：

- `stage_type: llm`
- `model_stage: AR`
- `worker_cls: vllm_omni.worker.gpu_ar_worker.GPUARWorker`
- `scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler`
- `engine_output_type: latent` 时表示 AR 结果要给下游图像生成使用。

Diffusion stage 使用：

- `stage_type: diffusion`
- `model_stage: dit` 或 `model_stage: diffusion`
- `distributed_executor_backend: mp`
- `parallel_config` 配置 TP、EP、SP、CFG parallel 等扩散并行策略。
- `omni_kv_config.need_recv_cache: true` 表示接收上游 AR KV cache。

### 2.3 AR 到 Diffusion 的衔接

`hunyuan_image3_it2i.yaml` 中显式使用：

```yaml
custom_process_input_func: vllm_omni.model_executor.stage_input_processors.hunyuan_image3.ar2diffusion
```

`ar2diffusion()` 做的事比较直接：

1. 读取上游 AR stage 的 `engine_outputs`。
2. 提取 `output.cumulative_token_ids` 和生成文本。
3. 从原始 prompt 里取 `height`、`width`、`prompt`、`multi_modal_data`。
4. 组装 diffusion stage 可消费的输入 dict。
5. 把原始图片转发给 diffusion stage，供图生图/编辑任务继续作为条件图像。
6. 转发 `seed`、`num_inference_steps`、`guidance_scale`、`negative_prompt` 等参数。

在带 KV 复用的配置中，stage edge 和 `omni_kv_config` 还会负责跨 stage 传递 KV cache：

```text
AR prefill finished
        |
        v
send KV cache
        |
        v
Diffusion stage receive KV cache
        |
        v
DiT denoising reuses AR context
```

## 3. 推理流程

### 3.1 Text-to-Image

示例脚本把 `--modality text2img` 映射到 `t2i_think`，默认 prompt 模板类似：

```text
<|startoftext|>{system_prompt}<think>{user_prompt}
```

流程：

1. `end2end.py` 构造 prompt dict，`modalities=["image"]`。
2. `Omni` 根据 stage config 初始化 stage。
3. AR stage 生成思考、recaption、图像尺寸/比例等结构 token。
4. 如果是两 stage/KV 配置，AR 输出和/或 KV cache 被传给 diffusion stage。
5. Diffusion stage 的 `HunyuanImage3Pipeline.forward()` 读取 sampling params：
   - `num_inference_steps`
   - `guidance_scale`
   - `seed`
   - `height` / `width`
6. `prepare_model_inputs()` 构造 diffusion 输入 token、2D RoPE、timestep token、CFG 分支。
7. `HunyuanImage3Text2ImagePipeline.__call__()` 采样初始 Gaussian latents。
8. 对每个 timestep：
   - 将当前 latents 作为 image tokens 写入输入序列。
   - 调用 `forward_call()` 得到 `diffusion_prediction`。
   - 当 `guidance_scale > 1` 时执行 classifier-free guidance。
   - 调用 `FlowMatchEulerDiscreteScheduler.step()` 更新 latents。
   - 更新 KV cache / position ids / attention mask。
9. 最终 latents 经 VAE decode，返回 image。

### 3.2 Image-to-Image / Image Editing

流程：

1. prompt 中包含 `<img>`，`multi_modal_data` 携带 PIL image。
2. AR stage 的 multimodal processor 对输入图像做双路预处理：
   - SigLIP2 processor -> ViT patch tokens。
   - resize/crop + normalize -> VAE pixel values。
3. `<img>` 占位符被扩展成包含 size、ratio、VAE token、ViT token 的结构化片段。
4. `embed_multimodal()` 用 VAE/ViT embedding 替换这些 token 位置。
5. AR stage 生成编辑相关的 CoT/recaption/控制 token。
6. `ar2diffusion()` 把原始图片、AR token、尺寸和采样参数传给 Diffusion stage。
7. Diffusion stage 再次把条件图像编码为 VAE tokens 和 ViT tokens，放进 denoising transformer 的条件序列。
8. Denoising + VAE decode 得到编辑后的图片。

### 3.3 Image-to-Text

流程：

1. prompt 中包含 `<img>`，`modalities=["text"]`。
2. AR stage 通过 ViT + VAE 双路 embedding 读取图片。
3. 自定义 sampler 阻止生成图像相关特殊 token，例如 `<boi>`、`<eoi>`、`<img_size_*>`、`<img_ratio_*>`。
4. 只输出文本，通常在 `</answer>` 或 EOS 处停止。

### 3.4 Text-to-Text

流程：

1. prompt 只有文本，`requires_multimodal_data: false`。
2. 只跑 AR stage。
3. 使用 comprehension mode 的采样约束，输出文本。

## 4. Diffusion loop 的关键细节

### 4.1 CFG

`guidance_scale > 1.0` 时启用 classifier-free guidance：

- 非 CFG parallel：把 batch 扩成 conditioned/unconditioned 两份，前向后 `chunk(2)`。
- CFG parallel：当 `cfg_parallel_world_size == 2` 时，两个 rank 分别跑 conditioned/unconditioned，再 `all_gather` 合并。

合并公式在 `ClassifierFreeGuidance` 中：

```text
pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
```

### 4.2 Attention backend

`HunyuanImage3Pipeline` 初始化时强制：

```python
os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"
```

原因是该模型需要处理混合 causal attention 和 full attention，当前实现只声明支持 Torch SDPA。

### 4.3 2D RoPE

HunyuanImage3 的二维 RoPE 是 interleaved 形式：y/x 位置在频率维交错排列。vLLM 默认 mRoPE 使用连续频段分配，不匹配原始权重。因此 AR 侧显式替换 rotary embedding；Diffusion 侧则通过 `build_batch_2d_rope()` 构造对应的 cos/sin。

### 4.4 MoE 与并行

`HunyuanImage3Model` 支持 MoE：

- `_is_moe(config)` 判断 `num_experts`。
- `HunyuanFusedMoE` 负责专家权重映射。
- Diffusion stage 的 `parallel_config.enable_expert_parallel: true` 可以启用 EP。
- 也支持 TP、SP、CFG parallel、VAE patch parallel、HSDP 等通用 diffusion 并行配置字段。

## 5. 现有配置文件说明

当前存在这些 HunyuanImage3 YAML：

```text
vllm_omni/model_executor/stage_configs/
  hunyuan_image3_i2t.yaml
  hunyuan_image3_it2i.yaml
  hunyuan_image3_moe.yaml
  hunyuan_image3_moe_dit_2gpu_fp8.yaml
  hunyuan_image3_t2i.yaml
  hunyuan_image3_t2i_2gpu.yaml
  hunyuan_image3_t2t.yaml
```

### 5.1 `hunyuan_image3_i2t.yaml`

用途：图生文 / 图像理解。

结构：

- 单 stage：`stage_type: llm`
- `devices: "0,1,2,3"`
- `requires_multimodal_data: true`
- `model_stage: AR`
- `model_arch: HunyuanImage3ForCausalMM`
- `tensor_parallel_size: 4`
- `is_comprehension: true`
- `final_output_type: text`

采样：

- `temperature: 0.0`
- `top_p: 0.95`
- `top_k: 1024`
- `max_tokens: 2048`
- `stop_token_ids: [127957, 128026]`
- `detokenize: True`

适合只需要图片理解，不需要 Diffusion stage 的场景。

### 5.2 `hunyuan_image3_t2t.yaml`

用途：纯文本问答/生成。

结构和 I2T 类似，但：

- `requires_multimodal_data: false`
- 不需要输入图像。

同样是单 AR stage，输出 text。

### 5.3 `hunyuan_image3_it2i.yaml`

用途：图文到图 / 图片编辑。

结构：

```text
Stage 0: llm / AR       devices 0,1,2,3
Stage 1: diffusion/DiT  devices 4,5,6,7
```

Stage 0：

- `requires_multimodal_data: true`
- `engine_output_type: latent`
- `is_comprehension: false`
- `final_output: false`
- 生成结果由下游 diffusion 使用。

Stage 1：

- `model_stage: dit`
- `distributed_executor_backend: mp`
- `parallel_config.tensor_parallel_size: 4`
- `parallel_config.enable_expert_parallel: true`
- `omni_kv_config.need_recv_cache: true`
- `engine_input_source: [0]`
- `custom_process_input_func: ...hunyuan_image3.ar2diffusion`
- `final_output_type: image`

默认 diffusion 参数：

- `num_inference_steps: 50`
- `guidance_scale: 2.5`

这是最完整的图像编辑路径：AR 读图和指令，Diffusion 生成最终图片。

### 5.4 `hunyuan_image3_moe.yaml`

用途：带 AR -> DiT KV reuse 的 8 卡 MoE 配置。

结构：

```text
Stage 0: llm / AR       devices 0,1,2,3
Stage 1: diffusion      devices 4,5,6,7
Runtime edge: 0 -> 1
```

特点：

- Stage 0 设置 `omni_kv_config.need_send_cache: true`。
- `kv_transfer_criteria.type: prefill_finished`，AR prefill 完成后发送 KV cache。
- Stage 1 设置 `omni_kv_config.need_recv_cache: true`。
- runtime edge 设置 `window_size: -1`，表示上游完成后再触发下游。

注意：该配置的 Stage 0 同时写了 `final_output: true` 和 `final_output_type: text`，Stage 1 也写了 `final_output: true` 和 `final_output_type: image`。从注释看它用于 text-to-image + image-to-text 的混合能力，但具体请求路径要看 Orchestrator 如何按 modality 收集 final output。

### 5.5 `hunyuan_image3_t2i.yaml`

用途：当前文件注释写的是 HunyuanImage-3.0 DiT 配置，已验证 4x H20。

结构：

- 单 stage：`stage_type: diffusion`
- `devices: "0,1,2,3"`
- `model_stage: dit`
- `tensor_parallel_size: 4`
- `enable_expert_parallel: true`
- `omni_kv_config.need_recv_cache: true`
- `final_output_type: image`
- 默认 `seed: 42`

这个配置只定义了 diffusion stage。结合代码和注释看，它适合已有上游条件/KV 或直接走 diffusion pipeline 的场景；如果期望完整 AR -> DiT 文生图，需要使用包含 AR stage 的配置，或补齐上游 stage/edge。

### 5.6 `hunyuan_image3_t2i_2gpu.yaml`

用途：文件名写 2 GPU，但当前内容是单 AR stage。

结构：

- 单 stage：`stage_type: llm`
- `devices: "0,1"`
- `model_stage: AR`
- `tensor_parallel_size: 2`
- `engine_output_type: latent`
- `is_comprehension: true`
- `final_output_type: text`

需要特别注意：从 YAML 内容看它不是完整的 2 GPU 文生图 pipeline，因为没有 diffusion stage，也没有 runtime edge。文档/README 表格中称它为 T2I 2 GPU 配置，但当前文件内容更像 AR-only 2 GPU 配置，可能是 PR 中尚未整理完的命名或配置遗留。

### 5.7 `hunyuan_image3_moe_dit_2gpu_fp8.yaml`

用途：2x H200 上的 DiT FP8 在线量化配置。

结构：

- 单 diffusion stage。
- `devices: "0,1"`
- `quantization: "fp8"`
- `tensor_parallel_size: 2`
- `enable_expert_parallel: true`
- `omni_kv_config.need_recv_cache: true`
- `final_output_type: image`

和 `hunyuan_image3_t2i.yaml` 类似，它只描述 DiT/Diffusion 侧，不包含 AR stage。

## 6. 关键配置字段速查

### Stage 通用字段

- `stage_id`：stage 编号，runtime edge 和 `engine_input_source` 会引用它。
- `stage_type`：`llm` 或 `diffusion`。
- `runtime.process`：是否独立进程运行该 stage。
- `runtime.devices`：该 stage 使用的 GPU 列表。
- `runtime.max_batch_size`：stage 级任务 batch 上限。
- `runtime.requires_multimodal_data`：是否需要把原始多模态输入转发到该 stage。

### AR stage 字段

- `model_stage: AR`：选择 AR 路径。
- `model_arch: HunyuanImage3ForCausalMM`：通过 vLLM model registry 加载 AR 类。
- `worker_cls: GPUARWorker`：AR worker。
- `scheduler_cls: OmniARScheduler`：AR scheduler。
- `engine_output_type: latent`：AR 输出不是最终文本，而是供下游图像生成消费。
- `mm_encoder_tp_mode: data`：启用 vLLM 现有 ViT data-parallel 路径，使 AR 侧 ViT 禁用 TP、独立于 AR decoder 的 TP。
- `hf_overrides.rope_parameters`：强制 RoPE 参数，配合 HunyuanImage3 自定义 2D RoPE。
- `is_comprehension`：用于区分理解类输出和生成类输出。
- `default_sampling_params`：AR token 采样参数。

### Diffusion stage 字段

- `model_stage: dit` / `diffusion`：选择 diffusion 路径。
- `distributed_executor_backend: mp`：多进程扩散执行。
- `parallel_config.tensor_parallel_size`：DiT tensor parallel。
- `parallel_config.enable_expert_parallel`：MoE expert parallel。
- `parallel_config.cfg_parallel_size`：CFG 分支并行。
- `parallel_config.sequence_parallel_size`：序列并行。
- `parallel_config.vae_patch_parallel_size`：VAE patch parallel。
- `omni_kv_config.need_recv_cache`：接收上游 KV cache。
- `engine_input_source`：该 stage 从哪些上游 stage 取输入。
- `custom_process_input_func`：上游输出到本 stage 输入的转换函数。
- `default_sampling_params.num_inference_steps`：扩散步数。
- `default_sampling_params.guidance_scale`：CFG 强度。

### Runtime edge 字段

- `runtime.enabled`：启用 stage runtime。
- `runtime.edges`：定义 stage 间拓扑。
- `from` / `to`：边的源 stage 和目标 stage。
- `window_size: -1`：等待上游完整完成后触发下游。
- `max_inflight`：stage 间同时在途任务数。

## 7. 当前适配状态与注意点

1. HunyuanImage3 当前仍使用 legacy `stage_args` YAML，而不是新 schema `deploy` YAML。
2. 配置文件和 README 中部分描述不完全一致，尤其是 `hunyuan_image3_t2i.yaml`、`hunyuan_image3_t2i_2gpu.yaml` 是否代表完整 AR -> DiT pipeline，需要以 YAML 实际内容为准。
3. Diffusion pipeline 强制使用 Torch SDPA，切换其他 attention backend 可能不正确。
4. AR 侧 sampler 当前断言 batch size 为 1，因此相关 YAML 多数设置 `max_num_seqs: 1`。
5. 图片输入会同时产生 VAE 和 ViT token，显存和 token 数都比单纯文本模型更高。
6. HunyuanImage3 的 2D RoPE 不能直接使用 vLLM 默认 mRoPE，否则位置编码布局和权重不匹配。
7. 对图像生成质量和速度影响最大的参数通常是 `num_inference_steps`、`guidance_scale`、输出尺寸、TP/EP/CFG parallel、是否启用 cache/TeaCache。
