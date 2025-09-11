from typing import List, Sequence
import math
import os
from pathlib import Path
import safetensors
import sys
import time
import json
import torch
import transformers
import numpy as np
from PIL import Image

from libinfinicore_infer import (
    Qwen3VLModel,
    Qwen3VLMetaCStruct,
    DataType,
    DeviceType,
    KVCacheCStruct,
)
from infer_task import InferTask, KVCache

from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref

torch.set_default_device("cpu")


def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280):
    """基于 transformers 的 smart_resize 实现"""
    import math
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio too large: {max(height, width) / min(height, width)}")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def preprocess_image_qwen3vl(image_path: str):
    """
    完整的 Qwen3-VL 图像预处理流程
    基于 transformers 的实现：加载→resize→rescale→normalize→CHW→reshape→permute→flatten
    """
    # 配置参数 (从 Qwen3-VL config 中获取)
    patch_size = 16
    merge_size = 2  # spatial_merge_size
    temporal_patch_size = 2
    factor = patch_size * merge_size  # 28
    min_pixels = 4 * 28 * 28
    max_pixels = 16384 * 28 * 28

    # 1. 加载图像
    image = Image.open(image_path).convert('RGB')
    height, width = image.size[1], image.size[0]  # PIL: (width, height)

    # 2. Smart resize (保持宽高比，满足像素数和因子整除要求)
    resized_height, resized_width = smart_resize(height, width, factor, min_pixels, max_pixels)
    image = image.resize((resized_width, resized_height), Image.BILINEAR)

    print(f"图像预处理: {width}×{height} -> {resized_width}×{resized_height}")

    # 3. 转换为张量
    patches = torch.tensor(np.array(image)).float()

    # 4. Rescale (0-255 -> 0-1)
    patches = patches / 255.0

    # 5. Normalize (ImageNet 标准)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    patches = (patches - mean) / std

    # 6. CHW 调整: [H, W, C] -> [C, H, W]
    patches = patches.permute(2, 0, 1)

    # 7. 添加 batch 和时间维度 [C, H, W] -> [1, C, H, W] (模拟单帧)
    patches = patches.unsqueeze(0)

    # 8. Temporal padding (确保帧数能被 temporal_patch_size 整除)
    if patches.shape[0] % temporal_patch_size != 0:
        repeats = patches[-1:].repeat(temporal_patch_size - patches.shape[0] % temporal_patch_size, 1, 1, 1)
        patches = torch.cat([patches, repeats], dim=0)

    # 9. Grid 计算
    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

    # 9. Reshape 和 Permute (按照 transformers 实现)
    patches = patches.view(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)

    # 10. Flatten
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
    )

    # Grid_thw (注意：这里是原始的 grid 大小，不是 merge 后的)
    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int32)

    return flatten_patches, grid_thw


def compute_2d_mrope_pos_ids(grid_thw: torch.Tensor, spatial_merge_size: int = 2):
    """
    计算 2D MRoPE 的 pos_ids，基于 vLLM 的实现

    Args:
        grid_thw: [batch, 3] 张量，包含 [t, h, w] (原始 grid 大小)
        spatial_merge_size: 空间合并大小，默认2

    Returns:
        pos_ids: [num_patches, 2] 张量，包含 [h_pos, w_pos] 坐标
    """
    pos_ids_list = []

    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)

        # 按照 vLLM 的 rot_pos_emb 实现，考虑 spatial_merge_size
        # 生成高度位置索引
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

        # 生成宽度位置索引
        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()

        # 组合坐标并重复时间维度
        pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
        pos_ids_list.append(pos_ids)

    return torch.cat(pos_ids_list, dim=0)


class Qwen3VLMetaFromConfig(Qwen3VLMetaCStruct):
    def __init__(self, config, dtype=torch.bfloat16, max_tokens=None):
        if dtype == torch.float16:
            dt_ = DataType.INFINI_DTYPE_F16
        elif dtype == torch.float32:
            dt_ = DataType.INFINI_DTYPE_F32
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16
        else:
            dt_ = DataType.INFINI_DTYPE_BF16

        self.scale_input = 1.0
        self.scale_output = 1.0
        self.scale_o = 1.0
        self.scale_down = 1.0
        has_qkv_bias = 0
        eos_token_id = config["eos_token_id"]
        vision_config = config.get("vision_config", {})

        super().__init__(
            dt_logits=dt_,
            dt_linear_w=dt_,
            dt_norm_w=dt_,
            nlayer=config["num_hidden_layers"],
            d=config["hidden_size"],
            nh=config["num_attention_heads"],
            nkvh=config["num_key_value_heads"],
            dh=config["head_dim"],
            di=config["intermediate_size"],
            dctx=config["max_position_embeddings"] if max_tokens is None else max_tokens,
            dvoc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            theta=config["rope_theta"],
            end_token=eos_token_id,
            has_qkv_bias=has_qkv_bias,
            # vision encoder parameters
            use_qk_norm=1 if config.get("use_qk_norm", False) else 0,
            vision_hidden_size=vision_config.get("hidden_size", 768),
            vision_layers=vision_config.get("depth", 12),
            vision_heads=vision_config.get("num_heads", 12),
            patch_size=vision_config.get("patch_size", 16),
            img_size=vision_config.get("img_size", 768),
            image_token_id=int(config.get("image_token_id", 151654)),
            video_token_id=int(config.get("video_token_id", 151656)),
        )
        self.torch_dtype_logits = dtype
        # 保留到python对象上，供上层使用
        try:
            self.image_token_id = int(config.get("image_token_id", 151654))
        except Exception:
            self.image_token_id = 151654
        try:
            self.video_token_id = int(config.get("video_token_id", 151656))
        except Exception:
            self.video_token_id = 151656


class Qwen3VLBatchedTask:
    def __init__(self, tasks: List[InferTask], image_path: str | None = None, video_path: str | None = None, config: dict | None = None):
        self.tasks = tasks
        self.nreq = len(tasks)

        # Precompute fields
        token_lists = [t.tokens for t in tasks]
        self.req_lens_list = [len(toks) for toks in token_lists]
        self.req_pos_list = [t.pos for t in tasks]
        self.kv_cache_ptrs = [t.kvcache().data() for t in tasks]
        self.temperaturas_list = [t.temperature for t in tasks]
        self.topks_list = [t.topk for t in tasks]
        self.topps_list = [t.topp for t in tasks]

        # Flatten token lists - 对于 ViT，tokens 实际上是 patch embeddings
        flat_tokens = [tok for toks in token_lists for tok in toks]

        # 统一：ntok 始终为文本 token 数；pixel_values 仅在 prefill(首轮，pos==0) 且有图像时提供
        self.ntok = len(flat_tokens)
        self.pixel_values = None
        self.patch_dim = 0
        self.grid_thw = None
        self.image_path = image_path
        self.video_path = video_path
        # 从config中读取image/video token id（若存在）
        self.image_token_id = None
        self.video_token_id = None
        if isinstance(config, dict):
            self.image_token_id = config.get("image_token_id", None)
            self.video_token_id = config.get("video_token_id", None)
        # prefill 判断：仅当该 batch 中存在 pos==0 的请求且包含图像占位符时，计算像素与pos_ids
        any_prefill_with_image = False
        def is_image_tok(tid: int) -> bool:
            if self.image_token_id is not None:
                try:
                    return tid == int(self.image_token_id)
                except Exception:
                    pass
            return 151652 <= tid <= 151656
        def is_video_tok(tid: int) -> bool:
            if self.video_token_id is not None:
                try:
                    return tid == int(self.video_token_id)
                except Exception:
                    pass
            return tid == 151656

        for task in tasks:
            if task.pos == 0 and any(is_image_tok(token) or is_video_tok(token) for token in task.tokens):
                any_prefill_with_image = True
                break
        if any_prefill_with_image:
            try:
                # 优先使用image_path，否则尝试video_path（暂复用图像预处理以打通管道）
                src_path = self.image_path if self.image_path is not None else self.video_path
                if src_path is None:
                    raise RuntimeError("no image/video path provided for prefill with vision input")
                self.pixel_values, self.grid_thw = preprocess_image_qwen3vl(src_path)
                self.patch_dim = self.pixel_values.shape[1]
            except Exception as _e:
                self.pixel_values = None
                self.grid_thw = None
                self.patch_dim = 0

        # 实现 2D MRoPE pos_ids 计算
        # 集成图像 pos_ids 到批处理中
        flat_pos_ids = []
        self.has_vision = False  # 默认无视觉输入
        self.vision_pos_shape = None

        if any_prefill_with_image and getattr(self, 'pixel_values', None) is not None and getattr(self, 'grid_thw', None) is not None:
            try:
                pos_ids = compute_2d_mrope_pos_ids(self.grid_thw)
                for pos in pos_ids:
                    flat_pos_ids.extend([int(pos[0]), int(pos[1])])
                self.has_vision = True
                self.vision_pos_shape = pos_ids.shape
            except Exception as e:
                print(f"警告: 图像 pos_ids 计算失败，prefill 将降级为纯文本: {e}")
                self.has_vision = False
        else:
            # 纯文本或decode：为每个token提供简化pos_ids，避免C端空指针
            for toks in token_lists:
                for i in range(len(toks)):
                    flat_pos_ids.extend([i, 0])
            self.has_vision = False

        self.pos_ids_len = len(flat_pos_ids)

        # Convert to ctypes arrays in one pass
        self.tokens = (c_uint * self.ntok)(*flat_tokens)
        self.req_lens = (c_uint * self.nreq)(*self.req_lens_list)
        self.req_pos = (c_uint * self.nreq)(*self.req_pos_list)
        self.pos_ids = (c_uint * max(1, self.pos_ids_len))(*([0] + flat_pos_ids)[:max(1, self.pos_ids_len)])
        self.kv_caches = (POINTER(KVCacheCStruct) *
                          self.nreq)(*self.kv_cache_ptrs)
        self.temperaturas = (c_float * self.nreq)(*self.temperaturas_list)
        self.topks = (c_uint * self.nreq)(*self.topks_list)
        self.topps = (c_float * self.nreq)(*self.topps_list)

    def input_args(self):
        # pixel_values 作为裸指针传递；无视觉输入则传空指针
        if getattr(self, 'pixel_values', None) is not None:
            # 确保是连续内存
            pv = self.pixel_values.contiguous()
            pixel_values_ptr = c_void_p(int(pv.data_ptr()))
        else:
            pixel_values_ptr = c_void_p(0)

        return (
            self.tokens,
            self.ntok,
            self.req_lens,
            self.nreq,
            self.req_pos,
            self.pos_ids,
            self.pos_ids_len,
            pixel_values_ptr,
            self.kv_caches,
            self.temperaturas,
            self.topks,
            self.topps,
        )

    def get_vision_info(self):
        """获取视觉相关的信息，供 C++ 端使用"""
        return {
            'has_vision': getattr(self, 'has_vision', False),
            'vision_pos_shape': getattr(self, 'vision_pos_shape', None),
            'pos_ids_should_be_2d': getattr(self, 'has_vision', False)
        }


class Qwen3VLForCausalLM:
    def __init__(self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None):
        load_start_time = time.time()
        print(f"Creating model on {ndev} devices...")
        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config
        eos_token_id = self.config["eos_token_id"]
        self.eos_token_id = (
            [eos_token_id] if type(eos_token_id) == int else eos_token_id
        )
        self.dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
        self.ndev = ndev
        self.device = device
        self.meta = Qwen3VLMetaFromConfig(config, max_tokens=max_tokens)

        self.qwen3vl_model = Qwen3VLModel()

        self.weights = self.qwen3vl_model.create_weights(
            byref(self.meta),
            self.device,
            ndev,
            self.dev_ids,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir_path, trust_remote_code=True
        )

        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")

        load_start_time = time.time()
        print("Loading model weights to host...")

        self.load_all_safetensors_from_dir(os.path.join(model_dir_path))

        self.model_instance = self.qwen3vl_model.create_model(
            byref(self.meta),
            self.weights,
        )
        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")

    def load_all_safetensors_from_dir(self, dir_path_: str):
        dir_path_ = Path(dir_path_)
        total_keys = 0
        for file in sorted(dir_path_.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    total_keys += 1
                    # print(key)
                    tensor = f.get_tensor(key)
                    if "o_proj.scales" in key:
                        tensor = tensor * self.meta.scale_o
                    elif "down_proj.scales" in key:
                        tensor = tensor * self.meta.scale_down
                    elif "embed_tokens.weight" in key:
                        tensor = tensor * self.meta.scale_input
                    elif "lm_head.weight" in key:
                        tensor = tensor * self.meta.scale_output
                    self.qwen3vl_model.load_weight(
                        self.weights, key, tensor.data_ptr()
                    )
        print(f"加载的张量 key 总数: {total_keys}")

    def max_context_len(self):
        return self.meta.dctx

    def create_kv_cache(self):
        return self.qwen3vl_model.create_kv_cache(
            self.meta.nlayer,
            self.meta.dctx,
            self.meta.nkvh,
            self.meta.dh,
            self.meta.dh,
            self.meta.dt_logits,
            self.device,
            self.dev_ids,
            self.ndev,
        )

    def drop_kv_cache(self, kv_cache):
        self.qwen3vl_model.drop_kv_cache(kv_cache)

    def batch_infer_one_round(self, tasks: List[InferTask], image_path: str | None = None, video_path: str | None = None):
        output = (c_uint * len(tasks))()
        batch_inputs = Qwen3VLBatchedTask(tasks, image_path=image_path, video_path=video_path, config=self.config)
        self.qwen3vl_model.infer_batch(
            self.model_instance,
            *(batch_inputs.input_args()),
            output,
        )
        return list(output)

    def generate(self, input_content, max_steps, topp_=1.0, topk_=1, temperature_=1.0, image_path=None, video_path=None):
        input_content = self.tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": input_content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        print(input_content, end="", flush=True)
        tokens = self.tokenizer.encode(input_content)
        infer_task = InferTask(
            0,
            tokens,
            self.max_context_len(),
            temperature_,
            topk_,
            topp_,
            self.eos_token_id,
        )
        infer_task.bind_kvcache(KVCache(self))

        steps = 0
        total_time = 0
        output_content = ""

        for step_i in range(max_steps):
            start_time = time.time()
            # prefill: step 0，传入image/video；decode：后续步不传
            output_tokens = self.batch_infer_one_round(
                [infer_task],
                image_path=image_path if step_i == 0 else None,
                video_path=video_path if step_i == 0 else None,
            )
            end_time = time.time()
            steps += 1
            # output_str = (
            #     self.tokenizer._tokenizer.id_to_token(output_tokens[0])
            #     .replace("▁", " ")
            #     .replace("<0x0A>", "\n")
            # )
            output_str = self.tokenizer.decode(output_tokens[0])
            output_content += output_str
            print(output_str, end="", flush=True)
            if output_tokens[0] in self.eos_token_id:
                break
            infer_task.next(output_tokens[0])

            if step_i > 0:
                total_time += end_time - start_time

        print("\n")
        avg_time = total_time * 1000 / (steps - 1)
        print(f"Time per step: {avg_time:.3f}ms")

        infer_task._kv_cache.drop(self)
        return output_content, avg_time

    def perplexity(self, test_sequences: List[Sequence[int]], batch_size=10):
        tasks = [InferTask(i, [], self.max_context_len(), 1.0, 1, 1.0, self.eos_token_id) for i in range(batch_size)]
        kv_caches = [KVCache(self) for _ in range(batch_size)]

        nll = 0.0
        total_len = 0

        for i in range(0, len(test_sequences), batch_size):
            batch_id = 0
            true_tokens = []
            while batch_id < batch_size and batch_id + i < len(test_sequences):
                input_tokens = test_sequences[i + batch_id][:-1]
                true_tokens.extend(test_sequences[i + batch_id][1:])
                tasks[batch_id].tokens = input_tokens
                tasks[batch_id].bind_kvcache(kv_caches[batch_id])
                batch_id += 1

            batch_inputs = Qwen3VLBatchedTask(tasks[:batch_id], image_path=None, config=self.config)
            logits = torch.zeros(
                (batch_inputs.ntok, self.meta.dvoc), dtype=self.meta.torch_dtype_logits
            )
            # 评测路径：decode阶段不传像素；传递pos_ids以保持mrope输入稳定
            self.qwen3vl_model.forward_batch(
                self.model_instance,
                batch_inputs.tokens,
                batch_inputs.ntok,
                batch_inputs.req_lens,
                batch_inputs.nreq,
                batch_inputs.req_pos,
                batch_inputs.pos_ids,
                batch_inputs.pos_ids_len,
                c_void_p(0),
                batch_inputs.kv_caches,
                logits.data_ptr(),
            )

            logits = logits.float()
            token_ids = torch.tensor(true_tokens, dtype=torch.int64)  # [ntok,]
            log_probs = torch.nn.functional.log_softmax(
                logits, dim=-1)  # (ntok, vocab)
            token_logprobs = log_probs[
                torch.arange(batch_inputs.ntok), token_ids
            ]  # (ntok,)

            start = 0
            for l in batch_inputs.req_lens_list:
                nll += -token_logprobs[start: start + l].sum().item()
                start += l
            total_len += token_logprobs.numel()

        for task in tasks:
            task.release_kvcache()

        return math.exp(nll / total_len)

    def destroy_model_instance(self):
        self.qwen3vl_model.destroy_model(self.model_instance)
        print("Model destroyed")


def test():
    if len(sys.argv) < 3:
        print(
            "Usage: python qwen3vl.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)
    model_path = sys.argv[2]
    device_type = DeviceType.DEVICE_TYPE_CPU
    if sys.argv[1] == "--cpu":
        device_type = DeviceType.DEVICE_TYPE_CPU
    elif sys.argv[1] == "--nvidia":
        device_type = DeviceType.DEVICE_TYPE_NVIDIA
    elif sys.argv[1] == "--cambricon":
        device_type = DeviceType.DEVICE_TYPE_CAMBRICON
    elif sys.argv[1] == "--ascend":
        device_type = DeviceType.DEVICE_TYPE_ASCEND
    elif sys.argv[1] == "--metax":
        device_type = DeviceType.DEVICE_TYPE_METAX
    elif sys.argv[1] == "--moore":
        device_type = DeviceType.DEVICE_TYPE_MOORE
    elif sys.argv[1] == "--iluvatar":
        device_type = DeviceType.DEVICE_TYPE_ILUVATAR
    else:
        print(
            "Usage: python qwen3vl.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)

    # 首先测试 pos_ids 计算
    # test_pos_ids_calculation()

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    max_tokens = 1024
    model = Qwen3VLForCausalLM(model_path, device_type, ndev, max_tokens=max_tokens)
    image_path = "/home/cearx/qy/model/Qwen3-VL-2B-Vit-86M-0828/image3.jpg"
    model.generate("描述这张图片", 500, image_path=image_path)
    model.destroy_model_instance()


def test_pos_ids_calculation():
    """测试 2D MRoPE pos_ids 计算"""
    print("=== 测试 2D MRoPE pos_ids 计算 ===")

    # 测试图像路径
    image_path = "/home/cearx/qy/model/Qwen3-VL-2B-Vit-86M-0828/image3.jpg"

    try:
        # 预处理图像
        pixel_values, grid_thw = preprocess_image_qwen3vl(image_path)
        print(f"图像预处理完成:")
        print(f"  pixel_values shape: {pixel_values.shape}")
        print(f"  grid_thw: {grid_thw}")

        # 计算 pos_ids
        pos_ids = compute_2d_mrope_pos_ids(grid_thw)
        print(f"pos_ids 计算完成:")
        print(f"  pos_ids shape: {pos_ids.shape}")
        print(f"  pos_ids 前10个元素:")
        print(f"  {pos_ids[:10]}")
        print(f"  pos_ids 最后10个元素:")
        print(f"  {pos_ids[-10:]}")

        # 验证 pos_ids 的合理性
        t, h, w = grid_thw[0].tolist()
        # grid_thw 现在已经是 spatial merge 后的网格大小
        expected_patches = t * h * w
        actual_patches = pos_ids.shape[0]
        print(f"期望 patch 数量: {expected_patches}")
        print(f"实际 patch 数量: {actual_patches}")

        # 检查 pos_ids 的值范围
        h_max = pos_ids[:, 0].max().item()
        w_max = pos_ids[:, 1].max().item()
        print(f"pos_ids 高度范围: 0 到 {h_max}")
        print(f"pos_ids 宽度范围: 0 到 {w_max}")

        # 坐标范围应该对应 grid_thw 的范围
        expected_h_max = h - 1
        expected_w_max = w - 1
        print(f"预期高度范围: 0 到 {expected_h_max}")
        print(f"预期宽度范围: 0 到 {expected_w_max}")

        if expected_patches == actual_patches:
            print("✓ pos_ids 数量验证通过!")
        else:
            print("✗ pos_ids 数量验证失败!")
            print(f"  详细信息: grid_thw={grid_thw}, 期望={expected_patches}, 实际={actual_patches}")

        # 检查坐标范围是否正确
        if h_max == expected_h_max and w_max == expected_w_max:
            print("✓ pos_ids 坐标范围验证通过!")
        else:
            print("✗ pos_ids 坐标范围验证失败!")
            print(f"  实际坐标最大值: h={h_max}, w={w_max}")
            print(f"  期望坐标最大值: h={expected_h_max}, w={expected_w_max}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test()
