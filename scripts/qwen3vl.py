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
        )
        self.torch_dtype_logits = dtype


class Qwen3VLBatchedTask:
    def __init__(self, tasks: List[InferTask]):
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

        # Flatten token lists
        flat_tokens = [tok for toks in token_lists for tok in toks]
        self.ntok = len(flat_tokens)

        # Convert to ctypes arrays in one pass
        self.tokens = (c_uint * self.ntok)(*flat_tokens)
        self.req_lens = (c_uint * self.nreq)(*self.req_lens_list)
        self.req_pos = (c_uint * self.nreq)(*self.req_pos_list)
        self.kv_caches = (POINTER(KVCacheCStruct) *
                          self.nreq)(*self.kv_cache_ptrs)
        self.temperaturas = (c_float * self.nreq)(*self.temperaturas_list)
        self.topks = (c_uint * self.nreq)(*self.topks_list)
        self.topps = (c_float * self.nreq)(*self.topps_list)

    def input_args(self):
        return (
            self.tokens,
            self.ntok,
            self.req_lens,
            self.nreq,
            self.req_pos,
            self.kv_caches,
            self.temperaturas,
            self.topks,
            self.topps,
        )


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

    def batch_infer_one_round(self, tasks: List[InferTask]):
        output = (c_uint * len(tasks))()
        batch_inputs = Qwen3VLBatchedTask(tasks)
        self.qwen3vl_model.infer_batch(
            self.model_instance,
            *(batch_inputs.input_args()),
            output,
        )
        return list(output)

    def generate(self, input_content, max_steps, topp_=1.0, topk_=1, temperature_=1.0):
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
            output_tokens = self.batch_infer_one_round([infer_task])
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
        tasks = [
            InferTask(i, [], self.max_context_len(),
                      1.0, 1, 1.0, self.eos_token_id)
            for i in range(batch_size)
        ]
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

            batch_inputs = Qwen3VLBatchedTask(tasks[:batch_id])
            logits = torch.zeros(
                (batch_inputs.ntok, self.meta.dvoc), dtype=self.meta.torch_dtype_logits
            )
            self.qwen3vl_model.forward_batch(
                self.model_instance,
                batch_inputs.tokens,
                batch_inputs.ntok,
                batch_inputs.req_lens,
                batch_inputs.nreq,
                batch_inputs.req_pos,
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

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model = Qwen3VLForCausalLM(model_path, device_type, ndev)
    model.generate("山东最高的山是？", 500)
    model.destroy_model_instance()


if __name__ == "__main__":
    test()
