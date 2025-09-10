#include "qwen3_vl.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"

#include <random>
#include <thread>
#include <vector>

inline void createDeviceResource(DeviceResource *rsrc, const Qwen3VLMeta *meta,
                                 std::shared_ptr<Qwen3VLDeviceWeight> weights,
                                 infiniDevice_t device, int idev,
                                 int ndev, int dev_id,
                                 infinicclComm_t comm) {
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    *rsrc = DeviceResource{
        device,
        dev_id,
        handle,
        weights,
        stream,
        comm,
        memory_pool,
    };
    RUN_INFINI(infinirtDeviceSynchronize());
}

inline void releaseDeviceResource(DeviceResource &res) {
    infinirtDeviceSynchronize();
    // Release individual Tensors

    infiniopDestroyHandle(res.handle);
    res.handle = nullptr;
    infinirtStreamDestroy(res.stream);
    res.stream = nullptr;
    infinicclCommDestroy(res.comm);
    res.comm = nullptr;
}

void inferDeviceBatch(const Qwen3VLMeta *meta, DeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      const uint32_t *pos_ids, uint32_t pos_ids_len,
                      const float *pixel_values, uint32_t /*is_vision_mode*/, // 视觉数据指针，是否视觉模式不再需要
                      struct KVCache **kv_caches,
                      const float *temperature, const uint32_t *topk, const float *topp,
                      uint32_t *output, void *last_logits) {
    auto nlayer = meta->nlayer;
    auto nkvh = meta->nkvh / ndev;
    auto nh = meta->nh / ndev;
    auto ngroup = nh / nkvh;
    // auto dctx = meta.dctx;
    auto dh = meta->dh;
    auto d = meta->d;
    auto dt_logits = meta->dt_logits;
    auto di = meta->di / ndev;
    auto dvoc = meta->dvoc;
    auto stream = rsrc.stream;
    auto weight = rsrc.weights;
    bool has_qkv_bias = meta->has_qkv_bias;

    // Allocate buffers
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto q_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    auto k_buf = Tensor::buffer(dt_logits, {ntok, nkvh * dh}, rsrc.memory_pool);
    auto v_buf = Tensor::buffer(dt_logits, {ntok, nkvh * dh}, rsrc.memory_pool);

    auto gate_buf = Tensor::buffer(dt_logits, {ntok, di}, rsrc.memory_pool);
    auto up_buf = Tensor::buffer(dt_logits, {ntok, di}, rsrc.memory_pool);

    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(nreq);

    // Prepare inputs
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    size_t req_start = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        for (uint32_t i = 0; i < req_lens[req]; i++) {
            batch_pos_ids[req_start + i] = req_pos[req] + i;
        }
        req_start += req_lens[req];
    }

    // printf("here1\n");

    // 统一路径：若有视觉输入，先跑ViT得到visual_embeds；随后构建logits_in（视觉token用visual_embeds替换）
    std::shared_ptr<Tensor> pos_ids_buf;
    uint32_t num_patches = (pos_ids_len >= 2) ? (pos_ids_len / 2) : 0;
    if (pos_ids != nullptr && num_patches > 0) {
        if (rsrc.device == INFINI_DEVICE_CPU) {
            pos_ids_buf = Tensor::weight(const_cast<uint32_t *>(pos_ids), INFINI_DTYPE_U32, {num_patches, 2});
        } else {
            pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {num_patches, 2}, rsrc.memory_pool);
            RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), pos_ids, sizeof(uint32_t) * pos_ids_len,
                                           INFINIRT_MEMCPY_H2D, stream));
        }
    }

    // printf("here2\n");

    std::shared_ptr<Tensor> visual_embeds; // [num_patches, vision_hidden_size]
    const bool has_vision = (pixel_values != nullptr) && (num_patches > 0);
    if (has_vision) {
        // 根据权重形状确定实际参数: [vision_hidden_size, 3, temporal, patch, patch]
        uint32_t in_channels = 3;
        uint32_t temporal_patch_size = 2;
        uint32_t patch_size = 16;
        uint32_t vision_hidden_size = static_cast<uint32_t>(meta->vision_hidden_size);
        uint32_t patch_feature_dim = in_channels * temporal_patch_size * patch_size * patch_size;

        // 输入像素: [num_patches, 3, 2, 16, 16]
        std::shared_ptr<Tensor> pixel_values_buf;
        if (rsrc.device == INFINI_DEVICE_CPU) {
            pixel_values_buf = Tensor::weight(const_cast<float *>(pixel_values), dt_logits,
                                              {num_patches, in_channels, temporal_patch_size, patch_size, patch_size});
        } else {
            pixel_values_buf = Tensor::buffer(dt_logits, {num_patches, in_channels, temporal_patch_size, patch_size, patch_size}, rsrc.memory_pool);
            RUN_INFINI(infinirtMemcpyAsync(pixel_values_buf->data(), pixel_values,
                                           sizeof(float) * num_patches * patch_feature_dim,
                                           INFINIRT_MEMCPY_H2D, stream));
        }

        auto conv_output = Tensor::buffer(dt_logits, {num_patches, vision_hidden_size, 1, 1, 1}, rsrc.memory_pool);
        std::vector<int64_t> pads = {0, 0, 0};
        std::vector<int64_t> strides = {int64_t(temporal_patch_size), int64_t(patch_size), int64_t(patch_size)};
        std::vector<int64_t> dilations = {1, 1, 1};
        conv3d(conv_output,
               pixel_values_buf,
               weight->w_v_patch_embed_proj[0],
               weight->b_v_patch_embed_proj[0],
               pads, strides, dilations);
        visual_embeds = conv_output->view({num_patches, vision_hidden_size});
    }

    // printf("here3\n");
    // 构建 logits_in：文本token查表，视觉token用visual_embeds顺序替代
    size_t vis_idx = 0;
    for (uint32_t i = 0; i < ntok; i++) {
        const bool is_image_tok = (meta->image_token_id != 0 && tokens[i] == meta->image_token_id);
        const bool is_video_tok = (meta->video_token_id != 0 && tokens[i] == meta->video_token_id);
        if (has_vision && (is_image_tok || is_video_tok) && visual_embeds) {
            if (vis_idx < num_patches) {
                // 若视觉hidden与文本hidden不同，按较小维度拷贝
                uint32_t copy_dim = std::min<uint32_t>(d, static_cast<uint32_t>(visual_embeds->shape()[1]));
                if (copy_dim > 0) {
                    RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                                   visual_embeds->data(vis_idx * visual_embeds->shape()[1]),
                                                   dsize(dt_logits) * copy_dim, INFINIRT_MEMCPY_D2D, stream));
                }
                // 如需补零，这里暂不填充，其余维度保留为初始值（后续算子会覆盖）。
                vis_idx++;
            } else {
                // 安全保护：视觉patch不足，回退到文本嵌入
                RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                               weight->w_in_embd->data(tokens[i] * d),
                                               dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
            }
        } else {
            RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                           weight->w_in_embd->data(tokens[i] * d),
                                           dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
        }
    }

    // printf("here4\n");

    // Attention
    // attention inner
    size_t max_qk_size = 0;
    size_t max_seq_len = 0;

    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];
        auto seq_len = req_lens[req];
        auto total_len = past_len + seq_len;

        max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
    }

    auto qk_buf = Tensor::buffer(dt_logits, {nh * max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto q_rearrange = rearrange_q_buf->view({nkvh, ngroup, max_seq_len, dh});
    auto attn_val_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_gemm = attn_val_buf->view({nkvh, ngroup, max_seq_len, dh});

    // printf("here5\n");

    // Compute vit
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        // 1. Attention
        // rms norm
        rmsnorm(logits_out, logits_in, weight->w_attn_norm[layer], meta->epsilon);
        // qkv_proj
        linear(q_buf, logits_out,
               weight->w_attn_q[layer],
               1.0, 0.0, nullptr, has_qkv_bias ? weight->b_attn_q[layer] : nullptr);
        linear(k_buf, logits_out,
               weight->w_attn_k[layer],
               1.0, 0.0, nullptr, has_qkv_bias ? weight->b_attn_k[layer] : nullptr);
        linear(v_buf, logits_out,
               weight->w_attn_v[layer],
               1.0, 0.0, nullptr, has_qkv_bias ? weight->b_attn_v[layer] : nullptr);
        // mrope_2d（无pos_ids时跳过以避免崩溃）
        if (pos_ids_buf) {
            mrope_2d(q_buf->view({nh, ntok, dh}), q_buf->view({nh, ntok, dh}), pos_ids_buf, weight->sin_table, weight->cos_table);
            mrope_2d(k_buf->view({nkvh, ntok, dh}), k_buf->view({nkvh, ntok, dh}), pos_ids_buf, weight->sin_table, weight->cos_table);
        }

        // printf("here5.1\n");

        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto past_len = req_pos[req];
            auto seq_len = req_lens[req];
            auto total_len = past_len + seq_len;
            auto o = o_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto q = q_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto k = k_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, dh});
            auto v = v_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, dh});

            // self attention
            // concat
            rearrange(kv_caches[req]->k[idev][layer]->slice(0, past_len, seq_len), k);
            rearrange(kv_caches[req]->v[idev][layer]->slice(0, past_len, seq_len), v);
            // qk
            rearrange(q_rearrange->slice(2, 0, seq_len), q);
            auto qk_gemm = qk_buf->slice(0, 0, nh * seq_len * total_len)->view({nkvh, ngroup * seq_len, total_len});
            auto k_gemm = kv_caches[req]->k[idev][layer]->slice(0, 0, total_len)->permute({1, 2, 0});
            linear(qk_gemm, rearrange_q_buf->slice(1, 0, ngroup * seq_len), k_gemm, 1.f / float(sqrt(dh)), 0.f, nullptr, nullptr);
            // softmax
            auto qk_softmax = qk_gemm->view({nh, seq_len, total_len});
            causalSoftmax(qk_softmax, qk_softmax);
            auto v_gemm = kv_caches[req]->v[idev][layer]->slice(0, 0, total_len)->permute({1, 0, 2});
            linear(attn_val_buf->slice(1, 0, ngroup * seq_len), qk_gemm, v_gemm, 1.f, 0.f, nullptr, nullptr);
            // rearrange attn val
            rearrange(o, attn_val_gemm->slice(2, 0, seq_len));

            token_offset += seq_len;
        }

        // printf("here5.2\n");

        // o_proj
        linear(logits_in, o_buf, weight->w_attn_out[layer],
               1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual
        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }

        // printf("here5.3\n");

        // 2. FFN
        rmsnorm(logits_out, logits_in, weight->w_ffn_norm[layer], meta->epsilon);
        linear(gate_buf, logits_out,
               weight->w_ffn_gate[layer],
               1.0, 0.0, nullptr, nullptr);
        linear(up_buf, logits_out,
               weight->w_ffn_up[layer],
               1.0, 0.0, nullptr, nullptr);
        swiglu(gate_buf, up_buf, gate_buf);
        linear(logits_in, gate_buf,
               weight->w_ffn_down[layer],
               1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual
        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
    }

    // todo merger & deepstack

    // todo concat img_embd & text_embd

    // Compute llm
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        // 1. Attention
        // rms norm
        rmsnorm(logits_out, logits_in, weight->w_attn_norm[layer], meta->epsilon);
        // qkv_proj
        linear(q_buf, logits_out,
               weight->w_attn_q[layer],
               1.0, 0.0, nullptr, has_qkv_bias ? weight->b_attn_q[layer] : nullptr);
        linear(k_buf, logits_out,
               weight->w_attn_k[layer],
               1.0, 0.0, nullptr, has_qkv_bias ? weight->b_attn_k[layer] : nullptr);
        linear(v_buf, logits_out,
               weight->w_attn_v[layer],
               1.0, 0.0, nullptr, has_qkv_bias ? weight->b_attn_v[layer] : nullptr);
        // todo 3d mrope
        if (pos_ids_buf) {
            mrope_2d(q_buf->view({nh, ntok, dh}), q_buf->view({nh, ntok, dh}), pos_ids_buf, weight->sin_table, weight->cos_table);
            mrope_2d(k_buf->view({nkvh, ntok, dh}), k_buf->view({nkvh, ntok, dh}), pos_ids_buf, weight->sin_table, weight->cos_table);
        }

        // printf("here5.1\n");

        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto past_len = req_pos[req];
            auto seq_len = req_lens[req];
            auto total_len = past_len + seq_len;
            auto o = o_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto q = q_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto k = k_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, dh});
            auto v = v_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, dh});

            // self attention
            // concat
            rearrange(kv_caches[req]->k[idev][layer]->slice(0, past_len, seq_len), k);
            rearrange(kv_caches[req]->v[idev][layer]->slice(0, past_len, seq_len), v);
            // qk
            rearrange(q_rearrange->slice(2, 0, seq_len), q);
            auto qk_gemm = qk_buf->slice(0, 0, nh * seq_len * total_len)->view({nkvh, ngroup * seq_len, total_len});
            auto k_gemm = kv_caches[req]->k[idev][layer]->slice(0, 0, total_len)->permute({1, 2, 0});
            linear(qk_gemm, rearrange_q_buf->slice(1, 0, ngroup * seq_len), k_gemm, 1.f / float(sqrt(dh)), 0.f, nullptr, nullptr);
            // softmax
            auto qk_softmax = qk_gemm->view({nh, seq_len, total_len});
            causalSoftmax(qk_softmax, qk_softmax);
            auto v_gemm = kv_caches[req]->v[idev][layer]->slice(0, 0, total_len)->permute({1, 0, 2});
            linear(attn_val_buf->slice(1, 0, ngroup * seq_len), qk_gemm, v_gemm, 1.f, 0.f, nullptr, nullptr);
            // rearrange attn val
            rearrange(o, attn_val_gemm->slice(2, 0, seq_len));

            token_offset += seq_len;
        }

        // printf("here5.2\n");

        // o_proj
        linear(logits_in, o_buf, weight->w_attn_out[layer],
               1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual
        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }

        // printf("here5.3\n");

        // 2. FFN
        rmsnorm(logits_out, logits_in, weight->w_ffn_norm[layer], meta->epsilon);
        linear(gate_buf, logits_out,
               weight->w_ffn_gate[layer],
               1.0, 0.0, nullptr, nullptr);
        linear(up_buf, logits_out,
               weight->w_ffn_up[layer],
               1.0, 0.0, nullptr, nullptr);
        swiglu(gate_buf, up_buf, gate_buf);
        linear(logits_in, gate_buf,
               weight->w_ffn_down[layer],
               1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual
        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
    }

    // printf("here6\n");

    // Sample and Output
    if (idev == 0) {
        if (last_logits != nullptr) {
            rmsnorm(logits_out, logits_in, weight->w_out_norm, meta->epsilon);
            auto last_logits_buf = Tensor::buffer(dt_logits, {ntok, dvoc}, rsrc.memory_pool);
            linear(last_logits_buf, logits_out, weight->w_out_embd, 1.0, 0.0, nullptr, nullptr);
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(last_logits, last_logits_buf->data(), dsize(dt_logits) * ntok * dvoc, INFINIRT_MEMCPY_D2H));
        }
        if (output != nullptr) {
            size_t token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                token_offset += seq_len;
                rmsnorm(logits_out->slice(0, req, 1),
                        logits_in->slice(0, token_offset - 1, 1),
                        weight->w_out_norm,
                        meta->epsilon);
            }
            linear(prob_buf, logits_out->slice(0, 0, nreq), weight->w_out_embd, 1.0, 0.0, nullptr, nullptr);
            std::random_device _rd;
            std::mt19937 gen(_rd());
            token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                float random_val = std::uniform_real_distribution<float>(0, 1)(gen);
                randomSample(result_buf->slice(0, req, 1)->view_as({}, {}),
                             prob_buf->slice(0, req, 1)->view_as({dvoc}, {1}),
                             random_val, topp[req], topk[req], temperature[req]);
                token_offset += seq_len;
            }
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(result_cpu.data(), result_buf->data(),
                                      sizeof(int64_t) * nreq, INFINIRT_MEMCPY_D2H));
            for (uint32_t req = 0; req < nreq; req++) {
                output[req] = uint32_t(result_cpu[req]);
            }
        }
    }

    // printf("here7\n");
}

__C void
inferBatchQwen3VL(struct Qwen3VLModel *model,
                  const uint32_t *tokens, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                  const uint32_t *pos_ids, uint32_t pos_ids_len,
                  const float *pixel_values,
                  struct KVCache **kv_caches,
                  const float *temperature, const uint32_t *topk, const float *topp,
                  uint32_t *output) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.pos_ids = pos_ids;
    model->req.pos_ids_len = pos_ids_len;
    model->req.pixel_values = pixel_values;
    model->req.kv_caches = kv_caches;
    model->req.output = output;
    model->req.logits = nullptr;
    model->req.temperature = temperature;
    model->req.topk = topk;
    model->req.topp = topp;

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

__C void
forwardBatchQwen3VL(struct Qwen3VLModel *model,
                    const uint32_t *tokens, uint32_t ntok,
                    const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                    const uint32_t *pos_ids, uint32_t pos_ids_len,
                    const float *pixel_values,
                    struct KVCache **kv_caches,
                    void *logits) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.pos_ids = pos_ids;
    model->req.pos_ids_len = pos_ids_len;
    model->req.pixel_values = pixel_values;
    model->req.kv_caches = kv_caches;
    model->req.output = nullptr;
    model->req.logits = logits;
    model->req.temperature = nullptr;
    model->req.topk = nullptr;
    model->req.topp = nullptr;

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

void launchDevice(const Qwen3VLMeta *meta, std::shared_ptr<Qwen3VLDeviceWeight> weights, DeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {
    // Create Device Resource
    createDeviceResource(rsrc, meta, weights, device, idev, ndev, dev_id, comm);

    CacheManager cache_manager(100);
    InferenceContext ctx(rsrc->handle, rsrc->memory_pool, &cache_manager, rsrc->stream);

    // Set the inference context for this thread
    setInferenceContext(&ctx);

    {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.loaded = true;
        lock.unlock();
        state.cv_load.notify_one();
    }

    // Infer Loop
    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
        // quit if exit_flag is set
        if (state.exit_flag) {
            break;
        }

        inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok,
                         req.req_lens, req.nreq, req.req_pos, req.pos_ids, req.pos_ids_len,
                         req.pixel_values, 0,
                         req.kv_caches, req.temperature, req.topk, req.topp, req.output, req.logits);

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    // Clean-Up
    releaseDeviceResource(*rsrc);
    setInferenceContext(nullptr); // Clear the context when done
}

Qwen3VLModel::Qwen3VLModel(const Qwen3VLMeta *meta, const ModelWeights *weights_) {
    auto weights = (Qwen3VLWeights *)(weights_);
    device = weights->device();
    dev_ids = weights->devIds();
    int ndev = int(dev_ids.size());
    dev_resources = std::vector<DeviceResource>(ndev);
    states = std::vector<InferState>(ndev);
    threads.resize(ndev);

    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchDevice, meta, weights->device_weights()[i], &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
    }
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

__C struct Qwen3VLModel *
createQwen3VLModel(const Qwen3VLMeta *meta,
                   const ModelWeights *weights) {
    Qwen3VLModel *model = new Qwen3VLModel(meta, weights);
    return model;
}

__C void destroyQwen3VLModel(struct Qwen3VLModel *model) {
    auto ndev = model->dev_resources.size();

    for (size_t idev = 0; idev < ndev; idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].exit_flag = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }

    for (size_t idev = 0; idev < ndev; idev++) {
        model->threads[idev].join();
    }

    delete model;
}
