"""
trainer.py
----------
检索任务训练: 1 query + K candidate 独立 forward, 在指定层池化得 embedding,
图文 emb 平均, 算 InfoNCE。SAE hook 在每次 forward 时被触发。

支持单卡 / DDP 多卡: 通过 accelerate 自动处理。
  单卡: python run.py --config ...
  多卡: accelerate launch --num_processes 2 run.py --config ...
"""
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.tensorboard import SummaryWriter

try:
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    _HAS_ACCELERATE = True
except ImportError:
    _HAS_ACCELERATE = False


def _format_log(log_dict, keys=None):
    """精简格式化 log_dict 用于 iter 级打印 (空间紧凑)"""
    if keys is None:
        # 默认只显示几个关键指标,iter 日志不要太长
        keys = ["loss/total", "loss/info_nce_orig", "loss/info_nce_sae",
                "metric/top1_acc_orig", "metric/top1_acc_sae"]
    parts = []
    for k in keys:
        if k in log_dict:
            short_k = k.split("/")[-1]   # "loss/info_nce_orig" -> "info_nce_orig"
            parts.append(f"{short_k}={log_dict[k]:.3f}")
    return " ".join(parts)


class Trainer:
    def __init__(self, cfg, model, processor, sae=None, sae_hook=None,
                 train_dataset=None, eval_dataset=None, collate_fn=None,
                 accelerator=None):
        self.cfg = cfg
        self.model = model
        self.processor = processor
        self.sae = sae
        self.sae_hook = sae_hook

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collate_fn = collate_fn

        # accelerator: 由外部 (run.py) 创建并传入。单卡时也建议创建一个空的,
        # 这样 self.is_main_process 等 API 一致, 代码逻辑不分叉。
        if accelerator is None and _HAS_ACCELERATE:
            # find_unused_parameters=True 因为 LoRA 让大部分参数没有梯度,
            # DDP 默认会报错说找不到梯度。这个开关让它容忍这种情况。
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            accelerator = Accelerator(
                gradient_accumulation_steps=cfg.train.grad_accum_steps,
                kwargs_handlers=[ddp_kwargs],
            )
        self.accelerator = accelerator
        self.is_distributed = accelerator is not None and accelerator.num_processes > 1
        self.is_main = accelerator is None or accelerator.is_main_process

        self.device = next(model.parameters()).device
        self.output_dir = cfg.output_dir

        # 只在主进程创建日志/输出目录, 避免多进程同时创建冲突
        if self.is_main:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.output_dir, "logs"))
        else:
            self.writer = None

        # 池化层 (与 SAE 挂载层一致)
        self.pool_layer_idx = getattr(cfg.train, "pool_layer_idx", cfg.sae.layer_idx)

        self._pool_cache = {}
        self._pool_handle = self._register_pool_hook()

        self.global_step = 0
        self.best_metric = -float("inf")

    # ---------------------------------------------------------
    # 工具: 只在主进程打印
    # ---------------------------------------------------------
    def _print(self, msg):
        if self.is_main:
            print(msg, flush=True)

    # ---------------------------------------------------------
    # Hidden state 抓取 (用于池化得 embedding)
    # ---------------------------------------------------------
    def _register_pool_hook(self):
        base = self.model.base_model.model if hasattr(self.model, "base_model") else self.model
        layers = None
        for path in [
            lambda m: m.language_model.model.layers,
            lambda m: m.language_model.layers,
            lambda m: m.model.language_model.model.layers,
            lambda m: m.model.layers,
        ]:
            try:
                layers = path(base); break
            except AttributeError:
                continue
        assert layers is not None, "无法定位 LLM decoder layers"

        layer = layers[self.pool_layer_idx]

        def _hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._pool_cache["hidden"] = hidden  # [M, L, D], 保留梯度
            return output

        return layer.register_forward_hook(_hook)

    # ---------------------------------------------------------
    # 池化: hidden [M, L, D] + attention_mask [M, L] -> emb [M, D]
    # ---------------------------------------------------------
    @staticmethod
    def _masked_mean_pool(hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # [M, L, 1]
        summed = (hidden * mask).sum(dim=1)                    # [M, D]
        denom = mask.sum(dim=1).clamp_min(1e-6)                # [M, 1]
        return summed / denom

    # ---------------------------------------------------------
    # 把展平 emb 按 (sample_ids, role_ids) 切回 (query_emb, cand_embs)
    # 图文按 (img_present, text_present) 平均
    # ---------------------------------------------------------
    def _build_sample_embs(self, flat_emb, batch):
        """
        flat_emb:        [M, D]
        sample_ids:      [M]    每条展平样本属于原 batch 哪条 (0..B-1)
        role_ids:        [M]    0 = query, 1..K = 第 i 个 candidate
        text_present:    [M]
        img_present:     [M]

        当前实现是: 文本和图共享同一次 forward 的 emb (因为 LlavaNextProcessor
        会把 <image> token 和文本一起喂进 LLM)。所以"图文平均"在这层不需要单独
        分开 forward, 自然由 mean pooling 完成了视觉 token + 文本 token 的平均。

        若想做严格的图/文独立 forward 再平均, 需要把 collate 改成对图文样本各 forward
        一次, 这会让显存翻倍, 当前不推荐。

        返回:
          query_embs:  [B, D]
          cand_embs:   List[Tensor [K_i, D]]   (K_i 可不一致)
          gold_idx:    [B]
        """
        sample_ids = batch["sample_ids"]
        role_ids = batch["role_ids"]
        gold_idx = batch["gold_idx"]
        B = int(gold_idx.shape[0])

        query_embs = []
        cand_embs_list = []
        for b in range(B):
            mask_b = (sample_ids == b)
            roles_b = role_ids[mask_b]
            embs_b = flat_emb[mask_b]   # [(1+K_b), D]

            # query: role==0
            q_idx = (roles_b == 0).nonzero(as_tuple=True)[0]
            assert q_idx.numel() == 1, f"sample {b} should have exactly 1 query"
            q_emb = embs_b[q_idx[0]]   # [D]
            query_embs.append(q_emb)

            # candidates: role>=1, 按 role 升序排
            c_mask = (roles_b >= 1)
            c_roles = roles_b[c_mask]
            c_embs = embs_b[c_mask]
            order = torch.argsort(c_roles)
            cand_embs_list.append(c_embs[order])   # [K_b, D]

        query_embs = torch.stack(query_embs, dim=0)   # [B, D]
        return query_embs, cand_embs_list, gold_idx

    # ---------------------------------------------------------
    # InfoNCE 损失
    # ---------------------------------------------------------
    def _retrieval_loss(self, query_embs, cand_embs_list, gold_idx, sim_fn=None):
        """
        默认: 用 query 与该样本自己的 K 个 candidate 算 softmax (in-sample negatives)。

        sim_fn: 自定义相似度函数, 接收 (q [D], c [K, D]) 返回 logits [K]。
                None 时用默认的余弦相似度 (q @ c.T)/tau。
        """
        B = query_embs.shape[0] if isinstance(query_embs, torch.Tensor) else len(query_embs)
        temperature = getattr(self.cfg.train, "info_nce_temp", 0.07)

        losses = []
        accs = []
        for b in range(B):
            q = query_embs[b]
            c = cand_embs_list[b]
            if sim_fn is None:
                # 默认: 余弦相似度
                q_n = F.normalize(q, dim=-1)
                c_n = F.normalize(c, dim=-1)
                logits = (q_n @ c_n.T) / temperature
            else:
                logits = sim_fn(q, c) / temperature
            target = gold_idx[b].to(logits.device)
            losses.append(F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0)))
            accs.append((logits.argmax() == target).float())
        loss = torch.stack(losses).mean()
        acc = torch.stack(accs).mean()
        return loss, acc

    # ---------------------------------------------------------
    # SAE 稀疏路径: 用 SAE top-K 激活做检索
    # ---------------------------------------------------------
    @staticmethod
    def _sparse_pool(top_acts, top_indices, attention_mask):
        """
        Token 级稀疏激活池化成样本级稀疏向量。
        
        Args:
            top_acts:       [M, L, K]    每个 token 的 top-K 激活值
            top_indices:    [M, L, K]    对应的 feature 下标 (0..num_latents-1)
            attention_mask: [M, L]       padding 位置为 0

        Returns:
            pooled_acts:    [M, L*K]     展平后 (含 padding 位置, 但已 mask 为 0)
            pooled_indices: [M, L*K]     展平后的 indices
            
        说明: 不在这里去重/累加同 index 的激活, 因为稀疏点积函数会处理。
              这里只做"乘以 mask + 展平"两件事。
        """
        M, L, K = top_acts.shape
        # 把 padding token 的激活值清零, 但 indices 保留(避免下游索引错误)
        mask = attention_mask.unsqueeze(-1).to(top_acts.dtype)   # [M, L, 1]
        acts_masked = top_acts * mask                             # [M, L, K]
        # 归一化: 除以非 padding token 数 (mean pool 的语义)
        n_valid = mask.sum(dim=1).clamp_min(1.0)                  # [M, 1]
        acts_masked = acts_masked / n_valid.unsqueeze(-1)         # [M, L, K]

        # 展平 token 和 K 维
        pooled_acts = acts_masked.reshape(M, L * K)               # [M, L*K]
        pooled_indices = top_indices.reshape(M, L * K)            # [M, L*K]
        return pooled_acts, pooled_indices

    @staticmethod
    def _sparse_cosine_sim(q_acts, q_indices, c_acts, c_indices):
        """
        计算 query (1 个稀疏向量) 与 K 个 candidate 稀疏向量的余弦相似度。

        Args:
            q_acts:    [N_q]          query 的稀疏激活值
            q_indices: [N_q]          query 的 indices  
            c_acts:    [K_c, N_c]     K_c 个 candidate, 每个 N_c 个稀疏值
            c_indices: [K_c, N_c]     对应 indices

        Returns:
            sim: [K_c]   K 个相似度

        实现: 用 scatter 把稀疏向量"虚拟稠密化"到一个共享的稠密表示空间,
              但只在 query indices + cand indices 的并集上展开。
              复杂度 O(N_q + K_c * N_c), 不构造 [num_latents] 大矩阵。
        """
        K_c, N_c = c_acts.shape
        N_q = q_acts.shape[0]

        # 求 query 和所有 cand 的 indices 并集 (用 unique 拿到一个共享的小词表)
        all_indices = torch.cat([q_indices, c_indices.reshape(-1)], dim=0)
        unique_indices, inverse = torch.unique(all_indices, return_inverse=True)
        V = unique_indices.shape[0]   # 并集大小, V << num_latents
        
        # query 在并集词表上的稠密表示
        q_inv = inverse[:N_q]                                # [N_q]
        q_dense = torch.zeros(V, dtype=q_acts.dtype, device=q_acts.device)
        q_dense.scatter_add_(0, q_inv, q_acts)               # 同 index 累加

        # candidates 在并集词表上的稠密表示
        c_inv = inverse[N_q:].reshape(K_c, N_c)              # [K_c, N_c]
        c_dense = torch.zeros(K_c, V, dtype=c_acts.dtype, device=c_acts.device)
        c_dense.scatter_add_(1, c_inv, c_acts)

        # 余弦相似度
        q_norm = q_dense / (q_dense.norm() + 1e-8)
        c_norm = c_dense / (c_dense.norm(dim=-1, keepdim=True) + 1e-8)
        sim = c_norm @ q_norm                                # [K_c]
        return sim

    def _build_sparse_sample_embs(self, sae_cache, batch):
        """
        从 SAE hook cache 中拿到 token 级稀疏激活, 池化成每个样本的稀疏向量,
        再按 sample_ids/role_ids 切分。
        
        返回:
          q_sparse_list:    List[B] of (acts [N_q], indices [N_q])
          c_sparse_list:    List[B] of (acts [K_b, N_c], indices [K_b, N_c])
        """
        top_acts = sae_cache["z"]              # [M, L, K]
        top_indices = sae_cache["z_indices"]   # [M, L, K]
        attention_mask = batch["attention_mask"].to(top_acts.device)
        sample_ids = batch["sample_ids"]
        role_ids = batch["role_ids"]
        B = int(batch["gold_idx"].shape[0])

        # 1) 池化到样本级稀疏向量: [M, L*K]
        pooled_acts, pooled_indices = self._sparse_pool(top_acts, top_indices, attention_mask)

        # 2) 按 sample_ids/role_ids 切分
        q_list = []
        c_list = []
        for b in range(B):
            mask_b = (sample_ids == b)
            roles_b = role_ids[mask_b]
            acts_b = pooled_acts[mask_b]      # [(1+K_b), L*K]
            inds_b = pooled_indices[mask_b]   # [(1+K_b), L*K]

            q_idx = (roles_b == 0).nonzero(as_tuple=True)[0]
            q_acts = acts_b[q_idx[0]]         # [L*K]
            q_inds = inds_b[q_idx[0]]
            q_list.append((q_acts, q_inds))

            c_mask = (roles_b >= 1)
            c_roles = roles_b[c_mask]
            order = torch.argsort(c_roles)
            c_acts = acts_b[c_mask][order]    # [K_b, L*K]
            c_inds = inds_b[c_mask][order]
            c_list.append((c_acts, c_inds))

        return q_list, c_list

    def _retrieval_loss_sparse(self, q_sparse_list, c_sparse_list, gold_idx):
        """对 SAE 稀疏路径算 InfoNCE"""
        B = len(q_sparse_list)
        temperature = getattr(self.cfg.train, "info_nce_temp", 0.07)
        losses = []
        accs = []
        for b in range(B):
            q_acts, q_inds = q_sparse_list[b]
            c_acts, c_inds = c_sparse_list[b]
            logits = self._sparse_cosine_sim(q_acts, q_inds, c_acts, c_inds) / temperature
            target = gold_idx[b].to(logits.device)
            losses.append(F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0)))
            accs.append((logits.argmax() == target).float())
        return torch.stack(losses).mean(), torch.stack(accs).mean()

    # ---------------------------------------------------------
    # 总 loss = (loss_orig + loss_sae) / 2 + 可选 SAE 正则
    # ---------------------------------------------------------
    def compute_loss(self, batch):
        # 1) forward LLaVA, 池化层 hidden state 通过 _pool_handle 被存进 self._pool_cache
        model_inputs = {k: v.to(self.device) for k, v in batch.items()
                        if k in ("input_ids", "attention_mask", "pixel_values", "image_sizes")
                        and isinstance(v, torch.Tensor)}
        self._pool_cache.clear()
        _ = self.model(**model_inputs, use_cache=False)
        hidden = self._pool_cache["hidden"]                            # [M, L, D]

        # 2) 路径 A: 原始 hidden state pool 后算 InfoNCE
        flat_emb = self._masked_mean_pool(hidden, model_inputs["attention_mask"])
        query_embs, cand_embs_list, gold_idx = self._build_sample_embs(flat_emb, batch)
        nce_loss_orig, acc_orig = self._retrieval_loss(query_embs, cand_embs_list, gold_idx)

        log_dict = {
            "loss/info_nce_orig": nce_loss_orig.item(),
            "metric/top1_acc_orig": acc_orig.item(),
        }
        total = nce_loss_orig
        n_paths = 1

        # 3) 路径 B: SAE 稀疏激活算 InfoNCE (如果 SAE hook 抓到了)
        use_sae_path = (
            self.sae_hook is not None
            and self.sae_hook.cache
            and self.sae_hook.cache.get("z") is not None
            and self.sae_hook.cache.get("z_indices") is not None
        )
        if use_sae_path:
            q_sparse_list, c_sparse_list = self._build_sparse_sample_embs(
                self.sae_hook.cache, batch
            )
            nce_loss_sae, acc_sae = self._retrieval_loss_sparse(
                q_sparse_list, c_sparse_list, gold_idx
            )
            total = total + nce_loss_sae
            n_paths = 2
            log_dict["loss/info_nce_sae"] = nce_loss_sae.item()
            log_dict["metric/top1_acc_sae"] = acc_sae.item()

        # 4) 两路求平均
        total = total / n_paths
        # 兼容老指标名 (info_nce 现在 = 两路平均后)
        log_dict["loss/info_nce"] = total.item()
        log_dict["metric/top1_acc"] = (
            (acc_orig.item() + acc_sae.item()) / 2 if use_sae_path else acc_orig.item()
        )

        # 5) 可选: 额外的 SAE 重构/稀疏正则 (除非你想做这个,否则默认关掉)
        if self.sae_hook is not None and self.sae_hook.cache:
            cache = self.sae_hook.cache
            if self.cfg.sae.lambda_sparsity > 0 and cache.get("z") is not None:
                z = cache["z"]
                sparsity = z.abs().mean()
                total = total + self.cfg.sae.lambda_sparsity * sparsity
                log_dict["loss/sae_sparsity"] = sparsity.item()
            if self.cfg.sae.lambda_align > 0 and cache.get("x_hat") is not None:
                x, x_hat = cache["x"], cache["x_hat"]
                align = F.mse_loss(x_hat.float(), x.float())
                total = total + self.cfg.sae.lambda_align * align
                log_dict["loss/sae_align"] = align.item()

        log_dict["loss/total"] = total.item()
        return total, log_dict

    # ---------------------------------------------------------
    # Train / Eval / Save / Load
    # ---------------------------------------------------------
    def fit(self):
        cfg = self.cfg
        acc = self.accelerator

        loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.sae is not None and cfg.sae.trainable:
            params += [p for p in self.sae.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        total_steps = math.ceil(len(loader) / cfg.train.grad_accum_steps) * cfg.train.epochs
        warmup_steps = int(total_steps * cfg.train.warmup_ratio)
        scheduler = get_scheduler(
            cfg.train.lr_scheduler, optimizer,
            num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )

        # ---- 启动信息 ----
        n_trainable = sum(p.numel() for p in params if p.requires_grad)
        n_processes = acc.num_processes if acc is not None else 1
        effective_bs = cfg.train.batch_size * cfg.train.grad_accum_steps * n_processes
        self._print(f"[fit] dataset size: {len(self.train_dataset)} samples")
        self._print(f"[fit] iters per epoch: {len(loader)}")
        self._print(f"[fit] grad_accum_steps: {cfg.train.grad_accum_steps}, "
                    f"num_processes: {n_processes}")
        self._print(f"[fit] effective batch size: {effective_bs} "
                    f"(per_device {cfg.train.batch_size} x accum {cfg.train.grad_accum_steps} "
                    f"x procs {n_processes})")
        self._print(f"[fit] total optimizer steps: {total_steps}, warmup: {warmup_steps}")
        self._print(f"[fit] trainable params: {n_trainable:,}")
        self._print(f"[fit] starting training...")

        # ---- 用 accelerator 包装 ----
        if acc is not None:
            self.model, optimizer, loader, scheduler = acc.prepare(
                self.model, optimizer, loader, scheduler
            )
            self.device = acc.device

        self.model.train()
        optimizer.zero_grad()

        # 计时器,用于估算 it/s
        import time
        t_iter_start = time.time()
        iter_times = []   # 最近几个 iter 的耗时,用于平滑速度估计

        for epoch in range(cfg.train.epochs):
            self._print(f"[fit] >>> epoch {epoch} / {cfg.train.epochs - 1}")
            for step, batch in enumerate(loader):
                # 第一个 iter 单独标记 (用户能看到"模型真的开始跑了")
                if step == 0 and epoch == 0:
                    self._print(f"[fit] first iter starting (model forward + backward)...")

                ctx = acc.accumulate(self.model) if acc is not None else _nullcontext()
                with ctx:
                    loss, log_dict = self.compute_loss(batch)

                    if acc is not None:
                        acc.backward(loss)
                        if acc.sync_gradients:
                            acc.clip_grad_norm_(params, cfg.train.max_grad_norm)
                    else:
                        (loss / cfg.train.grad_accum_steps).backward()
                        if (step + 1) % cfg.train.grad_accum_steps == 0:
                            torch.nn.utils.clip_grad_norm_(params, cfg.train.max_grad_norm)

                    is_step_boundary = (
                        (acc is not None and acc.sync_gradients)
                        or (acc is None and (step + 1) % cfg.train.grad_accum_steps == 0)
                    )
                    if is_step_boundary:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        self.global_step += 1
                        self._on_step_end(epoch, log_dict, scheduler)

                # 记录 iter 耗时
                iter_time = time.time() - t_iter_start
                iter_times.append(iter_time)
                if len(iter_times) > 20:
                    iter_times.pop(0)
                t_iter_start = time.time()

                # 第一个 iter 完成时单独报: 确认整个 forward/backward 走通
                if step == 0 and epoch == 0:
                    self._print(f"[fit] first iter done in {iter_time:.1f}s "
                                f"({_format_log(log_dict)})")

                # iter 级进度: 每 iter_log_steps 个 iter 打一行简短状态
                # (注意区分: log_steps 是 optimizer step 计数, 这里是 iter 计数)
                iter_log_every = getattr(cfg.train, "iter_log_steps",
                                         max(cfg.train.grad_accum_steps // 2, 1))
                if (step + 1) % iter_log_every == 0:
                    avg_it = sum(iter_times) / len(iter_times)
                    self._print(
                        f"[iter] epoch {epoch} iter {step+1}/{len(loader)} "
                        f"step {self.global_step}/{total_steps} "
                        f"{avg_it:.1f}s/it {_format_log(log_dict)}"
                    )

        self.save("adapter_last")
        if self.writer is not None:
            self.writer.close()
        self._print("[fit] training done.")

    def _on_step_end(self, epoch, log_dict, scheduler):
        """每次 optimizer.step 后的回调: 日志 / 评估 / 保存"""
        cfg = self.cfg

        # 日志: 多卡时聚合 loss/metric (各卡平均), 只在主进程打印/写盘
        if self.global_step % cfg.train.log_steps == 0:
            agg = self._aggregate_log_dict(log_dict)
            if self.is_main:
                lr = scheduler.get_last_lr()[0]
                msg = (f"[step {self.global_step}] epoch={epoch} lr={lr:.2e} | "
                       + " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in agg.items()))
                print(msg, flush=True)
                for k, v in agg.items():
                    self.writer.add_scalar(k, v, self.global_step)
                self.writer.add_scalar("lr", lr, self.global_step)

        # 评估 + 保存最优 (eval_steps <= 0 表示禁用训练中评估)
        if (cfg.train.eval_steps > 0
                and self.global_step % cfg.train.eval_steps == 0
                and self.eval_dataset is not None):
            self._print(f"[step {self.global_step}] running evaluation...")
            metric = self.evaluate()
            if self.is_main:
                self.writer.add_scalar("eval/top1_acc", metric, self.global_step)
            if metric > self.best_metric:
                self.best_metric = metric
                self._print(f"[step {self.global_step}] new best! "
                            f"saving adapter_best (acc={metric:.4f})")
                self.save("adapter_best")
            self.model.train()

        # 定期 checkpoint
        if self.global_step % cfg.train.save_steps == 0:
            self.save(f"adapter_step{self.global_step}")

    def _aggregate_log_dict(self, log_dict):
        """多卡时把各 rank 的 log 平均"""
        if self.accelerator is None or not self.is_distributed:
            return log_dict
        out = {}
        for k, v in log_dict.items():
            t = torch.tensor(v, device=self.accelerator.device)
            t = self.accelerator.gather(t.unsqueeze(0)).mean()
            out[k] = t.item()
        return out

    @torch.no_grad()
    def evaluate(self, dataset=None):
        dataset = dataset or self.eval_dataset
        # eval 用很少的 worker: 数据量小不是瓶颈, 且多次 eval (step 100/200/...) 累积
        # worker 线程会导致 "can't start new thread"。DDP 下还会 ×进程数, 更要省。
        eval_workers = min(2, self.cfg.data.num_workers)
        loader = DataLoader(
            dataset, batch_size=self.cfg.eval.batch_size, shuffle=False,
            num_workers=eval_workers, collate_fn=self.collate_fn,
        )
        if self.accelerator is not None:
            loader = self.accelerator.prepare(loader)

        self.model.eval()
        total_acc = torch.tensor(0.0, device=self.device)
        n = torch.tensor(0, device=self.device)
        total_iters = len(loader)
        self._print(f"[eval] running on {len(dataset)} samples ({total_iters} iters)...")

        for i, batch in enumerate(loader):
            _, log_dict = self.compute_loss(batch)
            B = int(batch["gold_idx"].shape[0])
            total_acc += log_dict["metric/top1_acc"] * B
            n += B
            # 评估进度: 每 10% 报一次
            if (i + 1) % max(total_iters // 10, 1) == 0:
                pct = (i + 1) / total_iters * 100
                self._print(f"[eval] progress {i+1}/{total_iters} ({pct:.0f}%)")

        if self.accelerator is not None and self.is_distributed:
            total_acc = self.accelerator.reduce(total_acc, reduction="sum")
            n = self.accelerator.reduce(n, reduction="sum")

        acc = (total_acc / n.clamp_min(1)).item()
        self._print(f"[eval] done, top1_acc={acc:.4f}")

        # 显式释放 eval loader 的 worker 进程, 防多次 eval 累积线程 -> can't start new thread
        try:
            if hasattr(loader, "_iterator") and loader._iterator is not None:
                loader._iterator._shutdown_workers()
            del loader
            import gc; gc.collect()
        except Exception:
            pass

        self.model.train()   # eval 后切回 train 模式
        return acc

    def save(self, tag="adapter"):
        """只在主进程保存, 避免多卡同时写盘冲突"""
        # 多卡时先在所有卡上等齐
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        if not self.is_main:
            return

        save_dir = os.path.join(self.output_dir, tag)
        os.makedirs(save_dir, exist_ok=True)

        # 拿到没有 DDP 包装的真实 model 再保存
        if self.accelerator is not None:
            unwrapped = self.accelerator.unwrap_model(self.model)
        else:
            unwrapped = self.model

        unwrapped.save_pretrained(save_dir)
        if self.processor is not None:
            self.processor.save_pretrained(save_dir)
        if self.sae is not None and self.cfg.sae.trainable:
            torch.save(self.sae.state_dict(), os.path.join(save_dir, "sae.pt"))
        print(f"[save] -> {save_dir}", flush=True)

    def load(self, adapter_path):
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        sae_file = os.path.join(adapter_path, "sae.pt")
        if self.sae is not None and os.path.isfile(sae_file):
            self.sae.load_state_dict(torch.load(sae_file, map_location=self.device))
        self._print(f"[load] <- {adapter_path}")

    def __del__(self):
        try:
            self._pool_handle.remove()
        except Exception:
            pass


# 单卡 fallback: contextlib.nullcontext 在 Python 3.7+ 是标准的
from contextlib import nullcontext as _nullcontext