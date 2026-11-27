"""Chandra OCR on chandra_cuda's engine: pages in, raw model text out, as chandra's HF backend.

transformers provides the processor, the vision tower (on the GPU; its language model stays on
the CPU, unused) and the (t, h, w) positions; the engine runs the language model's prefill and
decodes up to `slots` pages at a time, with MTP-3 and CUDA graphs by default. A page leaves its
slot at an end token or after max_output_tokens; free slots are refilled from the queue at once.
Graphs are captured once per number of decoding pages and re-pointed at the current pages when the
set changes.

    gen = CudaGenerator()
    results = gen.generate(batch)        # chandra.model.schema.GenerationResult per page
"""

from __future__ import annotations

import glob
import os
import time

import torch

from .engine import Decoder

PAGE_TOKENS_MAX = 8192  # prompt budget per page (a 6.29 MP page is ~6.1K image tokens + the prompt)
# The runaway stop of chandra's vLLM client: detect_repeat_token on the last 2,048 characters every
# 8 tokens; a page stops once it has fired on 4 checks in a row (its output up to there is kept).
REPEAT_CHECK_EVERY = 8
REPEAT_STOP_AFTER = 4
REPEAT_TAIL_CHARS = 2048
REPEAT_TAIL_TOKENS = 768


def _flat(x):
    if isinstance(x, (list, tuple)):
        return torch.cat([_flat(t) for t in x])
    if hasattr(x, "pooler_output") and x.pooler_output is not None:
        return _flat(x.pooler_output)
    if hasattr(x, "last_hidden_state"):
        return _flat(x.last_hidden_state)
    return x.reshape(-1, x.shape[-1])


class CudaGenerator:
    def __init__(self, checkpoint: str | None = None, slots: int = 16, weights: str = "fp8", kv: str = "int8",
                 state: str = "fp16", lm_head: str = "exact", mtp: bool = True, mtp_weights: str = "fp8",
                 draft_vocab: int = 32768, max_output_tokens: int | None = None, stop_on_repeat: bool = True,
                 fused_vision: bool = True):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from chandra.settings import settings

        name = checkpoint or settings.MODEL_CHECKPOINT
        self.max_output_tokens = max_output_tokens or settings.MAX_OUTPUT_TOKENS
        self.processor = AutoProcessor.from_pretrained(name)
        self.hf = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16,
                                                              device_map={"": "cpu"}).eval()
        self.hf.model.visual.to("cuda")
        if fused_vision:  # fused q/k rotary in the vision tower (bitwise the same features)
            from .vision import patch_vision

            patch_vision(self.hf.model.visual)
        snap = name if os.path.isdir(name) else glob.glob(os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{name.replace('/', '--')}/snapshots/*/"))[0]
        self.dec = Decoder(snap, max_seqs=slots, max_ctx=PAGE_TOKENS_MAX + self.max_output_tokens + 64, mtp=mtp,
                           draft_tokens=3, weights=weights, kv=kv, lm_head=lm_head, mtp_weights=mtp_weights,
                           state_dtype=torch.float16 if state == "fp16" else torch.float32)
        tok = self.processor.tokenizer
        eos = self.hf.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        eos.add(tok.convert_tokens_to_ids("<|im_end|>"))
        self.eos = eos
        if mtp:
            added = sorted(int(i) for i in tok.added_tokens_decoder)
            rows = sorted(set(range(draft_vocab)) | set(added) | eos) if draft_vocab else None
            self.dec.set_draft_head("fp8", rows=torch.tensor(rows, device="cuda") if rows else None)
        self.mtp = mtp
        self.slots = slots
        self.stop_on_repeat = stop_on_repeat
        self.img_tok = self.hf.config.image_token_id
        self._graphs: dict[int, object] = {}  # number of decoding pages -> captured graph(s)
        self.graph_captures = 0
        self.stats: dict[str, float] = {}  # host seconds per phase of the last generate()

    def _graph_for(self, seqs: list[int]):
        g = self._graphs.get(len(seqs))
        if g is None:
            g = self.dec.spec_graph(seqs) if self.mtp else self.dec.graph_step(seqs)
            self._graphs[len(seqs)] = g
            self.graph_captures += 1
        else:
            g.rebind(seqs)
        return g

    def _prepare(self, item):
        """CPU side of a page (run ahead in a thread pool): item, or a callable returning it (to render
        lazily), -> the processor's tensors."""
        from chandra.model.hf import process_batch_element

        if callable(item):
            item = item()
        return self.processor.apply_chat_template([[process_batch_element(item)]], tokenize=True,
                                                  add_generation_prompt=True, return_dict=True, return_tensors="pt")

    @torch.no_grad()
    def _prefill(self, slot: int, inputs):
        """A prepared page into a slot. Returns (first token, final-norm hidden state at the last position)."""
        ids = inputs["input_ids"].to("cuda")
        n = ids.shape[1]
        if n > PAGE_TOKENS_MAX:
            raise ValueError(f"prompt of {n} tokens > {PAGE_TOKENS_MAX}")
        thw = inputs["image_grid_thw"].to("cuda")
        feats = _flat(self.hf.model.get_image_features(inputs["pixel_values"].to("cuda", torch.bfloat16), thw))
        emb = self.dec.embed[ids[0]].clone()
        emb[ids[0] == self.img_tok] = feats.to(emb.dtype)
        # The positions on the CPU, where the (unused) language model lives; the inputs are small.
        pos, delta = self.hf.model.get_rope_index(inputs["input_ids"], inputs["mm_token_type_ids"],
                                                  inputs["image_grid_thw"], None, inputs["attention_mask"])
        pos3 = pos[:, 0].transpose(0, 1).contiguous().to("cuda")
        hidden, first = self.dec.prefill(slot, emb, pos3, int(delta[0]))
        if self.mtp:
            nxt = torch.cat([ids[0, 1:], first.view(1)])
            self.dec.mtp_prefill(slot, nxt[: n - 1], hidden[: n - 1], pos3[: n - 1])
        return int(first), hidden[-1].clone()

    @torch.no_grad()
    def generate(self, batch, max_output_tokens: int | None = None, on_result=None, prefetch: int | None = None,
                 cpu_workers: int = 1, **kwargs):
        """Pages in (chandra BatchInputItem, or callables returning one, from any iterable), GenerationResult
        out (raw text, token count), in order. The callables and the processor run ahead of the GPU in
        `cpu_workers` threads (default one: PDF renderers such as pdfium are not thread-safe), up to
        `prefetch` pages (default: the slot count). on_result(index, result), if given, is called as each
        page finishes (and the results are not kept)."""
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        from chandra.model.schema import GenerationResult
        from chandra.model.util import detect_repeat_token

        max_new = max_output_tokens or self.max_output_tokens
        tok = self.processor.tokenizer
        hits: dict[int, int] = {}

        def looping(it) -> bool:  # True once the page has repeated on REPEAT_STOP_AFTER checks in a row
            text = tok.decode(out[it][-REPEAT_TAIL_TOKENS:], skip_special_tokens=True)
            hits[it] = hits.get(it, 0) + 1 if detect_repeat_token(text[-REPEAT_TAIL_CHARS:]) else 0
            return hits[it] >= REPEAT_STOP_AFTER
        d = self.dec
        source = enumerate(iter(batch))
        pool = ThreadPoolExecutor(cpu_workers)
        queue = deque()  # (index, future of the prepared inputs)
        depth = prefetch or self.slots

        def top_up():
            while len(queue) < depth:
                nxt = next(source, None)
                if nxt is None:
                    return
                queue.append((nxt[0], pool.submit(self._prepare, nxt[1])))

        results_kept: dict[int, object] = {}

        def finish(it):
            ids = [t for t in out.pop(it) if t not in self.eos]
            n_tok = len(ids) + (1 if ended.pop(it, False) else 0)
            text = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            r = GenerationResult(raw=text, token_count=n_tok, error=False)
            hits.pop(it, None)
            if on_result is not None:
                on_result(it, r)
            else:
                results_kept[it] = r

        ended: dict[int, bool] = {}
        st = self.stats = {"wait_cpu": 0.0, "prefill": 0.0, "graph": 0.0, "step": 0.0, "host": 0.0}
        clock = time.perf_counter
        top_up()
        out: dict[int, list[int]] = {}
        slot_of: dict[int, int] = {}  # item -> slot, for pages still decoding
        free = list(range(self.slots))
        last: dict[int, int] = {}
        drafts: dict[int, torch.Tensor] = {}
        graph, members = None, []
        while queue or slot_of:
            top_up()
            if queue and free:  # admit pages into every free slot
                new = []
                while queue and free:
                    (it, fut), s = queue.popleft(), free.pop(0)
                    t0 = clock()
                    prepared = fut.result()
                    t1 = clock()
                    first, h = self._prefill(s, prepared)  # ends with a sync (the first token)
                    st["wait_cpu"] += t1 - t0
                    st["prefill"] += clock() - t1
                    top_up()
                    out[it] = [first]
                    if first in self.eos or max_new <= 1:
                        ended[it] = first in self.eos
                        free.append(s)
                        finish(it)
                        continue
                    slot_of[it], last[it] = s, first
                    new.append((it, h))
                if self.mtp and new:
                    t0 = clock()
                    dr = d.first_drafts(torch.tensor([last[it] for it, _ in new], device="cuda"),
                                        torch.stack([h for _, h in new]), [slot_of[it] for it, _ in new])
                    for j, (it, _) in enumerate(new):
                        drafts[it] = dr[j].clone()
                    st["prefill"] += clock() - t0
                graph = None
            if not slot_of:
                continue
            if graph is None:  # a new set of pages: the graphs for that many, re-pointed at them
                t0 = clock()
                members = sorted(slot_of, key=lambda it: slot_of[it])
                graph = self._graph_for([slot_of[it] for it in members])
                st["graph"] += clock() - t0
            t0 = clock()
            if self.mtp:
                emit, nlast, ndrafts = graph.step(torch.tensor([last[it] for it in members], device="cuda"),
                                                  torch.stack([drafts[it] for it in members]))
                for j, (it, t) in enumerate(zip(members, nlast.tolist())):
                    last[it] = t
                    drafts[it] = ndrafts[j].clone()
            else:
                _, nxt = graph(torch.tensor([last[it] for it in members], device="cuda"))
                for it in members:
                    d.length[slot_of[it]] += 1
                toks = nxt.tolist()
                emit = [[t] for t in toks]
                for it, t in zip(members, toks):
                    last[it] = t
            t1 = clock()
            st["step"] += t1 - t0
            for j, it in enumerate(members):
                for t in emit[j]:
                    out[it].append(t)
                    stop = t in self.eos or len(out[it]) >= max_new
                    if not stop and self.stop_on_repeat and len(out[it]) % REPEAT_CHECK_EVERY == 0:
                        stop = looping(it)
                    if stop:
                        ended[it] = t in self.eos
                        free.append(slot_of.pop(it))
                        graph = None
                        finish(it)
                        break
            st["host"] += clock() - t1
        pool.shutdown()
        return [results_kept[i] for i in sorted(results_kept)] if on_result is None else None
