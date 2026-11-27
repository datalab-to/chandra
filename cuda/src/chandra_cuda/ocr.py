"""Chandra OCR on chandra_cuda's engine: pages in, raw model text out, as chandra's HF backend.

transformers provides the processor, the vision tower (on the GPU; its language model stays on
the CPU, unused) and the (t, h, w) positions; the engine runs the language model's prefill and
decodes up to `slots` pages at a time, with MTP-3 and CUDA graphs by default. A page leaves its
slot at an end token or after max_output_tokens; free slots are refilled from the queue at once.
Graphs are captured once per number of decoding pages and re-pointed at the current pages when the
set changes. A page whose text repeats is generated again with sampling, as chandra's vLLM client
does (temperature 0.2 per attempt up to 0.8, top-p 0.95, up to max_retries times).

With overlap (the default where the driver can split a GPU's SMs, see partition.py), a worker thread
runs the vision tower and the prefill of the next pages on most of the SMs while the decode steps of
the current pages run on the rest; a page joins the decoding set when its prefill is done. The worker
runs pages through the vision tower ahead of free slots (up to the prefetch depth); while it has
nothing to do (every slot busy, every queued page through vision) the steps use the whole GPU. Each
page's tokens are the same as without overlap (the engine's kernels do not depend on the batch,
its plans not on the SM count).

    gen = CudaGenerator()
    results = gen.generate(batch)        # chandra.model.schema.GenerationResult per page
"""

from __future__ import annotations

import glob
import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

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
    def __init__(self, checkpoint: str | None = None, slots: int = 16, weights: str = "int8", kv: str = "int8",
                 state: str = "fp16", lm_head: str = "exact", mtp: bool = True, mtp_weights: str = "fp8",
                 draft_vocab: int = 32768, max_output_tokens: int | None = None, stop_on_repeat: bool = True,
                 fused_vision: bool = True, overlap: bool = True, decode_sms: int | None = None,
                 max_retries: int | None = None):
        """overlap: run vision and prefill beside the decode steps on separate SMs (when the driver
        supports green contexts); decode_sms: the decode steps' share while a prefill runs (default
        about 28% of the SMs: alone the steps need few SMs, but beside a prefill the two share DRAM
        bandwidth roughly by SM count). max_retries: sampled attempts for a page whose text repeats
        (default chandra's MAX_VLLM_RETRIES; 0 keeps the greedy text)."""
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from chandra.settings import settings

        name = checkpoint or settings.MODEL_CHECKPOINT
        self.max_output_tokens = max_output_tokens or settings.MAX_OUTPUT_TOKENS
        self.max_retries = settings.MAX_VLLM_RETRIES if max_retries is None else max_retries
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
        self._plain_graphs: dict[int, object] = {}  # the same for single-token steps (sampled retries)
        self.retried = 0
        self.graph_captures = 0
        self.stats: dict[str, float] = {}  # host seconds per phase of the last generate()
        self.overlap_parts = None
        if overlap:
            from .partition import sm_partitions

            sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
            self.overlap_parts = sm_partitions(torch.device("cuda", torch.cuda.current_device()),
                                               decode_sms or max(4, 2 * round(sms * 0.14)))
        self._shadows = self.dec.prefill_shadows() if self.overlap_parts else None

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
    def _vision(self, inputs):
        """A prepared page through the vision tower -> (input ids, prompt embeddings, (t, h, w) positions,
        rope delta)."""
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
        return ids, emb, pos3, int(delta[0])

    @torch.no_grad()
    def _lm_prefill(self, slot: int, ids, emb, pos3, delta: int, shadows: dict | None = None):
        """The language model's prefill (and the MTP layer's) of a page into a slot. Returns (first token,
        final-norm hidden state at the last position)."""
        sh = shadows or {}
        hidden, first = self.dec.prefill(slot, emb, pos3, delta, head=sh.get("lm_head"))
        if self.mtp:
            n = ids.shape[1]
            nxt = torch.cat([ids[0, 1:], first.view(1)])
            self.dec.mtp_prefill(slot, nxt[: n - 1], hidden[: n - 1], pos3[: n - 1], shadows=shadows)
        return int(first), hidden[-1].clone()

    def _prefill(self, slot: int, inputs):
        """A prepared page into a slot (vision, then prefill). Returns (first token, last hidden state)."""
        return self._lm_prefill(slot, *self._vision(inputs))

    @torch.no_grad()
    def generate(self, batch, max_output_tokens: int | None = None, on_result=None, prefetch: int | None = None,
                 cpu_workers: int = 1, **kwargs):
        """Pages in (chandra BatchInputItem, or callables returning one, from any iterable), GenerationResult
        out (raw text, token count), in order. The callables and the processor run ahead of the GPU in
        `cpu_workers` threads (default one: PDF renderers such as pdfium are not thread-safe), up to
        `prefetch` pages (default: the slot count). on_result(index, result), if given, is called as each
        page finishes (and the results are not kept)."""
        from chandra.model.schema import GenerationResult
        from chandra.model.util import detect_repeat_token

        max_new = max_output_tokens or self.max_output_tokens
        max_retries = getattr(self, "max_retries", 0)

        def should_retry(text) -> bool:  # chandra's vLLM client: retry a generation whose text repeats
            return bool(detect_repeat_token(text) or (len(text) > 50 and detect_repeat_token(text, cut_from_end=50)))
        tok = self.processor.tokenizer
        d = self.dec
        parts = getattr(self, "overlap_parts", None)
        clock = time.perf_counter
        st = self.stats = {"wait_cpu": 0.0, "prefill": 0.0, "graph": 0.0, "step": 0.0, "host": 0.0}
        hits: dict[int, int] = {}
        out: dict[int, list[int]] = {}
        ended: dict[int, bool] = {}
        results_kept: dict[int, object] = {}
        slot_of: dict[int, int] = {}  # item -> slot, for pages still decoding
        last: dict[int, int] = {}
        drafts: dict[int, torch.Tensor] = {}

        def looping(it) -> bool:  # True once the page has repeated on REPEAT_STOP_AFTER checks in a row
            text = tok.decode(out[it][-REPEAT_TAIL_TOKENS:], skip_special_tokens=True)
            hits[it] = hits.get(it, 0) + 1 if detect_repeat_token(text[-REPEAT_TAIL_CHARS:]) else 0
            return hits[it] >= REPEAT_STOP_AFTER

        items: dict[int, object] = {}  # pages in flight (kept for a sampled retry)
        retry: list = []  # (index, item, attempt) for the sampled pass

        def deliver(it, text, n_tok):
            items.pop(it, None)
            r = GenerationResult(raw=text, token_count=n_tok, error=False)
            if on_result is not None:
                on_result(it, r)
            else:
                results_kept[it] = r

        def finish(it, attempt=0):
            ids = [t for t in out.pop(it) if t not in self.eos]
            n_tok = len(ids) + (1 if ended.pop(it, False) else 0)
            text = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            hits.pop(it, None)
            if attempt < max_retries and should_retry(text):
                retry.append((it, items[it], attempt + 1))
            else:
                deliver(it, text, n_tok)

        source = enumerate(iter(batch))
        pool = ThreadPoolExecutor(cpu_workers)
        pending = deque()  # (index, future of the prepared inputs)
        depth = prefetch or self.slots

        def top_up():
            while len(pending) < depth:
                nxt = next(source, None)
                if nxt is None:
                    return
                items[nxt[0]] = nxt[1]
                pending.append((nxt[0], pool.submit(self._prepare, nxt[1])))

        free = list(range(self.slots))  # without overlap: free slots
        free_q: queue.Queue = queue.Queue()  # with overlap: free slots for the worker (None stops it)
        ready: queue.Queue = queue.Queue()  # with overlap: prefilled pages, then None (or an exception)
        release = free_q.put if parts else free.append
        full = torch.cuda.current_stream()
        cur = full
        worker = None
        worker_done = parts is None
        stop_worker = threading.Event()
        worker_busy = threading.Event()  # set while the worker has GPU work in flight

        def admit(it, s, first) -> bool:
            out[it] = [first]
            if first in self.eos or max_new <= 1:
                ended[it] = first in self.eos
                release(s)
                finish(it)
                return False
            slot_of[it], last[it] = s, first
            return True

        if parts:
            (_, dstream), (_, wstream) = parts
            for b in range(1, self.slots + 1):  # every graph before the worker runs (no capture beside it)
                if b not in self._graphs:
                    self._graph_for(list(range(b)))
            for k in ("w_vision_prefill", "w_wait_slot", "w_wait_cpu", "wait_prefill"):
                st[k] = 0.0
            for s in free:
                free_q.put(s)
            dev = torch.cuda.current_device()

            def work():
                try:
                    worker_busy.set()
                    torch.cuda.set_device(dev)
                    wstream.wait_stream(full)
                    seen = deque()  # (index, vision outputs) waiting for a slot
                    with torch.no_grad(), torch.cuda.stream(wstream):
                        top_up()
                        while not stop_worker.is_set():
                            s = None
                            if seen:
                                try:
                                    s = free_q.get_nowait()
                                except queue.Empty:
                                    pass
                            if s is None and pending and len(seen) < depth:
                                it, fut = pending.popleft()
                                top_up()
                                t0 = clock()
                                inputs = fut.result()
                                t1 = clock()
                                seen.append((it, self._vision(inputs)))
                                st["w_wait_cpu"] += t1 - t0
                                st["w_vision_prefill"] += clock() - t1
                                continue
                            if not seen:
                                break
                            if s is None:
                                t0 = clock()
                                wstream.synchronize()  # its work is done: the steps may take the whole GPU
                                worker_busy.clear()
                                s = free_q.get()
                                worker_busy.set()
                                st["w_wait_slot"] += clock() - t0
                                if s is None:
                                    break
                            it, v = seen.popleft()
                            t0 = clock()
                            first, h = self._lm_prefill(s, *v, shadows=self._shadows)
                            ev = torch.cuda.Event()
                            ev.record(wstream)
                            st["w_vision_prefill"] += clock() - t0
                            ready.put((it, s, first, h, ev))
                    ready.put(None)
                except BaseException as e:  # handed to the main thread
                    ready.put(e)
                finally:
                    worker_busy.clear()

            worker = threading.Thread(target=work, name="chandra-prefill", daemon=True)
            worker.start()

        graph, members = None, []
        try:
            while True:
                want = dstream if parts and worker_busy.is_set() else full
                if want is not cur:  # decode on its partition while the worker has work, else on the whole GPU
                    want.wait_stream(cur)
                    cur = want
                with torch.cuda.stream(cur):
                    new = []
                    if parts:
                        block = not slot_of
                        t0 = clock()
                        while not worker_done:
                            try:
                                r = ready.get(block=block)
                            except queue.Empty:
                                break
                            block = False
                            if r is None:
                                worker_done = True
                            elif isinstance(r, BaseException):
                                raise r
                            else:
                                it, s, first, h, ev = r
                                cur.wait_event(ev)
                                h.record_stream(cur)
                                if admit(it, s, first):
                                    new.append((it, h))
                        st["wait_prefill"] += clock() - t0
                        if worker_done and not slot_of and not new:
                            break
                    else:
                        top_up()
                        if not pending and not slot_of:
                            break
                        while pending and free:  # admit pages into every free slot
                            (it, fut), s = pending.popleft(), free.pop(0)
                            t0 = clock()
                            prepared = fut.result()
                            t1 = clock()
                            first, h = self._prefill(s, prepared)  # ends with a sync (the first token)
                            st["wait_cpu"] += t1 - t0
                            st["prefill"] += clock() - t1
                            top_up()
                            if admit(it, s, first):
                                new.append((it, h))
                    if new:
                        if self.mtp:
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
                                release(slot_of.pop(it))
                                graph = None
                                finish(it)
                                break
                    st["host"] += clock() - t1
        finally:
            if worker is not None:
                stop_worker.set()
                free_q.put(None)
                worker.join()
            full.wait_stream(cur)
        try:
            t0 = clock()
            self.retried = 0
            if retry:
                self._sampled_pass(retry, max_new, pool, out, ended, looping, finish)
            st["retries"] = float(self.retried)
            st["retry_s"] = clock() - t0
        finally:
            pool.shutdown()
        return [results_kept[i] for i in sorted(results_kept)] if on_result is None else None

    def _sample(self, hidden: torch.Tensor, temps: list, gens: list, top_p: float = 0.95) -> list:
        """Tokens sampled from bf16(hidden @ embed.T) (the tied LM head) at each row's temperature, top-p."""
        logits = (hidden @ self.dec.embed.T).float()
        toks = []
        for b in range(logits.shape[0]):
            p = torch.softmax(logits[b] / temps[b], dim=-1)
            ps, ix = torch.sort(p, descending=True)
            keep = (torch.cumsum(ps, 0) - ps) < top_p  # the smallest set reaching top_p (at least one)
            ps = ps * keep
            toks.append(int(ix[torch.multinomial(ps / ps.sum(), 1, generator=gens[b])]))
        return toks

    @torch.no_grad()
    def _sampled_pass(self, retry, max_new, pool, out, ended, looping, finish):
        """The pages in retry, (index, item, attempt) each, again from their prompts with sampling
        (temperature min(0.2 attempt, 0.8), top-p 0.95; seeded per page and attempt), on single-token
        graph steps, up to `slots` at a time; finish() decides on another attempt."""
        d = self.dec
        free = list(range(self.slots))
        act: dict = {}
        graph, members = None, []
        while retry or act:
            changed = False
            while retry and free:
                it, item, attempt = retry.pop(0)
                s = free.pop(0)
                self.retried += 1
                temp = min(0.2 * attempt, 0.8)
                gen = torch.Generator(device="cuda").manual_seed(1_000_003 * (it + 1) + attempt)
                _, h = self._prefill(s, pool.submit(self._prepare, item).result())
                t = self._sample(h[None], [temp], [gen])[0]
                out[it] = [t]
                if t in self.eos or max_new <= 1:
                    ended[it] = t in self.eos
                    free.append(s)
                    finish(it, attempt)
                    continue
                act[it] = {"slot": s, "temp": temp, "gen": gen, "attempt": attempt, "last": t}
                changed = True
            if not act:
                continue
            if changed or graph is None:
                members = sorted(act, key=lambda i: act[i]["slot"])
                seqs = [act[i]["slot"] for i in members]
                graph = self._plain_graphs.get(len(seqs))
                if graph is None:
                    graph = self._plain_graphs[len(seqs)] = d.graph_step(seqs)
                else:
                    graph.rebind(seqs)
            hidden, _ = graph(torch.tensor([act[i]["last"] for i in members], device="cuda"))
            for i in members:
                d.length[act[i]["slot"]] += 1
            toks = self._sample(hidden, [act[i]["temp"] for i in members], [act[i]["gen"] for i in members])
            for i, t in zip(members, toks):
                a = act[i]
                a["last"] = t
                out[i].append(t)
                stop = t in self.eos or len(out[i]) >= max_new
                if not stop and self.stop_on_repeat and len(out[i]) % REPEAT_CHECK_EVERY == 0:
                    stop = looping(i)
                if stop:
                    ended[i] = t in self.eos
                    free.append(a["slot"])
                    del act[i]
                    graph = None
                    finish(i, a["attempt"])
