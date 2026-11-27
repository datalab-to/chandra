"""Chandra OCR on chandra_cuda's engine: pages in, raw model text out, as chandra's HF backend.

transformers provides the processor, the vision tower (on the GPU; its language model stays on
the CPU, unused) and the (t, h, w) positions; the engine runs the language model's prefill and
decodes up to `slots` pages at a time, with MTP-3 and CUDA graphs by default. A page leaves its
slot at an end token or after max_output_tokens; free slots are refilled from the queue at once,
and new graphs are captured whenever the set of pages changes.

    gen = CudaGenerator()
    results = gen.generate(batch)        # chandra.model.schema.GenerationResult per page
"""

from __future__ import annotations

import glob
import os

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
                 draft_vocab: int = 32768, max_output_tokens: int | None = None, stop_on_repeat: bool = True):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from chandra.settings import settings

        name = checkpoint or settings.MODEL_CHECKPOINT
        self.max_output_tokens = max_output_tokens or settings.MAX_OUTPUT_TOKENS
        self.processor = AutoProcessor.from_pretrained(name)
        self.hf = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16,
                                                              device_map={"": "cpu"}).eval()
        self.hf.model.visual.to("cuda")
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

    @torch.no_grad()
    def _prefill(self, slot: int, item):
        """A page into a slot. Returns (first token, final-norm hidden state at the last position)."""
        from chandra.model.hf import process_batch_element

        inputs = self.processor.apply_chat_template([[process_batch_element(item)]], tokenize=True,
                                                    add_generation_prompt=True, return_dict=True,
                                                    return_tensors="pt")
        ids = inputs["input_ids"].to("cuda")
        n = ids.shape[1]
        if n > PAGE_TOKENS_MAX:
            raise ValueError(f"prompt of {n} tokens > {PAGE_TOKENS_MAX}")
        thw = inputs["image_grid_thw"].to("cuda")
        feats = _flat(self.hf.model.get_image_features(inputs["pixel_values"].to("cuda", torch.bfloat16), thw))
        emb = self.dec.embed[ids[0]].clone()
        emb[ids[0] == self.img_tok] = feats.to(emb.dtype)
        pos, delta = self.hf.model.get_rope_index(ids, inputs["mm_token_type_ids"].to("cuda"), thw, None,
                                                  inputs["attention_mask"].to("cuda"))
        pos3 = pos[:, 0].transpose(0, 1).contiguous()
        hidden, first = self.dec.prefill(slot, emb, pos3, int(delta[0]))
        if self.mtp:
            nxt = torch.cat([ids[0, 1:], first.view(1)])
            self.dec.mtp_prefill(slot, nxt[: n - 1], hidden[: n - 1], pos3[: n - 1])
        return int(first), hidden[-1].clone()

    @torch.no_grad()
    def generate(self, batch, max_output_tokens: int | None = None, **kwargs):
        """Pages (chandra BatchInputItem) in, GenerationResult out (raw text, token count)."""
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
        queue = list(range(len(batch)))
        out: dict[int, list[int]] = {}
        slot_of: dict[int, int] = {}  # item -> slot, for pages still decoding
        free = list(range(self.slots))
        last: dict[int, int] = {}
        drafts: dict[int, torch.Tensor] = {}
        graph, members = None, []
        while queue or slot_of:
            if queue and free:  # admit pages into every free slot
                new = []
                while queue and free:
                    it, s = queue.pop(0), free.pop(0)
                    first, h = self._prefill(s, batch[it])
                    out[it] = [first]
                    if first in self.eos or max_new <= 1:
                        free.append(s)
                        continue
                    slot_of[it], last[it] = s, first
                    new.append((it, h))
                if self.mtp and new:
                    dr = d.first_drafts(torch.tensor([last[it] for it, _ in new], device="cuda"),
                                        torch.stack([h for _, h in new]), [slot_of[it] for it, _ in new])
                    for j, (it, _) in enumerate(new):
                        drafts[it] = dr[j].clone()
                graph = None
            if not slot_of:
                continue
            if graph is None:  # a new set of pages: new graphs
                members = sorted(slot_of, key=lambda it: slot_of[it])
                seqs = [slot_of[it] for it in members]
                graph = d.spec_graph(seqs) if self.mtp else d.graph_step(seqs)
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
            for j, it in enumerate(members):
                for t in emit[j]:
                    out[it].append(t)
                    stop = t in self.eos or len(out[it]) >= max_new
                    if not stop and self.stop_on_repeat and len(out[it]) % REPEAT_CHECK_EVERY == 0:
                        stop = looping(it)
                    if stop:
                        free.append(slot_of.pop(it))
                        graph = None
                        break
        results = []
        for i in range(len(batch)):
            ids = [t for t in out.get(i, []) if t not in self.eos]
            text = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            results.append(GenerationResult(raw=text, token_count=len(out.get(i, [])), error=False))
        return results
