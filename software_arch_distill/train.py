"""
Full-sequence logits distillation with teacher via API (OpenRouter-style).
- Tries true KL using teacher top_k distributions if tokens map 1:1 to student tokenizer IDs.
- Otherwise falls back to sequence cross-entropy using teacher-generated text as hard labels.
Prereqs:
- Set OPENROUTER_API_KEY in your environment.
- Make sure your OpenRouter/Groq provider supports returning `top_logprobs` / token-level info.
"""

import os
import json
import time
from typing import List, Tuple, Optional, Dict

import requests
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

# -------------------------------
# Config
# -------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", None)
TEACHER_API_URL = os.getenv("TEACHER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
TEACHER_MODEL = "moonshotai/Kimi-K2-Instruct-0905"
# You may need to change TEACHER_API_URL to the exact endpoint your provider uses.
if OPENROUTER_API_KEY is None:
    raise RuntimeError("Set OPENROUTER_API_KEY environment variable before running.")

STUDENT_MODEL = "Qwen/Qwen3-4B-Thinking-2507"  # replace with your student
BATCH_SIZE = 1
MAX_NEW_TOKENS = 128         # how many tokens teacher should generate
TOP_K = 50                   # request top_k logprobs per token from teacher (if supported)
EPOCHS = 3
LR = 1e-5
CACHE_TEACHER = True         # set True to cache teacher responses on disk to avoid repeated API calls
CACHE_DIR = "./teacher_cache"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------------
# Dataset loading (same as yours)
# -------------------------------
file_path = hf_hub_download(
    repo_id="ajibawa-2023/Software-Architecture",
    filename="Software_Architecture_Final.jsonl",
    repo_type="dataset",
    use_auth_token=True
)

examples = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        examples.append({"input": obj.get("input", ""), "output": obj.get("output", "")})

hf_dataset = Dataset.from_list(examples)

# Minimal preprocessing
def extract_fields(example):
    return {"input": example.get("input", ""), "output": example.get("output", "")}

processed_data = hf_dataset.map(extract_fields, remove_columns=hf_dataset.column_names)

# -------------------------------
# Tokenizer & Student model
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, use_fast=False)
student = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL, torch_dtype=torch.float16 if device=="cuda" else None).to(device)
student.eval()  # we'll call .train() near optimizer loop

vocab_size = student.config.vocab_size
print("Student vocab size:", vocab_size)

# -------------------------------
# Dataset class & dataloader
# -------------------------------
class ArchitectureDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        return {
            "input_text": example["input"],
            "output_text": example["output"]
        }

train_dataset = ArchitectureDataset(processed_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------
# Teacher API helper (robust parsing)
# -------------------------------
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

def call_teacher_api(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, top_k: int = TOP_K) -> dict:
    payload = {
        "model": TEACHER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "logprobs": top_k,
        "temperature": 0.0,
        "echo": False,
    }
    resp = requests.post(TEACHER_API_URL, headers=HEADERS, json=payload, timeout=60)
    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Teacher API returned non-JSON: {e}, text: {resp.text}")
    if resp.status_code >= 400:
        raise RuntimeError(f"Teacher API error {resp.status_code}: {data}")
    return data

def parse_teacher_logprobs(resp_json: dict) -> Tuple[str, List[str], Optional[List[Dict[str, float]]]]:
    """
    Attempt to extract:
      - generated_text (str)
      - tokens (list of token strings in generation order)
      - top_logprobs (list of dicts mapping token-string->logprob) OR None
    The exact structure varies by provider; this function tries common variants.
    """
    # typical path: choices[0]['logprobs'] with keys 'tokens' and 'top_logprobs'
    choices = resp_json.get("choices") or resp_json.get("outputs") or []
    if not choices:
        raise RuntimeError(f"No choices/outputs in teacher response: {resp_json}")

    choice = choices[0]

    # many APIs put logprobs under choice['logprobs'] or choice['message' ]['logprobs'] or choice['logprobs']
    logp_container = choice.get("logprobs") or (choice.get("message") or {}).get("logprobs") or None
    if logp_container:
        tokens = logp_container.get("tokens") or logp_container.get("token") or logp_container.get("tokens") or None
        # top_logprobs could be 'top_logprobs' or 'topLogprobs' or 'top_log_probs'
        top_log = logp_container.get("top_logprobs") or logp_container.get("topLogprobs") or logp_container.get("top_log_probs") or None
        # fallback: some providers use 'content' with a list per step
        if tokens is None and isinstance(logp_container.get("content"), list):
            # e.g. logprobs.content = [{'token': 'x', 'logprob': -1.2, 'top_logprobs': {...}}, ...]
            tokens = [s.get("token") for s in logp_container["content"]]
            top_log = [s.get("top_logprobs") for s in logp_container["content"]]
    else:
        # alternative: some providers embed logprobs deeper
        # try choice['message']['content'] as text and hope no token-level logprobs available
        tokens = None
        top_log = None

    # extracted generated text
    gen_text = None
    # prefer choice['message']['content'] or choice['text'] or choice['message']['content'][0]['content'] patterns
    if isinstance(choice.get("message"), dict):
        gen_text = choice["message"].get("content")
    if gen_text is None:
        gen_text = choice.get("text") or choice.get("message") or choice.get("output_text") or None

    # If tokens absent, try to extract token list from other fields
    if tokens is None:
        # Some responses include logprobs at root or under different key; try common paths
        root_logprobs = resp_json.get("logprobs")
        if isinstance(root_logprobs, dict):
            tokens = root_logprobs.get("tokens")
            top_log = root_logprobs.get("top_logprobs")

    # If still no gen_text, try to build from tokens
    if gen_text in (None, "") and tokens:
        gen_text = "".join(tokens)

    return gen_text or "", tokens or [], top_log

# Cache helpers
def cache_filename_for_prompt(prompt: str) -> str:
    safe = str(abs(hash(prompt)))  # simplistic
    return os.path.join(CACHE_DIR, f"teacher_{safe}.json")

def get_teacher_response_cached(prompt: str):
    cache_path = cache_filename_for_prompt(prompt)
    if CACHE_TEACHER and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    data = call_teacher_api(prompt)
    if CACHE_TEACHER:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    return data

# -------------------------------
# Distillation step
# -------------------------------
scaler = GradScaler()
optimizer = optim.Adam(student.parameters(), lr=LR)

def distill_batch(prompt: str) -> float:
    """
    1) Query teacher (cached)
    2) Try to build teacher per-step full distribution (top_k => map to student ids)
       If mapping possible, compute KL across all target timesteps.
    3) Otherwise use CE against the full teacher-generated token sequence.
    Returns loss (float).
    """
    # 1) Call teacher
    resp = get_teacher_response_cached(prompt)
    gen_text, teacher_tokens, teacher_top_logprobs = parse_teacher_logprobs(resp)

    # if we don't have teacher tokens, try to use returned text
    if not gen_text:
        # If the provider returns full text under choices[0]['message']['content'] as text, parse that.
        # Fallback: try other fields
        choices = resp.get("choices") or [resp]
        gen_text = ""
        for ch in choices:
            if isinstance(ch, dict):
                gen_text = ch.get("message", {}).get("content") or ch.get("text") or ch.get("output_text") or gen_text

    if gen_text is None:
        raise RuntimeError("Could not extract generated text from teacher response.")

    # 2) Tokenize prompt and teacher output with student tokenizer
    prompt_enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    prompt_ids = prompt_enc["input_ids"][0]
    prompt_len = prompt_ids.size(0)

    teacher_enc = tokenizer(gen_text, add_special_tokens=False, return_tensors="pt")
    target_ids = teacher_enc["input_ids"][0]
    target_len = target_ids.size(0)

    # Build concatenated input for student: [prompt_ids + target_ids]
    input_ids = torch.cat([prompt_ids, target_ids], dim=0).unsqueeze(0).to(device)  # shape (1, S)
    attention_mask = torch.ones_like(input_ids).to(device)

    # 3) If teacher_top_logprobs available, attempt to map to student token ids per step
    can_do_kl = False
    teacher_probs_tensor = None  # shape (target_len, vocab)
    if teacher_top_logprobs and isinstance(teacher_top_logprobs, list) and len(teacher_top_logprobs) == len(teacher_tokens):
        # try mapping all top_logprobs entries to single student token ids
        mapping_possible = True
        # verify each top_logprobs dict keys map to exactly one id under student tokenizer
        # top_logprobs list: each item is a dict token_string->logprob
        per_step_mapped = []
        for step_dict in teacher_top_logprobs:
            if not isinstance(step_dict, dict):
                mapping_possible = False
                break
            mapped = {}
            for tstr, logp in step_dict.items():
                # encode token string w/ student tokenizer (no adding special tokens)
                # Note: sometimes token strings from API are the exact subtoken pieces, sometimes raw text
                ids = tokenizer.encode(tstr, add_special_tokens=False)
                if len(ids) != 1:
                    # if any token maps to multiple student ids, bail out KL path
                    mapping_possible = False
                    break
                mapped[ids[0]] = float(logp)
            if not mapping_possible:
                break
            per_step_mapped.append(mapped)

        if mapping_possible:
            # OK: build teacher_probs_tensor shape (target_len, vocab_size)
            t = torch.full((target_len, vocab_size), -1e9, dtype=torch.float32)
            for i, step_map in enumerate(per_step_mapped):
                for tok_id, logp in step_map.items():
                    if tok_id < vocab_size:
                        t[i, tok_id] = logp
            # convert logprobs -> probs (in a stable way)
            teacher_probs_tensor = F.softmax(t, dim=-1).to(device)  # (target_len, vocab)
            can_do_kl = True

    # 4) Student forward & loss
    student.train()
    optimizer.zero_grad()

    with autocast(enabled=(device=="cuda")):
        outputs = student(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (1, prompt_len + target_len, vocab)

        # we only care about logits that predict target tokens:
        logits_target = logits[:, prompt_len: prompt_len + target_len, :]  # shape (1, T, V)
        logits_target = logits_target.float()  # ensure float32 for stable kl/ce

        if can_do_kl:
            # compute student log probs
            student_log_probs = F.log_softmax(logits_target, dim=-1)  # (1, T, V)
            # teacher_probs_tensor is (T, V) -> move to batch dimension
            teacher_probs = teacher_probs_tensor.unsqueeze(0)  # (1, T, V)
            # KL divergence per token aggregated
            # F.kl_div expects input=log_prob, target=prob
            loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            mode = "KL"
        else:
            # fallback: cross-entropy against teacher token ids (hard labels)
            # labels shape (1, T)
            labels = target_ids.unsqueeze(0).to(device)
            # flatten logits: (batch*T, V), labels: (batch*T,)
            loss = F.cross_entropy(logits_target.view(-1, logits_target.size(-1)), labels.view(-1))
            mode = "CE"

    # backward and step
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    student.eval()
    return loss.item(), mode, gen_text, target_len

# -------------------------------
# Training loop
# -------------------------------
print("Starting training...")
for epoch in range(EPOCHS):
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        prompt_text = batch["input_text"][0]
        try:
            loss_val, mode_used, gen_text, tlen = distill_batch(prompt_text)
        except Exception as e:
            # Don't crash entire run on one API error; log and continue
            print("Teacher API / processing error for prompt:", prompt_text[:80], "ERR:", e)
            time.sleep(1.0)
            continue
        running_loss += loss_val
        pbar.set_postfix({"loss": f"{loss_val:.4f}", "mode": mode_used})
    avg = (running_loss / len(train_loader)) if len(train_loader) > 0 else 0.0
    print(f"Epoch {epoch+1} avg loss: {avg:.4f}")

# -------------------------------
# Save student
# -------------------------------
outdir = "./qwen3-4b-kd-finetuned"
os.makedirs(outdir, exist_ok=True)
student.save_pretrained(outdir)
tokenizer.save_pretrained(outdir)
print("Saved student model to", outdir)
