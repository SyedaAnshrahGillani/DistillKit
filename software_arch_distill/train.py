"""
Full-sequence logits distillation with teacher via API (OpenRouter-style).
Uses Kimi-K2-0905 as teacher. If teacher returns per-token top-k logprobs and mapping to student tokenizer works, uses KL divergence over the sequence; else falls back to cross-entropy (hard labels).
Prereqs:
- OPENROUTER_API_KEY environment variable set.
- Teacher API supports returning token-level info (top_logprobs or logprobs).
"""

import os
import json
import time
from typing import List, Optional, Dict, Tuple

import requests
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

# -------------------------------
# Config
# -------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", None)
if OPENROUTER_API_KEY is None:
    raise RuntimeError("Set OPENROUTER_API_KEY environment variable before running.")

TEACHER_API_URL = os.getenv("TEACHER_API_URL", "https://api.openrouter.ai/v1/chat/completions")
TEACHER_MODEL = "openrouter/moonshotai/kimi-k2-0905"

STUDENT_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
BATCH_SIZE = 1
MAX_NEW_TOKENS = 128
TOP_K = 50
EPOCHS = 3
LR = 1e-5
CACHE_TEACHER = True
CACHE_DIR = "./teacher_cache"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------------
# Dataset loading
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

hf_dataset = HFDataset.from_list(examples)

def extract_fields(example):
    return {"input": example.get("input", ""), "output": example.get("output", "")}

processed_data = hf_dataset.map(extract_fields, remove_columns=hf_dataset.column_names)

# -------------------------------
# Tokenizer & Student
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, use_fast=False)
student = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

vocab_size = student.config.vocab_size
print(f"Student vocab size: {vocab_size}")

# -------------------------------
# DataLoader
# -------------------------------
class ArchitectureDataset(Dataset):
    def __init__(self, hf_data, tokenizer, max_length=512):
        self.data = hf_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {
            "input_text": ex["input"],
            "output_text": ex["output"]
        }

train_dataset = ArchitectureDataset(processed_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------
# Teacher API helper
# -------------------------------
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

def call_teacher_api(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, top_k: int = TOP_K) -> Dict:
    payload = {
        "model": TEACHER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "logprobs": top_k,           # or use "top_k": top_k if needed
        "temperature": 0.0,
        "echo": False,
    }
    resp = requests.post(TEACHER_API_URL, headers=HEADERS, json=payload, timeout=60)
    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Teacher API returned non-JSON: {e}, text: {resp.text}")
    if resp.status_code != 200:
        raise RuntimeError(f"Teacher API error {resp.status_code}: {data}")
    return data

def parse_teacher_logprobs(resp_json: Dict) -> Tuple[str, List[str], Optional[List[Dict[str, float]]]]:
    choices = resp_json.get("choices") or resp_json.get("outputs") or []
    if not choices:
        raise RuntimeError(f"No choices in teacher response: {resp_json}")
    choice = choices[0]

    # Attempt extraction
    logp_container = None
    if "logprobs" in choice:
        logp_container = choice["logprobs"]
    elif "message" in choice and isinstance(choice["message"], dict) and "logprobs" in choice["message"]:
        logp_container = choice["message"]["logprobs"]

    tokens = None
    top_log = None

    if logp_container:
        tokens = logp_container.get("tokens") or logp_container.get("token") or None
        top_log = logp_container.get("top_logprobs") or logp_container.get("topLogprobs") or logp_container.get("top_log_probs") or None

        # fallback: some providers wrap in "content" list
        if (tokens is None or top_log is None) and "content" in logp_container and isinstance(logp_container["content"], list):
            content_list = logp_container["content"]
            tokens = [step.get("token") for step in content_list]
            top_log = [step.get("top_logprobs") for step in content_list]

    # Extract gen_text
    gen_text = None
    if "message" in choice and isinstance(choice["message"], dict):
        gen_text = choice["message"].get("content")
    if gen_text is None and "text" in choice:
        gen_text = choice["text"]
    if gen_text is None and "output_text" in choice:
        gen_text = choice["output_text"]

    # If still no gen_text, build from tokens
    if (gen_text is None or gen_text == "") and tokens:
        gen_text = "".join(tokens)

    return gen_text or "", tokens or [], top_log

# -------------------------------
# Caching
# -------------------------------
def cache_filename_for_prompt(prompt: str) -> str:
    safe = str(abs(hash(prompt)))
    return os.path.join(CACHE_DIR, f"teacher_{safe}.json")

def get_teacher_response_cached(prompt: str) -> Dict:
    path = cache_filename_for_prompt(prompt)
    if CACHE_TEACHER and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    data = call_teacher_api(prompt)
    if CACHE_TEACHER:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    return data

# -------------------------------
# Distillation
# -------------------------------
scaler = GradScaler()
optimizer = optim.Adam(student.parameters(), lr=LR)

def distill_batch(prompt: str) -> Tuple[float, str, str, int]:
    resp = get_teacher_response_cached(prompt)
    gen_text, teacher_tokens, teacher_top_logprobs = parse_teacher_logprobs(resp)

    if not gen_text:
        raise RuntimeError(f"Could not extract generated text for prompt: {prompt}")

    # Tokenize
    prompt_enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    prompt_ids = prompt_enc["input_ids"][0]
    prompt_len = prompt_ids.size(0)

    teacher_enc = tokenizer(gen_text, add_special_tokens=False, return_tensors="pt")
    target_ids = teacher_enc["input_ids"][0]
    target_len = target_ids.size(0)

    # Limit target length so we do not blow up memory
    if target_len > MAX_NEW_TOKENS:
        target_ids = target_ids[:MAX_NEW_TOKENS]
        target_len = MAX_NEW_TOKENS
        gen_text = tokenizer.decode(target_ids, skip_special_tokens=True)

    input_ids = torch.cat([prompt_ids, target_ids], dim=0).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Prepare teacher distribution if possible
    can_do_kl = False
    teacher_probs_tensor = None

    if teacher_top_logprobs and isinstance(teacher_top_logprobs, list) and len(teacher_top_logprobs) == len(teacher_tokens):
        per_step_mapped: List[Dict[int, float]] = []
        mapping_possible = True

        for step_dict in teacher_top_logprobs[:target_len]:
            if not isinstance(step_dict, dict):
                mapping_possible = False
                break
            mapped = {}
            for tstr, logp in step_dict.items():
                # map tstr to student token ids
                ids = tokenizer.encode(tstr, add_special_tokens=False)
                if len(ids) != 1:
                    mapping_possible = False
                    break
                mapped_id = ids[0]
                mapped[mapped_id] = float(logp)
            if not mapping_possible:
                break
            per_step_mapped.append(mapped)

        if mapping_possible:
            t = torch.full((target_len, vocab_size), -1e9, dtype=torch.float32)
            for i, step_map in enumerate(per_step_mapped):
                for tok_id, logp in step_map.items():
                    if tok_id < vocab_size:
                        t[i, tok_id] = logp
            teacher_probs_tensor = F.softmax(t, dim=-1).to(device)
            can_do_kl = True

    # Student forward + loss
    student.train()
    optimizer.zero_grad()

    with autocast(enabled=(device == "cuda")):
        outputs = student(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape (1, prompt_len + target_len, vocab_size)

        logits_target = logits[:, prompt_len: prompt_len + target_len, :].float()  # shape (1, T, V)

        if can_do_kl:
            student_log_probs = F.log_softmax(logits_target, dim=-1)
            teacher_probs = teacher_probs_tensor.unsqueeze(0)  # shape (1, T, V)
            loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            mode = "KL"
        else:
            labels = target_ids.unsqueeze(0).to(device)
            loss = F.cross_entropy(logits_target.view(-1, logits_target.size(-1)), labels.view(-1))
            mode = "CE"

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
            print("Teacher API / processing error for prompt:", prompt_text[:80], "ERR:", e)
            time.sleep(1.0)
            continue
        running_loss += loss_val
        pbar.set_postfix({"loss": f"{loss_val:.4f}", "mode": mode_used})
    avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

# -------------------------------
# Save student
# -------------------------------
outdir = "./qwen3-4b-kd-finetuned"
os.makedirs(outdir, exist_ok=True)
student.save_pretrained(outdir)
tokenizer.save_pretrained(outdir)
print(f"Saved student model to {outdir}")
