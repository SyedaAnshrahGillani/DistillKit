"""
Full-sequence logits distillation using OpenRouter's Kimi-K2-0905 model.
Uses KL divergence if teacher returns token-level logprobs & top_logprobs, mapping to student tokenizer works; otherwise falls back to cross-entropy.
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

TEACHER_API_URL = "https://openrouter.ai/api/v1/chat/completions"  # Fixed: removed env variable for standard endpoint
TEACHER_MODEL = "moonshotai/kimi-k2-0905"  # model name per OpenRouter

STUDENT_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
BATCH_SIZE = 1
MAX_NEW_TOKENS = 128
TOP_K = 5   # Reduced from 20 to 5 for better compatibility
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
# Tokenizer & Student model
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
    def __init__(self, hf_data, tokenizer, max_length: int = 512):
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
    "Content-Type": "application/json",
    # Added recommended OpenRouter headers
    "HTTP-Referer": "https://your-site.com",  # Replace with your actual site
    "X-Title": "Logits-Distillation-Training"  # Your app name
}

def call_teacher_api(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, top_k: int = TOP_K, retry: int = 3) -> Dict:
    # Fixed payload according to OpenRouter docs
    payload = {
        "model": TEACHER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "logprobs": True,            # CRITICAL: Must be True when using top_logprobs
        "top_logprobs": top_k,       # integer ≤ 20 (reduced to 5 for compatibility)
        "stream": False
    }
    
    for attempt in range(retry):
        try:
            print(f"[Teacher API] Making request (attempt {attempt+1}/{retry})")
            resp = requests.post(TEACHER_API_URL, headers=HEADERS, json=payload, timeout=60)
            
            # Better error handling
            if resp.status_code == 200:
                data = resp.json()
                print(f"[Teacher API] Success! Response keys: {list(data.keys())}")
                return data
            elif resp.status_code == 401:
                print(f"[Teacher API] Authentication failed. Check your API key.")
                raise RuntimeError(f"Authentication failed: {resp.text}")
            elif resp.status_code == 429:
                print(f"[Teacher API] Rate limited. Waiting longer...")
                time.sleep(10 * (attempt + 1))
                continue
            else:
                try:
                    error_data = resp.json()
                    print(f"[Teacher API] HTTP {resp.status_code}: {error_data}")
                except:
                    print(f"[Teacher API] HTTP {resp.status_code}: {resp.text}")
                
        except requests.exceptions.ConnectionError as e:
            print(f"[Teacher API] Connection error (attempt {attempt+1}/{retry}): {e}")
            print("This might be a network connectivity issue. Check your internet connection.")
            time.sleep(5)
        except requests.exceptions.Timeout as e:
            print(f"[Teacher API] Timeout error (attempt {attempt+1}/{retry}): {e}")
            time.sleep(2)
        except Exception as e:
            print(f"[Teacher API] Unexpected error (attempt {attempt+1}/{retry}): {e}")
            time.sleep(2)
    
    raise RuntimeError(f"Teacher API failed after {retry} attempts for prompt: {prompt[:100]}")

def parse_teacher_logprobs(resp_json: Dict) -> Tuple[str, List[str], Optional[List[Dict[str, float]]]]:
    """Parse logprobs from OpenRouter response format"""
    choices = resp_json.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices in teacher response: {resp_json}")
    choice = choices[0]

    # Get generated text
    gen_text = None
    if "message" in choice and isinstance(choice["message"], dict):
        gen_text = choice["message"].get("content")
    if gen_text is None and "text" in choice:
        gen_text = choice["text"]

    # Parse logprobs structure - OpenRouter follows OpenAI format
    logprobs_data = None
    if "logprobs" in choice:
        logprobs_data = choice["logprobs"]
    elif "message" in choice and isinstance(choice["message"], dict) and "logprobs" in choice["message"]:
        logprobs_data = choice["message"]["logprobs"]

    tokens = []
    top_logprobs = None
    
    if logprobs_data:
        # OpenAI/OpenRouter format: logprobs.content is a list of token info
        if "content" in logprobs_data and isinstance(logprobs_data["content"], list):
            content_list = logprobs_data["content"]
            tokens = [item.get("token", "") for item in content_list]
            top_logprobs = [item.get("top_logprobs", {}) for item in content_list]
        else:
            print(f"[Parser] Unexpected logprobs format: {logprobs_data}")

    print(f"[Parser] Generated text length: {len(gen_text) if gen_text else 0}")
    print(f"[Parser] Tokens found: {len(tokens)}")
    print(f"[Parser] Top logprobs available: {top_logprobs is not None}")
    
    return gen_text or "", tokens, top_logprobs

# -------------------------------
# Caching
# -------------------------------
def cache_filename_for_prompt(prompt: str) -> str:
    safe = str(abs(hash(prompt)))
    return os.path.join(CACHE_DIR, f"teacher_{safe}.json")

def get_teacher_response_cached(prompt: str) -> Dict:
    path = cache_filename_for_prompt(prompt)
    if CACHE_TEACHER and os.path.exists(path):
        print(f"[Cache] Loading cached response for prompt")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    print(f"[Cache] No cache found, making API call")
    data = call_teacher_api(prompt)
    
    if CACHE_TEACHER:
        print(f"[Cache] Saving response to cache")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    return data

# -------------------------------
# Distillation setup
# -------------------------------
scaler = GradScaler()
optimizer = optim.Adam(student.parameters(), lr=LR)

def distill_batch(prompt: str) -> Tuple[float, str, str, int]:
    # Get teacher response
    resp = get_teacher_response_cached(prompt)
    gen_text, teacher_tokens, teacher_top_logprobs = parse_teacher_logprobs(resp)

    if not gen_text:
        raise RuntimeError(f"Could not extract generated text for prompt: {prompt[:100]}")

    # Tokenize prompt and target
    prompt_enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    prompt_ids = prompt_enc["input_ids"][0]
    prompt_len = prompt_ids.size(0)

    teacher_enc = tokenizer(gen_text, add_special_tokens=False, return_tensors="pt")
    target_ids = teacher_enc["input_ids"][0]
    target_len = target_ids.size(0)

    # Truncate if too long
    if target_len > MAX_NEW_TOKENS:
        target_ids = target_ids[:MAX_NEW_TOKENS]
        target_len = MAX_NEW_TOKENS
        gen_text = tokenizer.decode(target_ids, skip_special_tokens=True)

    # Prepare input for student
    input_ids = torch.cat([prompt_ids, target_ids], dim=0).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Try to create teacher probability tensor
    can_do_kl = False
    teacher_probs_tensor = None

    if teacher_top_logprobs and len(teacher_top_logprobs) > 0:
        print(f"[KL] Attempting to map {len(teacher_top_logprobs)} teacher logprob dicts")
        
        # Limit to target_len to match our truncated sequence
        teacher_top_logprobs = teacher_top_logprobs[:target_len]
        
        per_step_mapped: List[Dict[int, float]] = []
        mapping_possible = True
        
        for i, step_dict in enumerate(teacher_top_logprobs):
            if not isinstance(step_dict, dict) or len(step_dict) == 0:
                print(f"[KL] Step {i}: Invalid or empty logprobs dict")
                mapping_possible = False
                break
            
            mapped = {}
            for token_str, logprob in step_dict.items():
                try:
                    # Try to encode the token string to get token ID
                    token_ids = tokenizer.encode(token_str, add_special_tokens=False)
                    if len(token_ids) == 1:
                        token_id = token_ids[0]
                        if token_id < vocab_size:
                            mapped[token_id] = float(logprob)
                except Exception as e:
                    print(f"[KL] Failed to map token '{token_str}': {e}")
                    continue
            
            if len(mapped) == 0:
                print(f"[KL] Step {i}: No tokens successfully mapped")
                mapping_possible = False
                break
            
            per_step_mapped.append(mapped)
        
        if mapping_possible and len(per_step_mapped) == target_len:
            # Create teacher probability tensor
            teacher_logits = torch.full((target_len, vocab_size), -1e9, dtype=torch.float32)
            
            for i, step_map in enumerate(per_step_mapped):
                for token_id, logprob in step_map.items():
                    teacher_logits[i, token_id] = logprob
            
            teacher_probs_tensor = F.softmax(teacher_logits, dim=-1).to(device)
            can_do_kl = True
            print(f"[KL] Successfully created teacher probability tensor: {teacher_probs_tensor.shape}")
        else:
            print(f"[KL] Failed to create teacher probability tensor")

    # Training step
    student.train()
    optimizer.zero_grad()

    with autocast(enabled=(device == "cuda")):
        outputs = student(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Extract logits for the target sequence
        logits_target = logits[:, prompt_len: prompt_len + target_len, :].float()

        if can_do_kl:
            # Use KL divergence with teacher probabilities
            student_log_probs = F.log_softmax(logits_target, dim=-1)
            teacher_probs = teacher_probs_tensor.unsqueeze(0)
            loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            mode = "KL"
        else:
            # Fallback to cross-entropy with target tokens
            labels = target_ids.unsqueeze(0).to(device)
            loss = F.cross_entropy(logits_target.view(-1, logits_target.size(-1)), labels.view(-1))
            mode = "CE"

    # Backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    student.eval()
    return loss.item(), mode, gen_text[:100], target_len

# -------------------------------
# Training loop
# -------------------------------
print("Starting training...")

# Test API connectivity first
print("Testing OpenRouter API connectivity...")
try:
    test_resp = call_teacher_api("Hello, this is a test.")
    print("✓ API connectivity test successful!")
except Exception as e:
    print(f"✗ API connectivity test failed: {e}")
    print("Please check:")
    print("1. Your OPENROUTER_API_KEY environment variable is set correctly")
    print("2. You have credits in your OpenRouter account")
    print("3. Your internet connection is working")
    print("4. The model 'moonshotai/kimi-k2-0905' is available in your region")
    exit(1)

for epoch in range(EPOCHS):
    running_loss = 0.0
    successful_batches = 0
    kl_batches = 0
    ce_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch_idx, batch in enumerate(pbar):
        prompt_text = batch["input_text"][0]
        
        # Skip very short prompts
        if len(prompt_text.strip()) < 10:
            continue
            
        try:
            loss_val, mode_used, gen_preview, tlen = distill_batch(prompt_text)
            running_loss += loss_val
            successful_batches += 1
            
            if mode_used == "KL":
                kl_batches += 1
            else:
                ce_batches += 1
            
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}", 
                "mode": mode_used,
                "KL": kl_batches,
                "CE": ce_batches,
                "gen": gen_preview[:30] + "..."
            })
            
        except Exception as e:
            print(f"\nError processing batch {batch_idx}: {e}")
            print(f"Prompt: {prompt_text[:100]}...")
            continue
    
    avg_loss = running_loss / successful_batches if successful_batches > 0 else float('inf')
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  - Successful batches: {successful_batches}")
    print(f"  - KL divergence batches: {kl_batches}")
    print(f"  - Cross-entropy batches: {ce_batches}")
    print(f"  - Average loss: {avg_loss:.4f}")

# -------------------------------
# Save student
# -------------------------------
outdir = "./qwen3-4b-kd-finetuned"
os.makedirs(outdir, exist_ok=True)
student.save_pretrained(outdir)
tokenizer.save_pretrained(outdir)
print(f"Saved student model to {outdir}")
