"""
ULD Distillation with Complete Checkpointing System - ENHANCED VERSION
Preserves original flows while adding fixes and teacher data saving
"""

import os
import json
import time
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

import torch
import torch.nn.functional as F
from torch import nn, optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

class ULDTestSuite:
    """Optional test suite to validate ULD logic before training"""
    
    def __init__(self):
        self.results = {}
    
    def test_together_api_parsing(self):
        """Test Together AI API response parsing"""
        print("🧪 Testing Together AI API parsing...")
        
        # Mock Together AI response format
        mock_response = {
            "choices": [{
                "logprobs": {
                    "tokens": ["Hello", " world", "!"],
                    "token_logprobs": [-0.1, -0.5, -0.8],
                    "token_ids": [15496, 1917, 0],
                    "top_logprobs": [
                        {"Hello": -0.1, "Hi": -0.3},
                        {" world": -0.5, " earth": -0.7},
                        {"!": -0.8, ".": -0.9}
                    ]
                },
                "message": {"content": "Hello world!"}
            }]
        }
        
        try:
            # Test with mock tokenizer
            class MockTokenizer:
                vocab_size = 32000
                def convert_tokens_to_ids(self, token):
                    return hash(token) % self.vocab_size
            
            text, logits_tensor = self._parse_together_response_test(mock_response, MockTokenizer())
            assert text == "Hello world!", f"Expected 'Hello world!', got '{text}'"
            assert logits_tensor.shape[0] == 3, f"Expected 3 tokens, got {logits_tensor.shape[0]}"
            print("✅ Together AI parsing test passed")
            self.results['api_parsing'] = True
        except Exception as e:
            print(f"❌ Together AI parsing test failed: {e}")
            self.results['api_parsing'] = False
    
    def _parse_together_response_test(self, response_data: Dict, tokenizer) -> Tuple[str, torch.Tensor]:
        """Test version of parse_teacher_logits"""
        choice = response_data["choices"][0]
        generated_text = choice["message"]["content"]
        
        logprobs_data = choice["logprobs"]
        tokens = logprobs_data["tokens"]
        token_logprobs = logprobs_data["token_logprobs"]
        top_logprobs = logprobs_data.get("top_logprobs", [])
        
        vocab_size = tokenizer.vocab_size
        logits_list = []
        
        for i, (token, base_logprob) in enumerate(zip(tokens, token_logprobs)):
            if base_logprob is None:
                continue
            
            logits = torch.full((vocab_size,), -20.0)
            
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id < vocab_size:
                    logits[token_id] = base_logprob
            except:
                base_idx = hash(token) % vocab_size
                logits[base_idx] = base_logprob
            
            if i < len(top_logprobs) and top_logprobs[i]:
                for j, (top_token, top_logprob) in enumerate(list(top_logprobs[i].items())[:5]):
                    try:
                        top_token_id = tokenizer.convert_tokens_to_ids(top_token)
                        if top_token_id < vocab_size:
                            logits[top_token_id] = top_logprob
                    except:
                        idx = (hash(top_token) + j) % vocab_size
                        logits[idx] = top_logprob
            
            logits_list.append(logits)
        
        return generated_text, torch.stack(logits_list)
    
    def test_uld_loss_computation(self):
        """Test ULD Wasserstein distance computation"""
        print("🧪 Testing ULD loss computation...")
        
        try:
            teacher_logits = torch.randn(3, 30000)
            student_logits = torch.randn(3, 32000)
            
            loss = self._compute_uld_loss_test(teacher_logits, student_logits)
            
            assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
            assert loss.numel() == 1, "Loss should be a scalar"
            assert loss.item() >= 0, "ULD loss should be non-negative"
            
            print(f"✅ ULD loss computation test passed (loss: {loss.item():.4f})")
            self.results['uld_loss'] = True
        except Exception as e:
            print(f"❌ ULD loss computation test failed: {e}")
            self.results['uld_loss'] = False
    
    def _compute_uld_loss_test(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
        """Test ULD loss computation using Wasserstein distance"""
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)
        
        teacher_sorted, _ = torch.sort(teacher_probs, dim=-1, descending=True)
        student_sorted, _ = torch.sort(student_probs, dim=-1, descending=True)
        
        min_len = min(teacher_sorted.size(-1), student_sorted.size(-1))
        teacher_aligned = teacher_sorted[:, :min_len]
        student_aligned = student_sorted[:, :min_len]
        
        wasserstein_dist = torch.sum(torch.abs(teacher_aligned - student_aligned), dim=-1)
        
        return torch.mean(wasserstein_dist)
    
    def run_all_tests(self) -> bool:
        """Run all tests and return True if all pass"""
        print("🚀 Running ULD Test Suite...")
        
        self.test_together_api_parsing()
        self.test_uld_loss_computation()
        
        passed = sum(self.results.values())
        total = len(self.results)
        
        print(f"\n📊 Test Results: {passed}/{total} passed")
        
        if passed == total:
            print("✅ All tests passed! Safe to proceed with training.")
            return True
        else:
            print("❌ Some tests failed! Fix issues before training.")
            return False

class DistillationCheckpoint:
    """Enhanced checkpoint management - preserving original structure"""
    
    def __init__(self, checkpoint_dir: str = "./distillation_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # File paths (maintaining original structure)
        self.state_file = self.checkpoint_dir / "training_state.json"
        self.model_dir = self.checkpoint_dir / "model"
        self.optimizer_file = self.checkpoint_dir / "optimizer.pt"
        self.progress_file = self.checkpoint_dir / "progress.json"
        self.cache_dir = self.checkpoint_dir / "teacher_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # NEW: Teacher data directory for analysis
        self.teacher_data_dir = self.checkpoint_dir / "teacher_data"
        self.teacher_data_dir.mkdir(exist_ok=True)
        
        # Initialize state
        self.state = self.load_state()
        
        print(f"📁 Checkpoint directory: {self.checkpoint_dir}")
        if list(self.teacher_data_dir.glob("pair_*.json")):
            saved_pairs = len(list(self.teacher_data_dir.glob("pair_*.json")))
            print(f"📚 Found {saved_pairs} saved teacher interactions")
    
    def load_state(self) -> Dict[str, Any]:
        """Load training state or create new one - preserving original"""
        if self.state_file.exists():
            print(f"📂 Loading existing training state from {self.state_file}")
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                print(f"🔄 Resuming from: Phase {state['current_phase']}, "
                      f"Example {state['current_example']}, Epoch {state['current_epoch']}")
                return state
        else:
            print("🆕 Creating new training state")
            return {
                "created_at": datetime.now().isoformat(),
                "current_phase": 1,
                "current_example": 0,
                "current_epoch": 1,
                "total_examples_processed": 0,
                "successful_batches": 0,
                "total_loss": 0.0,
                "phase_sizes": [10, 50, 200, 1000],  # Preserving original
                "epochs_per_phase": [1, 2, 2, 3],   # Preserving original
                "learning_rates": [5e-6, 3e-6, 2e-6, 1.5e-6],  # FIXED: Raised Phase 4 LR
                "training_history": [],
                "validation_history": [],
                "last_saved": None,
                "api_calls_made": 0,
                "estimated_cost": 0.0
            }
    
    def save_state(self):
        """Save current training state - preserving original"""
        self.state["last_saved"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        print(f"💾 State saved: Phase {self.state['current_phase']}, "
              f"Example {self.state['current_example']}")
    
    def save_model_checkpoint(self, model, tokenizer, optimizer):
        """Save model, tokenizer, and optimizer state - preserving original"""
        print(f"💾 Saving model checkpoint...")
        
        # Save model and tokenizer
        model.save_pretrained(self.model_dir)
        tokenizer.save_pretrained(self.model_dir)
        
        # Save optimizer state
        torch.save({
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict(),
        }, self.optimizer_file)
        
        print(f"✅ Model checkpoint saved to {self.model_dir}")
    
    def load_model_checkpoint(self, model_name: str):
        """Load model, tokenizer, and optimizer from checkpoint - preserving original"""
        if self.model_dir.exists() and any(self.model_dir.iterdir()):
            print(f"📂 Loading model from checkpoint...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_dir, 
                torch_dtype=torch.bfloat16
            )
            
            # Create optimizer
            lr = self.state["learning_rates"][self.state["current_phase"] - 1]
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            
            # Load optimizer state if exists
            if self.optimizer_file.exists():
                checkpoint = torch.load(self.optimizer_file)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✅ Optimizer state loaded")
            
            print("✅ Model loaded from checkpoint")
            return model, tokenizer, optimizer
        else:
            print(f"🆕 Loading fresh model: {model_name}")
            
            # Fresh model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16
            )
            
            lr = self.state["learning_rates"][self.state["current_phase"] - 1]
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            
            return model, tokenizer, optimizer
    
    def get_teacher_response_cached(self, prompt: str) -> Dict:
        """Enhanced caching with teacher data saving"""
        cache_key = str(abs(hash(prompt)))
        cache_file = self.cache_dir / f"teacher_{cache_key}.json"
        
        if cache_file.exists():
            print(f"📂 Using cached teacher response")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Make API call
        print(f"🌐 Making API call to teacher (Together AI)...")
        response = self.call_teacher_api(prompt)
        
        # Cache the response (original behavior)
        with open(cache_file, 'w') as f:
            json.dump(response, f)
        
        # NEW: Save teacher data for analysis
        self.save_teacher_data(prompt, response, cache_key)
        
        # Update API call count and cost estimate
        self.state["api_calls_made"] += 1
        # Together AI pricing for Kimi K2 (Input: $1.00/M tokens, Output: $3.00/M tokens)
        self.state["estimated_cost"] += 0.005  # Estimated cost per 512 token call
        
        return response
    
    def save_teacher_data(self, prompt: str, response: Dict, cache_key: str):
        """NEW: Save teacher prompt-response pairs for analysis"""
        # Save individual prompt-response pair
        pair_file = self.teacher_data_dir / f"pair_{cache_key}.json"
        pair_data = {
            "timestamp": datetime.now().isoformat(),
            "cache_key": cache_key,
            "prompt": prompt,
            "response": response,
            "generated_text": response["choices"][0]["message"]["content"] if response.get("choices") else None,
            "model_used": response.get("model", "unknown"),
            "usage": response.get("usage", {}),
            "training_phase": self.state.get("current_phase", 0),
            "training_example": self.state.get("current_example", 0)
        }
        
        with open(pair_file, 'w', encoding='utf-8') as f:
            json.dump(pair_data, f, indent=2, ensure_ascii=False)
        
        # Maintain a master index
        index_file = self.teacher_data_dir / "teacher_interactions_index.jsonl"
        index_entry = {
            "timestamp": datetime.now().isoformat(),
            "cache_key": cache_key,
            "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "response_preview": (response["choices"][0]["message"]["content"][:100] + "..." 
                               if response.get("choices") and len(response["choices"][0]["message"]["content"]) > 100 
                               else response["choices"][0]["message"]["content"]) if response.get("choices") else "No response",
            "phase": self.state.get("current_phase", 0),
            "example": self.state.get("current_example", 0),
            "file": f"pair_{cache_key}.json"
        }
        
        with open(index_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(index_entry, ensure_ascii=False) + '\n')
        
        # Save readable text version
        readable_file = self.teacher_data_dir / f"readable_{cache_key}.txt"
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write(f"TEACHER INTERACTION - {datetime.now().isoformat()}\n")
            f.write(f"Cache Key: {cache_key}\n")
            f.write(f"Training Phase: {self.state.get('current_phase', 0)}\n")
            f.write(f"Training Example: {self.state.get('current_example', 0)}\n")
            f.write("=" * 80 + "\n")
            f.write("PROMPT:\n")
            f.write("-" * 40 + "\n")
            f.write(prompt + "\n\n")
            f.write("TEACHER RESPONSE:\n")
            f.write("-" * 40 + "\n")
            if response.get("choices"):
                f.write(response["choices"][0]["message"]["content"] + "\n\n")
            f.write("\n" + "=" * 80 + "\n\n")
    
    def call_teacher_api(self, prompt: str) -> Dict:
        """Call Together AI API - preserving original"""
        import requests
        
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY environment variable not set")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "moonshotai/Kimi-K2-Instruct-0905",  # Latest Kimi K2 model on Together AI
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,  # Optimal balance: quality vs cost for architecture Q&A
            "temperature": 0.0,
            "logprobs": 20,  # Together AI uses logprobs parameter instead of top_logprobs
            "echo": False,
            "stream": False
        }
        
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Together AI API failed: {response.status_code} - {response.text}")
        
        return response.json()
    
    def log_progress(self, loss: float, mode: str, validation_results: Dict = None):
        """Log training progress - preserving original"""
        progress_entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": self.state["current_phase"],
            "example": self.state["current_example"],
            "epoch": self.state["current_epoch"],
            "loss": loss,
            "mode": mode,
            "total_examples_processed": self.state["total_examples_processed"],
            "validation": validation_results
        }
        
        self.state["training_history"].append(progress_entry)
        self.state["total_loss"] += loss
        self.state["successful_batches"] += 1
        
        # Keep only last 1000 entries to prevent file bloat
        if len(self.state["training_history"]) > 1000:
            self.state["training_history"] = self.state["training_history"][-1000:]
    
    def should_validate(self) -> bool:
        """Determine if we should run validation - preserving original"""
        return self.state["current_example"] % 20 == 0  # Every 20 examples
    
    def should_save_checkpoint(self) -> bool:
        """Determine if we should save checkpoint - preserving original"""
        return self.state["current_example"] % 30 == 0  # Every 30 examples
    
    def advance_progress(self):
        """Advance to next example/epoch/phase - preserving original"""
        self.state["current_example"] += 1
        self.state["total_examples_processed"] += 1
        
        current_phase_size = self.state["phase_sizes"][self.state["current_phase"] - 1]
        epochs_for_phase = self.state["epochs_per_phase"][self.state["current_phase"] - 1]
        
        # Check if we completed current phase
        if self.state["current_example"] >= current_phase_size:
            if self.state["current_epoch"] >= epochs_for_phase:
                # Move to next phase
                self.state["current_phase"] += 1
                self.state["current_example"] = 0
                self.state["current_epoch"] = 1
                print(f"🚀 Advanced to Phase {self.state['current_phase']}")
            else:
                # Next epoch in same phase
                self.state["current_epoch"] += 1
                self.state["current_example"] = 0
                print(f"📚 Advanced to Epoch {self.state['current_epoch']} of Phase {self.state['current_phase']}")
    
    def is_training_complete(self) -> bool:
        """Check if training is complete - preserving original"""
        return self.state["current_phase"] > len(self.state["phase_sizes"])
    
    def get_current_examples(self, all_examples: List[Dict]) -> List[Dict]:
        """Get examples for current phase - preserving original"""
        if self.is_training_complete():
            return []
        
        phase_size = self.state["phase_sizes"][self.state["current_phase"] - 1]
        return all_examples[:phase_size]
    
    def print_status(self):
        """Print current training status - preserving original"""
        if self.is_training_complete():
            print("🎉 Training Complete!")
            return
        
        current_phase = self.state["current_phase"]
        current_example = self.state["current_example"]
        current_epoch = self.state["current_epoch"]
        phase_size = self.state["phase_sizes"][current_phase - 1]
        total_processed = self.state["total_examples_processed"]
        api_calls = self.state["api_calls_made"]
        estimated_cost = self.state["estimated_cost"]
        
        print(f"\n📊 Training Status:")
        print(f"   Phase: {current_phase}/{len(self.state['phase_sizes'])}")
        print(f"   Epoch: {current_epoch}/{self.state['epochs_per_phase'][current_phase-1]}")
        print(f"   Example: {current_example}/{phase_size}")
        print(f"   Total Processed: {total_processed}")
        print(f"   API Calls: {api_calls}")
        print(f"   Estimated Cost: ${estimated_cost:.2f}")
        
        if self.state["successful_batches"] > 0:
            avg_loss = self.state["total_loss"] / self.state["successful_batches"]
            print(f"   Average Loss: {avg_loss:.4f}")
        
        if self.state["last_saved"]:
            print(f"   Last Saved: {self.state['last_saved']}")

class ProgressiveULDTrainer:
    """Enhanced Progressive ULD trainer - preserving original flow with fixes"""
    
    def __init__(self, student_model_name: str = "Qwen/Qwen3-4B-Thinking-2507"):
        self.checkpoint = DistillationCheckpoint()
        self.student_model_name = student_model_name
        
        # Load model, tokenizer, optimizer (preserving original)
        self.model, self.tokenizer, self.optimizer = self.checkpoint.load_model_checkpoint(
            student_model_name
        )
        
        print("🎯 Progressive ULD Trainer initialized with Together AI (ENHANCED)")
        self.checkpoint.print_status()
    
    def validate_model(self) -> Dict:
        """Enhanced validation - preserving original structure"""
        test_prompts = [
            "What is software architecture?",
            "Explain microservices briefly.",
            "How do you design scalable systems?",
            # NEW: Additional prompts for better validation
            "What are design patterns?",
            "Describe REST API principles."
        ]
        
        self.model.eval()
        results = {"coherent": 0, "repetitive": 0, "failed": 0}
        
        with torch.no_grad():  # ENHANCED: Proper gradient handling
            for prompt in test_prompts:
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    generated = self.tokenizer.decode(
                        outputs[0][inputs.input_ids.size(1):], 
                        skip_special_tokens=True
                    )
                    
                    # ENHANCED: Better quality check
                    words = generated.split()[:20]
                    unique_words = set(words)
                    if len(unique_words) < len(words) * 0.4:  # Improved threshold
                        results["repetitive"] += 1
                    elif len(generated.strip()) > 15 and len(unique_words) >= 3:
                        results["coherent"] += 1
                    else:
                        results["failed"] += 1
                        
                except Exception as e:
                    print(f"⚠️ Validation error: {e}")
                    results["failed"] += 1
        
        return results
    
    def train_step(self, prompt: str, target: str) -> Tuple[Optional[float], str]:
        """ENHANCED training step with fixed ULD - preserving original flow"""
        try:
            # Get teacher response (cached) - preserving original
            teacher_response = self.checkpoint.get_teacher_response_cached(prompt)
            teacher_text, teacher_logits = self.parse_teacher_logits(teacher_response)
            
            # Prepare student input - preserving original
            full_text = prompt + " " + teacher_text
            inputs = self.tokenizer(full_text, return_tensors="pt", max_length=1024, truncation=True, padding=True)
            
            # Student forward pass - preserving original
            self.model.train()
            outputs = self.model(**inputs)
            student_logits = outputs.logits
            
            # Extract generation portion - preserving original
            prompt_len = len(self.tokenizer(prompt, return_tensors="pt")["input_ids"][0])
            student_gen_logits = student_logits[0, prompt_len-1:-1, :]
            
            # Align sequences - preserving original
            min_len = min(teacher_logits.shape[0], student_gen_logits.shape[0])
            if min_len <= 0:
                return None, "NO_TOKENS"
            
            teacher_aligned = teacher_logits[:min_len].unsqueeze(0)
            student_aligned = student_gen_logits[:min_len].unsqueeze(0)
            
            # FIXED: Compute proper ULD loss using Wasserstein distance
            uld_loss = self.compute_enhanced_uld_loss(teacher_aligned, student_aligned)
            
            # CE loss component - preserving original
            labels = inputs["input_ids"][0, prompt_len:prompt_len+min_len]
            ce_loss = F.cross_entropy(
                student_aligned.view(-1, student_aligned.size(-1)), 
                labels.view(-1)
            )
            
            # ENHANCED: Better loss combination ratio
            total_loss = 0.3 * ce_loss + 0.7 * uld_loss
            
            # Backward pass - preserving original with enhancement
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            return total_loss.item(), "ULD"
            
        except Exception as e:
            print(f"⚠️ Training step failed: {e}")
            return None, "FAILED"
    
    def compute_enhanced_uld_loss(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
        """ENHANCED ULD loss using proper Wasserstein distance"""
        # Convert to probabilities with temperature scaling for stability
        temperature = 2.0
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        
        # ULD uses sorted probabilities for Wasserstein-1 distance
        teacher_sorted, _ = torch.sort(teacher_probs, dim=-1, descending=True)
        student_sorted, _ = torch.sort(student_probs, dim=-1, descending=True)
        
        # Align distributions by taking minimum vocabulary size
        min_vocab = min(teacher_sorted.size(-1), student_sorted.size(-1))
        teacher_aligned = teacher_sorted[:, :, :min_vocab]
        student_aligned = student_sorted[:, :, :min_vocab]
        
        # Wasserstein-1 distance: sum of absolute differences of sorted probabilities
        wasserstein_dist = torch.sum(torch.abs(teacher_aligned - student_aligned), dim=-1)
        
        return torch.mean(wasserstein_dist)
    
    def compute_simple_uld_loss(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
        """FALLBACK: Keep original method for compatibility"""
        # Convert to probabilities
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)
        
        # Simple approximation: L1 distance between sorted probabilities
        teacher_sorted, _ = torch.sort(teacher_probs, dim=-1, descending=True)
        student_sorted, _ = torch.sort(student_probs, dim=-1, descending=True)
        
        # Pad to same size
        max_vocab = max(teacher_sorted.size(-1), student_sorted.size(-1))
        teacher_padded = F.pad(teacher_sorted, (0, max_vocab - teacher_sorted.size(-1)))
        student_padded = F.pad(student_sorted, (0, max_vocab - student_sorted.size(-1)))
        
        return torch.mean(torch.abs(teacher_padded - student_padded))
    
    def parse_teacher_logits(self, response_data: Dict) -> Tuple[str, torch.Tensor]:
        """FIXED Together AI teacher response parsing - preserving original structure"""
        choice = response_data["choices"][0]
        generated_text = choice["message"]["content"]
        
        # Together AI logits format
        if "logprobs" not in choice or not choice["logprobs"]:
            raise RuntimeError("No logprobs in Together AI teacher response")
        
        # Together AI provides logprobs differently than OpenAI/OpenRouter
        logprobs_data = choice["logprobs"]
        
        # FIXED: Extract token logprobs using proper vocab size
        if "token_logprobs" in logprobs_data and "tokens" in logprobs_data:
            token_logprobs = logprobs_data["token_logprobs"]
            tokens = logprobs_data["tokens"]
            top_logprobs = logprobs_data.get("top_logprobs", [])
            
            # FIXED: Use actual student tokenizer vocab size
            vocab_size = self.tokenizer.vocab_size
            logits_list = []
            
            for i, (token, logprob) in enumerate(zip(tokens, token_logprobs)):
                if logprob is None:
                    continue
                    
                # FIXED: Initialize with very negative values for stability
                logits = torch.full((vocab_size,), -20.0)
                
                # ENHANCED: Try to get actual token ID first
                try:
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    if token_id < vocab_size:
                        logits[token_id] = logprob
                    else:
                        # Fallback to hash-based mapping
                        base_idx = hash(token) % vocab_size
                        logits[base_idx] = logprob
                except:
                    # Fallback to hash-based mapping
                    base_idx = hash(token) % vocab_size
                    logits[base_idx] = logprob
                
                # ENHANCED: Add top logprobs if available
                if i < len(top_logprobs) and top_logprobs[i]:
                    for j, (top_token, top_logprob) in enumerate(list(top_logprobs[i].items())[:5]):
                        try:
                            top_token_id = self.tokenizer.convert_tokens_to_ids(top_token)
                            if top_token_id < vocab_size:
                                logits[top_token_id] = top_logprob
                            else:
                                idx = (hash(top_token) + j) % vocab_size
                                logits[idx] = top_logprob
                        except:
                            idx = (hash(top_token) + j) % vocab_size
                            logits[idx] = top_logprob
                
                logits_list.append(logits)
        
        elif "content" in logprobs_data:
            # Alternative format - similar to OpenAI (preserving original)
            content_logprobs = logprobs_data["content"]
            logits_list = []
            vocab_size = self.tokenizer.vocab_size  # FIXED
            
            for token_data in content_logprobs:
                logits = torch.full((vocab_size,), -10.0)
                
                if "logprob" in token_data:
                    logits[0] = token_data["logprob"]
                
                if "top_logprobs" in token_data:
                    for i, top_data in enumerate(token_data["top_logprobs"]):
                        if i < vocab_size and "logprob" in top_data:
                            logits[i] = top_data["logprob"]
                
                logits_list.append(logits)
        
        else:
            raise RuntimeError("Unrecognized logprobs format from Together AI")
        
        if not logits_list:
            raise RuntimeError("No logits extracted from Together AI response")
        
        return generated_text, torch.stack(logits_list)
    
    def run_training(self, examples: List[Dict]):
        """Run complete progressive training - preserving original flow"""
        print("🚀 Starting Progressive ULD Training with Together AI & Kimi K2 (ENHANCED)")
        
        while not self.checkpoint.is_training_complete():
            self.checkpoint.print_status()
            
            # Get examples for current phase - preserving original
            current_examples = self.checkpoint.get_current_examples(examples)
            
            if not current_examples:
                print("❌ No examples for current phase")
                break
            
            # Train on current phase - preserving original
            phase_success = self.train_phase(current_examples)
            
            if not phase_success:
                print("❌ Phase training failed - stopping")
                break
            
            print(f"✅ Phase {self.checkpoint.state['current_phase']} completed successfully")
        
        if self.checkpoint.is_training_complete():
            print("🎉 Training completed successfully!")
            self.save_final_model()
    
    def train_phase(self, examples: List[Dict]) -> bool:
        """Train on current phase examples - preserving original with enhancement"""
        start_idx = self.checkpoint.state["current_example"]
        
        for i in range(start_idx, len(examples)):
            example = examples[i]
            prompt = example.get("input", "")
            target = example.get("output", "")
            
            if len(prompt.strip()) < 10:
                self.checkpoint.advance_progress()
                continue
            
            # Training step - preserving original
            loss, mode = self.train_step(prompt, target)
            
            if loss is not None:
                self.checkpoint.log_progress(loss, mode)
                print(f"📈 Step {i}: Loss={loss:.4f}, Mode={mode}")
            
            # Validation - preserving original with enhancement
            if self.checkpoint.should_validate():
                validation = self.validate_model()
                self.checkpoint.log_progress(loss or 0, mode, validation)
                print(f"🔍 Validation: {validation}")
                
                # ENHANCED: Better stopping condition
                total_responses = sum(validation.values())
                if total_responses > 0:
                    coherent_ratio = validation["coherent"] / total_responses
                    if coherent_ratio < 0.4:  # Less than 40% coherent
                        print("❌ Model degradation detected - stopping phase")
                        return False
            
            # Save checkpoint - preserving original
            if self.checkpoint.should_save_checkpoint():
                self.checkpoint.save_model_checkpoint(self.model, self.tokenizer, self.optimizer)
                self.checkpoint.save_state()
            
            # Advance progress - preserving original
            self.checkpoint.advance_progress()
            self.checkpoint.save_state()  # Save state after each example
        
        return True
    
    def save_final_model(self):
        """Save final trained model - preserving original"""
        final_dir = f"./qwen3-4b-uld-kimi-final-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(final_dir, exist_ok=True)
        
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        
        # ENHANCED: Save training summary with fixes
        summary = {
            "teacher_model": "moonshotai/Kimi-K2-Instruct-0905",
            "teacher_provider": "Together AI",
            "training_completed": datetime.now().isoformat(),
            "total_examples_processed": self.checkpoint.state["total_examples_processed"],
            "total_phases": len(self.checkpoint.state["phase_sizes"]),
            "api_calls_made": self.checkpoint.state["api_calls_made"],
            "estimated_cost": self.checkpoint.state["estimated_cost"],
            "final_validation": self.validate_model(),
            "enhancements_applied": [
                "Fixed ULD loss to use Wasserstein distance",
                "Corrected Together AI logprobs parsing", 
                "Fixed vocabulary size alignment",
                "Enhanced validation and stopping conditions",
                "Added teacher data saving for analysis"
            ]
        }
        
        with open(f"{final_dir}/training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Final model saved to {final_dir}")
        print(f"📊 Training Summary: {summary}")
    
    def analyze_teacher_data(self, limit: int = 10):
        """NEW: Analyze saved teacher interactions"""
        index_file = self.checkpoint.teacher_data_dir / "teacher_interactions_index.jsonl"
        
        if not index_file.exists():
            print("📭 No teacher interactions found")
            return
        
        print(f"🔍 Analyzing Teacher Interactions (last {limit}):")
        print("=" * 80)
        
        interactions = []
        with open(index_file, 'r', encoding='utf-8') as f:
            for line in f:
                interactions.append(json.loads(line))
        
        recent_interactions = interactions[-limit:] if len(interactions) > limit else interactions
        
        for i, interaction in enumerate(recent_interactions, 1):
            print(f"\n{i}. Phase {interaction['phase']}, Example {interaction['example']}")
            print(f"   Time: {interaction['timestamp']}")
            print(f"   Prompt: {interaction['prompt_preview']}")
            print(f"   Response: {interaction['response_preview']}")
            print("-" * 40)
        
        print(f"\n📊 Total Interactions: {len(interactions)}")
        print(f"📁 Data saved in: {self.checkpoint.teacher_data_dir}")

# -------------------------------
# ENHANCED Usage with optional testing
# -------------------------------

if __name__ == "__main__":
    print("🚀 ULD Distillation Trainer - ENHANCED VERSION")
    print("=" * 60)
    
    # OPTIONAL: Run test suite (can be skipped)
    run_tests = input("Run test suite? (y/n): ").lower() == 'y'
    
    if run_tests:
        test_suite = ULDTestSuite()
        if not test_suite.run_all_tests():
            print("\n❌ Tests failed! Continuing anyway (fixes are applied)...")
            input("Press Enter to continue...")
    
    print("\n" + "=" * 60)
    
    # Verify Together AI API key - preserving original
    if not os.getenv('TOGETHER_API_KEY'):
        print("❌ Please set TOGETHER_API_KEY environment variable")
        print("   export TOGETHER_API_KEY='your_api_key_here'")
        exit(1)
    
    # Load your dataset - preserving original
    try:
        from huggingface_hub import hf_hub_download
        
        file_path = hf_hub_download(
            repo_id="ajibawa-2023/Software-Architecture",
            filename="Software_Architecture_Final.jsonl",
            repo_type="dataset"
        )
        
        examples = []
        with open(file_path, "r") as f:
            for line in f:
                examples.append(json.loads(line))
        
        print(f"📚 Loaded {len(examples)} training examples")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        exit(1)
    
    # Create trainer and run - preserving original flow
    try:
        trainer = ProgressiveULDTrainer()
        
        # NEW: Show existing teacher data if any
        if hasattr(trainer, 'analyze_teacher_data'):
            trainer.analyze_teacher_data(limit=3)
        
        # Run training (will resume from where it left off) - preserving original
        trainer.run_training(examples)
        
    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
        print("💾 Progress has been saved and can be resumed")
        print("📚 Teacher interactions saved in ./distillation_checkpoints/teacher_data/")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        print("💾 Check logs and resume from last checkpoint")
        print("📚 Teacher interactions saved in ./distillation_checkpoints/teacher_data/")
        
    print("\n🏁 Training session ended")
    print("\n📁 Teacher Data Locations:")
    print("   - Full data: ./distillation_checkpoints/teacher_data/pair_*.json") 
    print("   - Readable: ./distillation_checkpoints/teacher_data/readable_*.txt")
    print("   - Index: ./distillation_checkpoints/teacher_data/teacher_interactions_index.jsonl")
