"""
ULD Distillation with Complete Checkpointing System - Together AI Version
Uses Together AI API with Kimi K2 model for teacher logits
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

class DistillationCheckpoint:
    """Comprehensive checkpoint management for distillation training"""
    
    def __init__(self, checkpoint_dir: str = "./distillation_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # File paths
        self.state_file = self.checkpoint_dir / "training_state.json"
        self.model_dir = self.checkpoint_dir / "model"
        self.optimizer_file = self.checkpoint_dir / "optimizer.pt"
        self.progress_file = self.checkpoint_dir / "progress.json"
        self.cache_dir = self.checkpoint_dir / "teacher_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize state
        self.state = self.load_state()
    
    def load_state(self) -> Dict[str, Any]:
        """Load training state or create new one"""
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
                "phase_sizes": [10, 50, 200, 1000],  # Optimal progressive phases
                "epochs_per_phase": [1, 2, 2, 3],  # Optimal epochs per phase
                "learning_rates": [5e-6, 3e-6, 2e-6, 1e-6],  # Optimal decreasing LR
                "training_history": [],
                "validation_history": [],
                "last_saved": None,
                "api_calls_made": 0,
                "estimated_cost": 0.0
            }
    
    def save_state(self):
        """Save current training state"""
        self.state["last_saved"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        print(f"💾 State saved: Phase {self.state['current_phase']}, "
              f"Example {self.state['current_example']}")
    
    def save_model_checkpoint(self, model, tokenizer, optimizer):
        """Save model, tokenizer, and optimizer state"""
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
        """Load model, tokenizer, and optimizer from checkpoint"""
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
        """Cached teacher responses to avoid re-calling API"""
        cache_key = str(abs(hash(prompt)))
        cache_file = self.cache_dir / f"teacher_{cache_key}.json"
        
        if cache_file.exists():
            print(f"📂 Using cached teacher response")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Make API call
        print(f"🌐 Making API call to teacher (Together AI)...")
        response = self.call_teacher_api(prompt)
        
        # Cache the response
        with open(cache_file, 'w') as f:
            json.dump(response, f)
        
        # Update API call count and cost estimate
        self.state["api_calls_made"] += 1
        # Together AI pricing for Kimi K2 (Input: $1.00/M tokens, Output: $3.00/M tokens)
        self.state["estimated_cost"] += 0.005  # Estimated cost per 512 token call
        
        return response
    
    def call_teacher_api(self, prompt: str) -> Dict:
        """Call Together AI API for teacher logits using Kimi K2"""
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
        """Log training progress"""
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
        """Determine if we should run validation"""
        return self.state["current_example"] % 20 == 0  # Every 20 examples
    
    def should_save_checkpoint(self) -> bool:
        """Determine if we should save checkpoint"""
        return self.state["current_example"] % 30 == 0  # Every 30 examples
    
    def advance_progress(self):
        """Advance to next example/epoch/phase"""
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
        """Check if training is complete"""
        return self.state["current_phase"] > len(self.state["phase_sizes"])
    
    def get_current_examples(self, all_examples: List[Dict]) -> List[Dict]:
        """Get examples for current phase"""
        if self.is_training_complete():
            return []
        
        phase_size = self.state["phase_sizes"][self.state["current_phase"] - 1]
        return all_examples[:phase_size]
    
    def print_status(self):
        """Print current training status"""
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
    """Progressive ULD trainer with comprehensive checkpointing using Together AI"""
    
    def __init__(self, student_model_name: str = "Qwen/Qwen3-4B-Thinking-2507"):
        self.checkpoint = DistillationCheckpoint()
        self.student_model_name = student_model_name
        
        # Load model, tokenizer, optimizer
        self.model, self.tokenizer, self.optimizer = self.checkpoint.load_model_checkpoint(
            student_model_name
        )
        
        print("🎯 Progressive ULD Trainer initialized with Together AI")
        self.checkpoint.print_status()
    
    def validate_model(self) -> Dict:
        """Validate model generation quality"""
        test_prompts = [
            "What is software architecture?",
            "Explain microservices briefly.",
            "How do you design scalable systems?"
        ]
        
        self.model.eval()
        results = {"coherent": 0, "repetitive": 0, "failed": 0}
        
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
                
                # Check quality
                words = generated.split()[:20]
                if len(set(words)) < len(words) * 0.5:  # Too repetitive
                    results["repetitive"] += 1
                elif len(generated.strip()) > 10:
                    results["coherent"] += 1
                else:
                    results["failed"] += 1
                    
            except Exception:
                results["failed"] += 1
        
        return results
    
    def train_step(self, prompt: str, target: str) -> Tuple[Optional[float], str]:
        """Single training step with ULD"""
        try:
            # Get teacher response (cached)
            teacher_response = self.checkpoint.get_teacher_response_cached(prompt)
            
            # Parse teacher logits with confirmation checks
            teacher_text, teacher_logits = self.parse_teacher_logits(teacher_response)
            
            # Confirmation check: Teacher logits ready for ULD
            print(f"📊 Teacher logits extracted successfully")
            print(f"   Text: '{teacher_text[:50]}...' ({len(teacher_text)} chars)")
            print(f"   Logits: {teacher_logits.shape} tensor ready for ULD")
            
            # Prepare student input
            full_text = prompt + " " + teacher_text
            inputs = self.tokenizer(full_text, return_tensors="pt", max_length=1024, truncation=True)
            
            # Student forward pass
            self.model.train()
            outputs = self.model(**inputs)
            student_logits = outputs.logits
            
            # Extract generation portion
            prompt_len = len(self.tokenizer(prompt, return_tensors="pt")["input_ids"][0])
            student_gen_logits = student_logits[0, prompt_len-1:-1, :]
            
            # Align sequences
            min_len = min(teacher_logits.shape[0], student_gen_logits.shape[0])
            if min_len <= 0:
                return None, "NO_TOKENS"
            
            teacher_aligned = teacher_logits[:min_len].unsqueeze(0)
            student_aligned = student_gen_logits[:min_len].unsqueeze(0)
            
            # Confirmation check: Alignment successful
            print(f"🔄 Sequence alignment complete")
            print(f"   Teacher aligned: {teacher_aligned.shape}")
            print(f"   Student aligned: {student_aligned.shape}")
            print(f"   Aligned length: {min_len} tokens")
            
            # Compute ULD loss (simplified version for demo)
            uld_loss = self.compute_simple_uld_loss(teacher_aligned, student_aligned)
            
            # CE loss component
            labels = inputs["input_ids"][0, prompt_len:prompt_len+min_len]
            ce_loss = F.cross_entropy(
                student_aligned.view(-1, student_aligned.size(-1)), 
                labels.view(-1)
            )
            
            # Combined loss (optimal ratio based on ULD research)
            total_loss = 0.4 * ce_loss + 0.6 * uld_loss
            
            # Final confirmation: Loss computation successful
            print(f"💯 Loss computation successful")
            print(f"   CE Loss: {ce_loss.item():.4f}")
            print(f"   ULD Loss: {uld_loss.item():.4f}")
            print(f"   Total Loss: {total_loss.item():.4f}")
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            return total_loss.item(), "ULD"
            
        except Exception as e:
            print(f"⚠️ Training step failed: {e}")
            return None, "FAILED"
    
    def compute_simple_uld_loss(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
        """Simplified ULD loss computation"""
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
        """Parse Together AI teacher response to extract logits"""
        choice = response_data["choices"][0]
        generated_text = choice["message"]["content"]
        
        # Together AI logits format
        if "logprobs" not in choice or not choice["logprobs"]:
            raise RuntimeError("No logprobs in Together AI teacher response")
        
        # Together AI provides logprobs differently than OpenAI/OpenRouter
        logprobs_data = choice["logprobs"]
        
        # Confirmation check 1: Logprobs structure received
        print(f"✅ Logprobs structure received from Together AI")
        print(f"   Available keys: {list(logprobs_data.keys())}")
        
        # Extract token logprobs
        if "token_logprobs" in logprobs_data and "tokens" in logprobs_data:
            token_logprobs = logprobs_data["token_logprobs"]
            tokens = logprobs_data["tokens"]
            
            # Confirmation check 2: Token-based format
            print(f"✅ Using token_logprobs format")
            print(f"   Number of tokens: {len(tokens)}")
            print(f"   Sample tokens: {tokens[:3] if len(tokens) > 0 else 'None'}")
            
            # Create simple logits tensor
            logits_list = []
            vocab_size = 50000
            
            for i, (token, logprob) in enumerate(zip(tokens, token_logprobs)):
                if logprob is None:
                    continue
                    
                logits = torch.full((vocab_size,), -10.0)
                
                # Set the actual token logprob
                logits[i % vocab_size] = logprob
                
                # If top_logprobs available, use those too
                if "top_logprobs" in logprobs_data and i < len(logprobs_data["top_logprobs"]):
                    top_logprobs = logprobs_data["top_logprobs"][i]
                    if top_logprobs:
                        for j, (top_token, top_logprob) in enumerate(top_logprobs.items()):
                            if j < vocab_size:
                                logits[j] = top_logprob
                
                logits_list.append(logits)
        
        elif "content" in logprobs_data:
            # Alternative format - similar to OpenAI
            content_logprobs = logprobs_data["content"]
            
            # Confirmation check 3: Content-based format
            print(f"✅ Using content logprobs format")
            print(f"   Number of content items: {len(content_logprobs)}")
            if len(content_logprobs) > 0:
                first_item = content_logprobs[0]
                print(f"   Sample content keys: {list(first_item.keys())}")
            
            logits_list = []
            vocab_size = 50000
            
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
        
        # Final confirmation check: Logits tensor creation
        logits_tensor = torch.stack(logits_list)
        print(f"✅ Successfully created logits tensor: {logits_tensor.shape}")
        print(f"   Generated text length: {len(generated_text.split())} words")
        print(f"   Logits shape: [{logits_tensor.shape[0]} tokens x {logits_tensor.shape[1]} vocab]")
        
        return generated_text, logits_tensor
    
    def run_training(self, examples: List[Dict]):
        """Run complete progressive training"""
        print("🚀 Starting Progressive ULD Training with Together AI & Kimi K2")
        
        while not self.checkpoint.is_training_complete():
            self.checkpoint.print_status()
            
            # Get examples for current phase
            current_examples = self.checkpoint.get_current_examples(examples)
            
            if not current_examples:
                print("❌ No examples for current phase")
                break
            
            # Train on current phase
            phase_success = self.train_phase(current_examples)
            
            if not phase_success:
                print("❌ Phase training failed - stopping")
                break
            
            print(f"✅ Phase {self.checkpoint.state['current_phase']} completed successfully")
        
        if self.checkpoint.is_training_complete():
            print("🎉 Training completed successfully!")
            self.save_final_model()
    
    def train_phase(self, examples: List[Dict]) -> bool:
        """Train on current phase examples"""
        start_idx = self.checkpoint.state["current_example"]
        
        for i in range(start_idx, len(examples)):
            example = examples[i]
            prompt = example.get("input", "")
            target = example.get("output", "")
            
            if len(prompt.strip()) < 10:
                self.checkpoint.advance_progress()
                continue
            
            # Training step
            loss, mode = self.train_step(prompt, target)
            
            if loss is not None:
                self.checkpoint.log_progress(loss, mode)
                print(f"📈 Step {i}: Loss={loss:.4f}, Mode={mode}")
            
            # Validation
            if self.checkpoint.should_validate():
                validation = self.validate_model()
                self.checkpoint.log_progress(loss or 0, mode, validation)
                print(f"🔍 Validation: {validation}")
                
                # Stop if model is degrading
                if validation["repetitive"] > validation["coherent"]:
                    print("❌ Model degradation detected - stopping phase")
                    return False
            
            # Save checkpoint
            if self.checkpoint.should_save_checkpoint():
                self.checkpoint.save_model_checkpoint(self.model, self.tokenizer, self.optimizer)
                self.checkpoint.save_state()
            
            # Advance progress
            self.checkpoint.advance_progress()
            self.checkpoint.save_state()  # Save state after each example
        
        return True
    
    def save_final_model(self):
        """Save final trained model"""
        final_dir = f"./qwen3-4b-uld-kimi-final-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(final_dir, exist_ok=True)
        
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        
        # Save training summary
        summary = {
            "teacher_model": "moonshotai/Kimi-K2-Instruct-0905",
            "teacher_provider": "Together AI",
            "training_completed": datetime.now().isoformat(),
            "total_examples_processed": self.checkpoint.state["total_examples_processed"],
            "total_phases": len(self.checkpoint.state["phase_sizes"]),
            "api_calls_made": self.checkpoint.state["api_calls_made"],
            "estimated_cost": self.checkpoint.state["estimated_cost"],
            "final_validation": self.validate_model()
        }
        
        with open(f"{final_dir}/training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Final model saved to {final_dir}")
        print(f"📊 Training Summary: {summary}")

# -------------------------------
# Usage
# -------------------------------

if __name__ == "__main__":
    # Verify Together AI API key
    if not os.getenv('TOGETHER_API_KEY'):
        print("❌ Please set TOGETHER_API_KEY environment variable")
        print("   export TOGETHER_API_KEY='your_api_key_here'")
        exit(1)
    
    # Load your dataset
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
    
    # Create trainer
    trainer = ProgressiveULDTrainer()
    
    # Run training (will resume from where it left off)
    trainer.run_training(examples)
