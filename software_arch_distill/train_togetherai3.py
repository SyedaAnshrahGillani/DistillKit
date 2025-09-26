"""
Enhanced ULD Distillation with Multiple Dataset Support - Together AI Version
Uses Together AI API with Kimi K2 model for teacher logits
Now supports multiple datasets with dynamic phase generation
FIXED: Preserves training progress across dataset switches
ADDED: Non-disruptive permanent archiving of teacher responses
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

class DatasetManager:
    """Manages multiple datasets with different formats"""
    
    AVAILABLE_DATASETS = {
        "1": {
            "name": "Software Architecture (Public JSONL)",
            "repo_id": "ajibawa-2023/Software-Architecture",
            "filename": "Software_Architecture_Final.jsonl",
            "format": "jsonl",
            "description": "Public software architecture Q&A dataset - 400k examples"
        },
        "2": {
            "name": "Cloud Dataset by Hamed (Public JSON)",
            "repo_id": "Anshrah/Cloud-Dataset-byHamed",
            "filename": "dataset.json",
            "format": "json",
            "description": "Cloud dataset shared by hamed"
        },
        "3": {
            "name": "Private JSON Dataset (Not defined-don't select)",
            "repo_id": None,
            "filename": None,
            "format": "json", 
            "description": "Private JSON dataset loaded from local file"
        },
        "4": {
            "name": "Private JSONL Dataset (Not defined-don't select)",
            "repo_id": None,
            "filename": None, 
            "format": "jsonl",
            "description": "Private JSONL dataset loaded from local file"
        }
    }
    
    @staticmethod
    def show_dataset_menu():
        """Display available datasets and get user selection"""
        print("\n" + "="*60)
        print("📂 AVAILABLE DATASETS")
        print("="*60)
        
        for key, dataset in DatasetManager.AVAILABLE_DATASETS.items():
            print(f"{key}. {dataset['name']}")
            print(f"   Format: {dataset['format'].upper()}")
            print(f"   Description: {dataset['description']}")
            print()
        
        while True:
            try:
                choice = input("Select dataset (1-4): ").strip()
                if choice in DatasetManager.AVAILABLE_DATASETS:
                    return choice
                else:
                    print("❌ Invalid choice. Please select 1, 2, 3, or 4.")
            except KeyboardInterrupt:
                print("\n👋 Training cancelled.")
                exit(0)
    
    @staticmethod
    def load_dataset(choice: str) -> Tuple[List[Dict], str]:
        """Load selected dataset and return examples with dataset info"""
        dataset_config = DatasetManager.AVAILABLE_DATASETS[choice]
        dataset_name = dataset_config["name"]
        
        if choice in ["1", "2"]:
            # Public HuggingFace datasets
            from huggingface_hub import hf_hub_download
            
            file_path = hf_hub_download(
                repo_id=dataset_config["repo_id"],
                filename=dataset_config["filename"],
                repo_type="dataset"
            )
            
            examples = []
            
            if dataset_config["format"] == "jsonl":
                # JSONL format
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            examples.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
            else:
                # JSON format
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        examples = data
                    else:
                        print("❌ JSON file should contain a list of objects.")
                        return [], ""
            
            print(f"✅ Loaded {len(examples):,} examples from {dataset_name}")
            print(f"📊 Total Dataset Size: {len(examples):,} examples")
            return examples, dataset_name
            
        elif choice in ["3", "4"]:
            # Private local datasets
            print(f"\n📁 Loading Private {dataset_config['format'].upper()} Dataset from Local File")
            
            while True:
                if choice == "3":
                    file_path = input("Enter path to your private JSON file: ").strip()
                else:
                    file_path = input("Enter path to your private JSONL file: ").strip()
                
                if not os.path.exists(file_path):
                    print("❌ File not found. Please check the path.")
                    continue
                
                try:
                    examples = []
                    
                    if choice == "3":  # JSON format
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                examples = data
                            else:
                                print("❌ JSON file should contain a list of objects.")
                                continue
                    
                    else:  # JSONL format
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                try:
                                    examples.append(json.loads(line.strip()))
                                except json.JSONDecodeError:
                                    continue
                    
                    if not examples:
                        print("❌ No valid examples found in file.")
                        continue
                    
                    # Validate format
                    if not DatasetManager.validate_dataset_format(examples):
                        continue
                    
                    dataset_name = f"Private {dataset_config['format'].upper()} ({os.path.basename(file_path)})"
                    print(f"✅ Loaded {len(examples):,} examples from private {dataset_name}")
                    print(f"📊 Total Dataset Size: {len(examples):,} examples")
                    return examples, dataset_name
                    
                except Exception as e:
                    print(f"❌ Error loading private file: {e}")
                    continue
    
    @staticmethod
    def validate_dataset_format(examples: List[Dict]) -> bool:
        """Validate that dataset has required fields"""
        required_fields = ["input", "output"]
        
        # Check first few examples
        for i, example in enumerate(examples[:3]):
            if not isinstance(example, dict):
                print(f"❌ Example {i+1} is not a dictionary object.")
                return False
            
            missing_fields = [field for field in required_fields if field not in example]
            if missing_fields:
                print(f"❌ Example {i+1} missing required fields: {missing_fields}")
                print("📋 Required format: {'input': 'question', 'output': 'answer'}")
                return False
        
        print("✅ Dataset format validated successfully")
        return True
    
    @staticmethod
    def calculate_dynamic_phases(dataset_size: int) -> Tuple[List[int], List[int], List[float]]:
        """Calculate optimal phases based on dataset size - Higher LRs for manual stopping"""
        if dataset_size <= 100:
            # Small dataset - Aggressive learning
            phases = [10, 25, 50, min(dataset_size, 100)]
            epochs = [2, 3, 3, 4]
            learning_rates = [1e-5, 8e-6, 6e-6, 4e-6]
        elif dataset_size <= 1000:
            # Medium dataset - Still aggressive
            phases = [20, 100, 300, min(dataset_size, 1000)]
            epochs = [2, 2, 3, 3]
            learning_rates = [8e-6, 6e-6, 4e-6, 3e-6]
        elif dataset_size <= 10000:
            # Large dataset - Higher LRs for faster progress
            phases = [50, 200, 1000, 5000, min(dataset_size, 10000)]
            epochs = [1, 2, 2, 2, 3]
            learning_rates = [8e-6, 6e-6, 4e-6, 3e-6, 2e-6]
        else:
            # Very large dataset - Keep high LRs since manual stopping
            phases = [100, 500, 2000, 10000, 50000, min(dataset_size, 100000)]
            epochs = [1, 1, 2, 2, 2, 3]
            learning_rates = [8e-6, 6e-6, 4e-6, 3e-6, 2e-6, 1e-6]
        
        # Ensure phases don't exceed dataset size
        phases = [min(phase, dataset_size) for phase in phases]
        
        print(f"📊 Dynamic phases calculated for {dataset_size} examples (Higher LRs for manual stopping):")
        print(f"   Phases: {phases}")
        print(f"   Epochs per phase: {epochs}")
        print(f"   Learning rates: {learning_rates}")
        
        return phases, epochs, learning_rates

class DistillationCheckpoint:
    """Enhanced checkpoint management with multi-dataset progress preservation and non-disruptive archiving"""
    
    def __init__(self, checkpoint_dir: str = "./distillation_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # File paths
        self.state_file = self.checkpoint_dir / "training_state.json"
        self.model_dir = self.checkpoint_dir / "model"
        self.optimizer_file = self.checkpoint_dir / "optimizer.pt"
        self.progress_file = self.checkpoint_dir / "progress.json"
        self.cache_dir = self.checkpoint_dir / "teacher_cache"
        self.archive_dir = self.checkpoint_dir / "teacher_archive"  # Archive for permanent storage
        self.cache_dir.mkdir(exist_ok=True)
        self.archive_dir.mkdir(exist_ok=True)
        
        # Initialize state
        self.state = self.load_state()
    
    def load_state(self) -> Dict[str, Any]:
        """Load training state or create new one"""
        if self.state_file.exists():
            print(f"📂 Loading existing training state from {self.state_file}")
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                print(f"🔄 Resuming from: Dataset '{state.get('current_dataset', 'Unknown')}', "
                      f"Phase {state['current_phase']}, Example {state['current_example']}")
                return state
        else:
            print("🆕 Creating new training state")
            return {
                "created_at": datetime.now().isoformat(),
                "current_dataset": None,
                "dataset_size": 0,
                "current_phase": 1,
                "current_example": 0,
                "current_epoch": 1,
                "total_examples_processed": 0,
                "successful_batches": 0,
                "total_loss": 0.0,
                "phase_sizes": [10, 50, 200, 1000],  # Will be updated dynamically
                "epochs_per_phase": [1, 2, 2, 3],
                "learning_rates": [5e-6, 3e-6, 2e-6, 5e-6],  # Optimal decreasing LR
                "training_history": [],
                "validation_history": [],
                "last_saved": None,
                "api_calls_made": 0,
                "estimated_cost": 0.0
            }
    
 doesn't exceed new dataset phases
            if current_phase > len(phases):
                print(f"⚠️  Current phase {current_phase} exceeds new dataset phases {len(phases)}")
                print("🔄 Adjusting to final phase of new dataset")
                self.state["current_phase"] = len(phases)
                self.state["current_example"] = 0
                self.state["current_epoch"] = 1
            
            # Validate current example doesn't exceed current phase size
            elif current_example >= phases[current_phase - 1]:
                print(f"⚠️  Current example {current_example} exceeds phase size {phases[current_phase - 1]}")
                print("🔄 Moving to next epoch or phase")
                
                # Check if we can advance to next epoch in same phase
                if current_epoch < epochs[current_phase - 1]:
                    self.state["current_epoch"] = current_epoch + 1
                    self.state["current_example"] = 0
                    print(f"📚 Advanced to Epoch {self.state['current_epoch']} of Phase {current_phase}")
                else:
                    # Move to next phase if available
                    if current_phase < len(phases):
                        self.state["current_phase"] = current_phase + 1
                        self.state["current_example"] = 0
                        self.state["current_epoch"] = 1
                        print(f"🚀 Advanced to Phase {self.state['current_phase']}")
                    else:
                        # Training complete
                        self.state["current_phase"] = len(phases) + 1
                        print("🎉 Training appears to be complete!")
            
            self.save_state()
            print(f"✅ Dataset configuration updated, training will continue from saved progress")
        else:
            print(f"📋 Same dataset selected: {dataset_name} - continuing with existing progress")
    
    def save_state(self):
        """Save current training state"""
        self.state["last_saved"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        print(f"💾 State saved: Dataset '{self.state.get('current_dataset', 'Unknown')}', "
              f"Phase {self.state['current_phase']}, Example {self.state['current_example']}")
    
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
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_dir, 
                torch_dtype=torch.bfloat16
            )
            
            # Create optimizer with current learning rate
            current_phase = self.state.get("current_phase", 1)
            lr_index = min(current_phase - 1, len(self.state["learning_rates"]) - 1)
            lr = self.state["learning_rates"][lr_index]
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
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16
            )
            
            current_phase = self.state.get("current_phase", 1)
            lr_index = min(current_phase - 1, len(self.state["learning_rates"]) - 1)
            lr = self.state["learning_rates"][lr_index]
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            
            return model, tokenizer, optimizer
    
    def archive_teacher_response_safe(self, prompt: str, response: Dict):
        """Archive teacher responses permanently - completely non-disruptive"""
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_key = str(abs(hash(prompt)))
            
            # Create archive data
            archive_data = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": response,
                "dataset": self.state.get("current_dataset", "unknown"),
                "phase": self.state.get("current_phase", 0),
                "example": self.state.get("current_example", 0)
            }
            
            # Save to archive (permanent storage)
            archive_file = self.archive_dir / f"teacher_response_{timestamp}_{cache_key[:8]}.json"
            with open(archive_file, 'w', encoding='utf-8') as f:
                json.dump(archive_data, f, indent=2, ensure_ascii=False)
            
            # Update counter if possible (backward compatible)
            try:
                if "archived_responses" not in self.state:
                    self.state["archived_responses"] = 0
                self.state["archived_responses"] += 1
            except:
                # If state update fails, continue silently
                pass
                
        except Exception as e:
            # Archive failure should never interrupt main flow
            pass
    
    def get_teacher_response_cached(self, prompt: str) -> Dict:
        """Cached teacher responses to avoid re-calling API with non-disruptive archiving"""
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
        
        # Archive the response permanently (non-disruptive)
        self.archive_teacher_response_safe(prompt, response)
        
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
            "dataset": self.state.get("current_dataset", "unknown"),
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
        
        # Keep only last 1000 entries
        if len(self.state["training_history"]) > 1000:
            self.state["training_history"] = self.state["training_history"][-1000:]
    
    def should_validate(self) -> bool:
        """Determine if we should run validation"""
        return self.state["current_example"] % 20 == 0
    
    def should_save_checkpoint(self) -> bool:
        """Determine if we should save checkpoint"""
        return self.state["current_example"] % 30 == 0
    
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
                
                # Update optimizer learning rate for new phase
                self.update_optimizer_lr()
            else:
                # Next epoch in same phase
                self.state["current_epoch"] += 1
                self.state["current_example"] = 0
                print(f"📚 Advanced to Epoch {self.state['current_epoch']} of Phase {self.state['current_phase']}")
    
    def update_optimizer_lr(self):
        """Update optimizer learning rate for current phase"""
        if hasattr(self, 'current_optimizer'):
            current_phase = self.state["current_phase"]
            if current_phase <= len(self.state["learning_rates"]):
                new_lr = self.state["learning_rates"][current_phase - 1]
                for param_group in self.current_optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"📊 Updated learning rate to {new_lr}")
    
    def is_training_complete(self) -> bool:
        """Check if training is complete"""
        return self.state["current_phase"] > len(self.state["phase_sizes"])
    
    def get_current_examples(self, all_examples: List[Dict]) -> List[Dict]:
        """Get examples for current phase - works across different datasets"""
        if self.is_training_complete():
            return []
        
        current_phase = self.state["current_phase"]
        phase_size = self.state["phase_sizes"][current_phase - 1]
        
        # Handle case where new dataset is smaller than phase size
        effective_phase_size = min(phase_size, len(all_examples))
        
        print(f"📊 Phase {current_phase}: Using {effective_phase_size} examples (requested: {phase_size}, available: {len(all_examples)})")
        
        return all_examples[:effective_phase_size]
    
    def print_status(self):
        """Print current training status with clear dataset vs checkpoint distinction"""
        if self.is_training_complete():
            print("🎉 Training Complete!")
            return
        
        current_dataset = self.state.get("current_dataset", "Unknown")
        dataset_size = self.state.get("dataset_size", 0)
        current_phase = self.state["current_phase"]
        current_example = self.state["current_example"]
        current_epoch = self.state["current_epoch"]
        
        # Handle case where phase might exceed available phases (edge case)
        if current_phase <= len(self.state["phase_sizes"]):
            phase_size = self.state["phase_sizes"][current_phase - 1]
            epochs_for_phase = self.state["epochs_per_phase"][current_phase - 1]
        else:
            phase_size = "N/A"
            epochs_for_phase = "N/A"
        
        total_processed = self.state["total_examples_processed"]
        api_calls = self.state["api_calls_made"]
        estimated_cost = self.state["estimated_cost"]
        
        # Optional archived responses counter (backward compatible)
        archived_responses = self.state.get("archived_responses", 0)
        
        print(f"\n📊 Training Status:")
        print(f"   🗂️  Current Dataset: {current_dataset} ({dataset_size:,} examples)")
        print(f"   📈 Dataset Phase Progress: {current_phase}/{len(self.state['phase_sizes'])} (Phase {phase_size} examples)")
        print(f"   📚 Current Epoch: {current_epoch}/{epochs_for_phase}")
        print(f"   📝 Current Example: {current_example}/{phase_size}")
        print(f"")
        print(f"   🧠 Model Checkpoint Status:")
        print(f"   ✅ Total Historical Training: {total_processed} examples across all datasets")
        print(f"   ✅ Model weights preserved from previous training")
        print(f"   ✅ API Calls Made: {api_calls}")
        if archived_responses > 0:
            print(f"   ✅ Archived Responses: {archived_responses}")
        print(f"   ✅ Estimated Cost: ${estimated_cost:.2f}")
        
        if self.state["successful_batches"] > 0:
            avg_loss = self.state["total_loss"] / self.state["successful_batches"]
            print(f"   ✅ Average Loss: {avg_loss:.4f}")
        
        if self.state["last_saved"]:
            print(f"   💾 Last Saved: {self.state['last_saved']}")
        
        print(f"")
        print(f"   🔄 Note: Using trained model from {total_processed} examples with new dataset's phase structure")

class ProgressiveULDTrainer:
    """Enhanced Progressive ULD trainer with multi-dataset support"""
    
    def __init__(self, student_model_name: str = "Qwen/Qwen3-4B-Thinking-2507"):
        self.checkpoint = DistillationCheckpoint()
        self.student_model_name = student_model_name
        
        # Load model, tokenizer, optimizer
        self.model, self.tokenizer, self.optimizer = self.checkpoint.load_model_checkpoint(
            student_model_name
        )
        
        # Store reference to optimizer in checkpoint for LR updates
        self.checkpoint.current_optimizer = self.optimizer
        
        print("🎯 Enhanced Progressive ULD Trainer with Multi-Dataset Support")
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
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    return_attention_mask=True,
                    padding=True
                )
                
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
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
                if len(set(words)) < len(words) * 0.5:
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
            # Get teacher response (cached with archiving)
            teacher_response = self.checkpoint.get_teacher_response_cached(prompt)
            teacher_text, teacher_logits = self.parse_teacher_logits(teacher_response)
            
            full_text = prompt + " " + teacher_text
            inputs = self.tokenizer(
                full_text, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True, 
                padding=True,
                return_attention_mask=True
            )
            
            prompt_inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                return_attention_mask=True
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]
            
            # Student forward pass
            self.model.train()
            outputs = self.model(**inputs)
            student_logits = outputs.logits
            
            # Extract generation portion
            student_gen_logits = student_logits[0, prompt_len-1:-1, :]
            
            # Align sequences
            min_len = min(teacher_logits.shape[0], student_gen_logits.shape[0])
            if min_len <= 0:
                return None, "NO_TOKENS"
            
            teacher_aligned = teacher_logits[:min_len].unsqueeze(0)
            student_aligned = student_gen_logits[:min_len].unsqueeze(0)
            
            # Compute ULD loss
            uld_loss = self.compute_simple_uld_loss(teacher_aligned, student_aligned)
            
            # CE loss component
            labels = inputs["input_ids"][0, prompt_len:prompt_len+min_len]
            ce_loss = F.cross_entropy(
                student_aligned.view(-1, student_aligned.size(-1)), 
                labels.view(-1)
            )
            
            # Combined loss
            total_loss = 0.4 * ce_loss + 0.6 * uld_loss
            
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
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)
        
        teacher_sorted, _ = torch.sort(teacher_probs, dim=-1, descending=True)
        student_sorted, _ = torch.sort(student_probs, dim=-1, descending=True)
        
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
        
        # Extract token logprobs
        if "token_logprobs" in logprobs_data and "tokens" in logprobs_data:
            token_logprobs = logprobs_data["token_logprobs"]
            tokens = logprobs_data["tokens"]
            
            # Create simple logits tensor
            logits_list = []
            vocab_size = 50000  # Same as base code
            
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
        
        return generated_text, torch.stack(logits_list)
    
    def run_training(self, examples: List[Dict], dataset_name: str):
        """Run complete progressive training with selected dataset"""
        print(f"🚀 Starting Progressive ULD Training on {dataset_name}")
        
        # Update dataset configuration
        self.checkpoint.update_dataset_config(dataset_name, len(examples))
        
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
                
                if validation["repetitive"] > validation["coherent"]:
                    print("❌ Model degradation detected - stopping phase")
                    return False
            
            # Save checkpoint
            if self.checkpoint.should_save_checkpoint():
                self.checkpoint.save_model_checkpoint(self.model, self.tokenizer, self.optimizer)
                self.checkpoint.save_state()
            
            # Advance progress
            self.checkpoint.advance_progress()
            self.checkpoint.save_state()
        
        return True
    
    def save_final_model(self):
        """Save final trained model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_clean = self.checkpoint.state.get("current_dataset", "unknown").replace(" ", "_")
        final_dir = f"./qwen3-4b-uld-{dataset_clean}-{timestamp}"
        os.makedirs(final_dir, exist_ok=True)
        
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        
        summary = {
            "teacher_model": "moonshotai/Kimi-K2-Instruct-0905",
            "teacher_provider": "Together AI",
            "dataset_used": self.checkpoint.state.get("current_dataset", "unknown"),
            "dataset_size": self.checkpoint.state.get("dataset_size", 0),
            "training_completed": datetime.now().isoformat(),
            "total_examples_processed": self.checkpoint.state["total_examples_processed"],
            "total_phases": len(self.checkpoint.state["phase_sizes"]),
            "api_calls_made": self.checkpoint.state["api_calls_made"],
            "archived_responses": self.checkpoint.state.get("archived_responses", 0),
            "estimated_cost": self.checkpoint.state["estimated_cost"],
            "final_validation": self.validate_model()
        }
        
        with open(f"{final_dir}/training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Final model saved to {final_dir}")
        print(f"📊 Training Summary: {summary}")

# -------------------------------
# Enhanced Main Execution
# -------------------------------

if __name__ == "__main__":
    print("🚀 Enhanced ULD Trainer with Multi-Dataset Support & Non-Disruptive Archiving")
    print("=" * 60)
    
    # Verify Together AI API key
    if not os.getenv('TOGETHER_API_KEY'):
        print("❌ Please set TOGETHER_API_KEY environment variable")
        print("   export TOGETHER_API_KEY='your_api_key_here'")
        exit(1)
    
    # Create trainer first (loads existing progress if available)
    trainer = ProgressiveULDTrainer()
    
    # Show training status if resuming
    if Path("./distillation_checkpoints/training_state.json").exists():
        print("\n📂 Existing training progress found!")
        trainer.checkpoint.print_status()
        
        # Check if progress seems reset and offer manual restoration
        if trainer.checkpoint.state['current_phase'] == 1 and trainer.checkpoint.state['current_example'] == 0:
            if trainer.checkpoint.state['total_examples_processed'] > 0:
                print("\n⚠️  Progress appears to have been reset despite having processed examples!")
                restore = input("Do you want to manually restore your progress? (y/n): ").strip().lower()
                if restore == 'y':
                    try:
                        phase = int(input("Enter your current phase: "))
                        example = int(input("Enter your current example: "))
                        epoch = int(input("Enter your current epoch (default 1): ") or "1")
                        
                        trainer.checkpoint.state['current_phase'] = phase
                        trainer.checkpoint.state['current_example'] = example  
                        trainer.checkpoint.state['current_epoch'] = epoch
                        trainer.checkpoint.save_state()
                        
                        print(f"✅ Progress restored to Phase {phase}, Example {example}, Epoch {epoch}")
                    except ValueError:
                        print("❌ Invalid input, continuing with current progress")
        
        print("You can continue with the same progress on a new dataset of your choice.\n")
    
    # Always show dataset selection (even when resuming)
    dataset_choice = DatasetManager.show_dataset_menu()
    examples, dataset_name = DatasetManager.load_dataset(dataset_choice)
    
    if not examples:
        print("❌ No examples loaded. Exiting.")
        exit(1)
    
    print(f"\n🎯 Selected Dataset: {dataset_name}")
    print(f"📊 Total Examples Available: {len(examples):,}")
    print(f"📊 Dataset Size for Training: {len(examples):,} examples")
    
    # Run training (will update dataset config and continue from current progress)
    trainer.run_training(examples, dataset_name)
