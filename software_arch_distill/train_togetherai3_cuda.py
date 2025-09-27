"""
Enhanced ULD Distillation with Per-Dataset Progress Tracking - GPU Enabled
Uses Together AI API with Kimi K2 model for teacher logits
Each dataset maintains separate progress while sharing model weights
Optimized for CUDA GPU training
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

# Enable optimized CUDA operations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class DatasetManager:
    """Manages multiple datasets with different formats"""
    
    AVAILABLE_DATASETS = {
        "1": {
            "name": "Software Architecture (Public JSONL)",
            "repo_id": "ajibawa-2023/Software-Architecture",
            "filename": "Software_Architecture_Final.jsonl",
            "format": "jsonl",
            "description": "Public software architecture Q&A dataset - 400k examples",
            "dataset_key": "software_architecture"
        },
        "2": {
            "name": "Cloud Dataset by Hamed (Public JSON)",
            "repo_id": "Anshrah/Cloud-Dataset-byHamed",
            "filename": "dataset.json",
            "format": "json",
            "description": "Cloud dataset shared by hamed",
            "dataset_key": "cloud_dataset_hamed"
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
                choice = input("Select dataset (1-2): ").strip()
                if choice in DatasetManager.AVAILABLE_DATASETS:
                    return choice
                else:
                    print("❌ Invalid choice. Please select 1 or 2.")
            except KeyboardInterrupt:
                print("\n👋 Training cancelled.")
                exit(0)
    
    @staticmethod
    def load_dataset(choice: str) -> Tuple[List[Dict], str, str]:
        """Load selected dataset and return examples with dataset info and key"""
        dataset_config = DatasetManager.AVAILABLE_DATASETS[choice]
        dataset_name = dataset_config["name"]
        dataset_key = dataset_config["dataset_key"]
        
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
                    return [], "", ""
        
        print(f"✅ Loaded {len(examples):,} examples from {dataset_name}")
        print(f"📊 Total Dataset Size: {len(examples):,} examples")
        return examples, dataset_name, dataset_key
    
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
        """Calculate optimal phases based on dataset size"""
        if dataset_size <= 100:
            phases = [10, 25, 50, min(dataset_size, 100)]
            epochs = [2, 3, 3, 4]
            learning_rates = [1e-5, 8e-6, 6e-6, 4e-6]
        elif dataset_size <= 1000:
            phases = [20, 100, 300, min(dataset_size, 1000)]
            epochs = [2, 2, 3, 3]
            learning_rates = [8e-6, 6e-6, 4e-6, 3e-6]
        elif dataset_size <= 10000:
            phases = [50, 200, 1000, 5000, min(dataset_size, 10000)]
            epochs = [1, 2, 2, 2, 3]
            learning_rates = [8e-6, 6e-6, 4e-6, 3e-6, 2e-6]
        else:
            phases = [100, 500, 2000, 10000, 50000, min(dataset_size, 100000)]
            epochs = [1, 1, 2, 2, 2, 3]
            learning_rates = [8e-6, 6e-6, 4e-6, 3e-6, 2e-6, 1e-6]
        
        # Ensure phases don't exceed dataset size
        phases = [min(phase, dataset_size) for phase in phases]
        
        print(f"📊 Dynamic phases calculated for {dataset_size} examples:")
        print(f"   Phases: {phases}")
        print(f"   Epochs per phase: {epochs}")
        print(f"   Learning rates: {learning_rates}")
        
        return phases, epochs, learning_rates

class DistillationCheckpoint:
    """Enhanced checkpoint management with per-dataset progress tracking"""
    
    def __init__(self, checkpoint_dir: str = "./distillation_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # File paths
        self.state_file = self.checkpoint_dir / "training_state.json"
        self.model_dir = self.checkpoint_dir / "model"
        self.optimizer_file = self.checkpoint_dir / "optimizer.pt"
        self.cache_dir = self.checkpoint_dir / "teacher_cache"
        self.archive_dir = self.checkpoint_dir / "teacher_archive"
        self.cache_dir.mkdir(exist_ok=True)
        self.archive_dir.mkdir(exist_ok=True)
        
        # Initialize state
        self.state = self.load_state()
        
        # Current dataset being worked on
        self.current_dataset_key = None
    
    def load_state(self) -> Dict[str, Any]:
        """Load training state or create new one"""
        if self.state_file.exists():
            print(f"📂 Loading existing training state from {self.state_file}")
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                print(f"🔄 Found training history for multiple datasets")
                return state
        else:
            print("🆕 Creating new training state")
            return {
                "created_at": datetime.now().isoformat(),
                "global_stats": {
                    "total_examples_processed": 0,
                    "successful_batches": 0,
                    "total_loss": 0.0,
                    "api_calls_made": 0,
                    "estimated_cost": 0.0,
                    "archived_responses": 0
                },
                "datasets": {},  # Per-dataset progress tracking
                "training_history": [],
                "last_saved": None
            }
    
    def migrate_legacy_state(self):
        """Migrate from old single-dataset format to new per-dataset format"""
        # Check if this is legacy format (has current_dataset, current_phase etc. at root level)
        if "current_dataset" in self.state and "datasets" not in self.state:
            print("🔄 Migrating legacy training state to per-dataset format...")
            
            # Extract legacy data
            legacy_data = {
                "current_dataset": self.state.get("current_dataset", "Unknown"),
                "dataset_size": self.state.get("dataset_size", 0),
                "current_phase": self.state.get("current_phase", 1),
                "current_example": self.state.get("current_example", 0),
                "current_epoch": self.state.get("current_epoch", 1),
                "phase_sizes": self.state.get("phase_sizes", [10, 50, 200, 1000]),
                "epochs_per_phase": self.state.get("epochs_per_phase", [1, 2, 2, 3]),
                "learning_rates": self.state.get("learning_rates", [5e-6, 3e-6, 2e-6, 5e-6])
            }
            
            # Create new structure
            new_state = {
                "created_at": self.state.get("created_at", datetime.now().isoformat()),
                "global_stats": {
                    "total_examples_processed": self.state.get("total_examples_processed", 0),
                    "successful_batches": self.state.get("successful_batches", 0),
                    "total_loss": self.state.get("total_loss", 0.0),
                    "api_calls_made": self.state.get("api_calls_made", 0),
                    "estimated_cost": self.state.get("estimated_cost", 0.0),
                    "archived_responses": self.state.get("archived_responses", 0)
                },
                "datasets": {},
                "training_history": self.state.get("training_history", []),
                "last_saved": self.state.get("last_saved")
            }
            
            # Try to map legacy data to dataset key based on characteristics
            # If we have significant progress, assume it's dataset 1 (Software Architecture)
            if legacy_data["current_phase"] > 1 or legacy_data["current_example"] > 0:
                dataset_key = "software_architecture"  # Default to dataset 1
                new_state["datasets"][dataset_key] = {
                    "dataset_name": "Software Architecture (Legacy Migration)",
                    "dataset_size": legacy_data["dataset_size"],
                    "current_phase": legacy_data["current_phase"],
                    "current_example": legacy_data["current_example"],
                    "current_epoch": legacy_data["current_epoch"],
                    "phase_sizes": legacy_data["phase_sizes"],
                    "epochs_per_phase": legacy_data["epochs_per_phase"],
                    "learning_rates": legacy_data["learning_rates"],
                    "last_trained": self.state.get("last_saved"),
                    "examples_processed": 0  # Will be calculated
                }
                print(f"✅ Migrated progress to dataset '{dataset_key}'")
                print(f"   Phase: {legacy_data['current_phase']}, Example: {legacy_data['current_example']}")
            
            # Replace old state
            self.state = new_state
            self.save_state()
            print("✅ Migration completed successfully")
    
    def get_dataset_progress(self, dataset_key: str) -> Dict[str, Any]:
        """Get progress for specific dataset"""
        if dataset_key not in self.state["datasets"]:
            return {
                "dataset_name": "Unknown",
                "dataset_size": 0,
                "current_phase": 1,
                "current_example": 0,
                "current_epoch": 1,
                "phase_sizes": [10, 50, 200, 1000],
                "epochs_per_phase": [1, 2, 2, 3],
                "learning_rates": [5e-6, 3e-6, 2e-6, 5e-6],
                "last_trained": None,
                "examples_processed": 0
            }
        return self.state["datasets"][dataset_key]
    
    def update_dataset_config(self, dataset_key: str, dataset_name: str, dataset_size: int):
        """Update dataset configuration and switch to it"""
        # Migrate legacy state first if needed
        if "current_dataset" in self.state:
            self.migrate_legacy_state()
        
        self.current_dataset_key = dataset_key
        
        # Check if this dataset exists
        if dataset_key in self.state["datasets"]:
            dataset_progress = self.state["datasets"][dataset_key]
            print(f"📋 Resuming dataset: {dataset_name}")
            print(f"🔄 Progress: Phase {dataset_progress['current_phase']}, "
                  f"Example {dataset_progress['current_example']}, "
                  f"Epoch {dataset_progress['current_epoch']}")
            
            # Update dataset size in case it changed
            self.state["datasets"][dataset_key]["dataset_size"] = dataset_size
            
        else:
            # New dataset - create fresh progress
            phases, epochs, learning_rates = DatasetManager.calculate_dynamic_phases(dataset_size)
            
            self.state["datasets"][dataset_key] = {
                "dataset_name": dataset_name,
                "dataset_size": dataset_size,
                "current_phase": 1,
                "current_example": 0,
                "current_epoch": 1,
                "phase_sizes": phases,
                "epochs_per_phase": epochs,
                "learning_rates": learning_rates,
                "last_trained": None,
                "examples_processed": 0
            }
            
            print(f"📋 New dataset initialized: {dataset_name}")
            print(f"🔄 Starting from Phase 1, Example 0")
        
        self.save_state()
    
    def get_current_progress(self) -> Dict[str, Any]:
        """Get progress for currently active dataset"""
        if not self.current_dataset_key:
            return self.get_dataset_progress("unknown")
        return self.get_dataset_progress(self.current_dataset_key)
    
    def save_state(self):
        """Save current training state"""
        self.state["last_saved"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        
        current_progress = self.get_current_progress()
        print(f"💾 State saved: Dataset '{current_progress.get('dataset_name', 'Unknown')}', "
              f"Phase {current_progress.get('current_phase', 1)}, "
              f"Example {current_progress.get('current_example', 0)}")
    
    def save_model_checkpoint(self, model, tokenizer, optimizer, device):
        """Save model, tokenizer, and optimizer state"""
        print(f"💾 Saving model checkpoint...")
        
        model.save_pretrained(self.model_dir)
        tokenizer.save_pretrained(self.model_dir)
        
        # Move optimizer state to CPU before saving to avoid GPU memory issues
        optimizer_state = {
            'optimizer_state_dict': {k: v.cpu() if torch.is_tensor(v) else v for k, v in optimizer.state_dict().items()},
            'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        }
        
        torch.save(optimizer_state, self.optimizer_file)
        
        print(f"✅ Model checkpoint saved to {self.model_dir}")
    
    def load_model_checkpoint(self, model_name: str, device: torch.device):
        """Load model, tokenizer, and optimizer from checkpoint"""
        if self.model_dir.exists() and any(self.model_dir.iterdir()):
            print(f"📂 Loading model from checkpoint...")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_dir, 
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            current_progress = self.get_current_progress()
            current_phase = current_progress.get("current_phase", 1)
            learning_rates = current_progress.get("learning_rates", [5e-6])
            lr_index = min(current_phase - 1, len(learning_rates) - 1)
            lr = learning_rates[lr_index]
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            
            if self.optimizer_file.exists():
                checkpoint = torch.load(self.optimizer_file, map_location=device)
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
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            current_progress = self.get_current_progress()
            current_phase = current_progress.get("current_phase", 1)
            learning_rates = current_progress.get("learning_rates", [5e-6])
            lr_index = min(current_phase - 1, len(learning_rates) - 1)
            lr = learning_rates[lr_index]
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            
            return model, tokenizer, optimizer
    
    def get_teacher_response_cached(self, prompt: str) -> Dict:
        """Cached teacher responses with non-disruptive archiving"""
        cache_key = str(abs(hash(prompt)))
        cache_file = self.cache_dir / f"teacher_{cache_key}.json"
        
        if cache_file.exists():
            print(f"📂 Using cached teacher response")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        print(f"🌐 Making API call to teacher (Together AI)...")
        response = self.call_teacher_api(prompt)
        
        with open(cache_file, 'w') as f:
            json.dump(response, f)
        
        self.state["global_stats"]["api_calls_made"] += 1
        self.state["global_stats"]["estimated_cost"] += 0.010
        
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
            "model": "moonshotai/Kimi-K2-Instruct-0905",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.0,
            "logprobs": 20,
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
        current_progress = self.get_current_progress()
        
        progress_entry = {
            "timestamp": datetime.now().isoformat(),
            "dataset_key": self.current_dataset_key,
            "dataset": current_progress.get("dataset_name", "unknown"),
            "phase": current_progress.get("current_phase", 1),
            "example": current_progress.get("current_example", 0),
            "epoch": current_progress.get("current_epoch", 1),
            "loss": loss,
            "mode": mode,
            "total_examples_processed": self.state["global_stats"]["total_examples_processed"],
            "validation": validation_results
        }
        
        self.state["training_history"].append(progress_entry)
        self.state["global_stats"]["total_loss"] += loss
        self.state["global_stats"]["successful_batches"] += 1
        
        if len(self.state["training_history"]) > 1000:
            self.state["training_history"] = self.state["training_history"][-1000:]
    
    def should_validate(self) -> bool:
        """Determine if we should run validation"""
        current_progress = self.get_current_progress()
        return current_progress.get("current_example", 0) % 20 == 0
    
    def should_save_checkpoint(self) -> bool:
        """Determine if we should save checkpoint"""
        current_progress = self.get_current_progress()
        return current_progress.get("current_example", 0) % 30 == 0
    
    def advance_progress(self):
        """Advance to next example/epoch/phase for current dataset"""
        if not self.current_dataset_key:
            return
        
        dataset_progress = self.state["datasets"][self.current_dataset_key]
        
        dataset_progress["current_example"] += 1
        dataset_progress["examples_processed"] += 1
        self.state["global_stats"]["total_examples_processed"] += 1
        
        current_phase_size = dataset_progress["phase_sizes"][dataset_progress["current_phase"] - 1]
        epochs_for_phase = dataset_progress["epochs_per_phase"][dataset_progress["current_phase"] - 1]
        
        if dataset_progress["current_example"] >= current_phase_size:
            if dataset_progress["current_epoch"] >= epochs_for_phase:
                dataset_progress["current_phase"] += 1
                dataset_progress["current_example"] = 0
                dataset_progress["current_epoch"] = 1
                print(f"🚀 Advanced to Phase {dataset_progress['current_phase']}")
                self.update_optimizer_lr()
            else:
                dataset_progress["current_epoch"] += 1
                dataset_progress["current_example"] = 0
                print(f"📚 Advanced to Epoch {dataset_progress['current_epoch']} of Phase {dataset_progress['current_phase']}")
        
        dataset_progress["last_trained"] = datetime.now().isoformat()
    
    def update_optimizer_lr(self):
        """Update optimizer learning rate for current phase"""
        if hasattr(self, 'current_optimizer'):
            current_progress = self.get_current_progress()
            current_phase = current_progress.get("current_phase", 1)
            learning_rates = current_progress.get("learning_rates", [5e-6])
            
            if current_phase <= len(learning_rates):
                new_lr = learning_rates[current_phase - 1]
                for param_group in self.current_optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"📊 Updated learning rate to {new_lr}")
    
    def is_training_complete(self) -> bool:
        """Check if training is complete for current dataset"""
        current_progress = self.get_current_progress()
        current_phase = current_progress.get("current_phase", 1)
        phase_sizes = current_progress.get("phase_sizes", [])
        return current_phase > len(phase_sizes)
    
    def get_current_examples(self, all_examples: List[Dict]) -> List[Dict]:
        """Get examples for current phase"""
        if self.is_training_complete():
            return []
        
        current_progress = self.get_current_progress()
        current_phase = current_progress.get("current_phase", 1)
        phase_sizes = current_progress.get("phase_sizes", [])
        
        if current_phase > len(phase_sizes):
            return []
        
        phase_size = phase_sizes[current_phase - 1]
        effective_phase_size = min(phase_size, len(all_examples))
        
        print(f"📊 Phase {current_phase}: Using {effective_phase_size} examples "
              f"(requested: {phase_size}, available: {len(all_examples)})")
        
        return all_examples[:effective_phase_size]
    
    def print_status(self):
        """Print current training status with per-dataset breakdown"""
        print(f"\n📊 Training Status:")
        
        # Global stats
        global_stats = self.state["global_stats"]
        print(f"   🌍 Global Progress:")
        print(f"   ✅ Total Examples Processed: {global_stats['total_examples_processed']}")
        print(f"   ✅ API Calls Made: {global_stats['api_calls_made']}")
        print(f"   ✅ Estimated Cost: ${global_stats['estimated_cost']:.2f}")
        
        if global_stats['successful_batches'] > 0:
            avg_loss = global_stats['total_loss'] / global_stats['successful_batches']
            print(f"   ✅ Average Loss: {avg_loss:.4f}")
        
        # Per-dataset breakdown
        print(f"\n   📂 Dataset Progress:")
        for dataset_key, progress in self.state["datasets"].items():
            status_icon = "🔄" if dataset_key == self.current_dataset_key else "📋"
            print(f"   {status_icon} {progress['dataset_name']} ({progress['dataset_size']:,} examples)")
            print(f"      Phase: {progress['current_phase']}/{len(progress['phase_sizes'])}, "
                  f"Example: {progress['current_example']}, "
                  f"Epoch: {progress['current_epoch']}")
            if progress['last_trained']:
                print(f"      Last trained: {progress['last_trained']}")
            
        # Current dataset detailed status
        if self.current_dataset_key:
            current_progress = self.get_current_progress()
            print(f"\n   🎯 Current Dataset Details:")
            print(f"   Dataset: {current_progress['dataset_name']}")
            
            if current_progress['current_phase'] <= len(current_progress['phase_sizes']):
                phase_size = current_progress['phase_sizes'][current_progress['current_phase'] - 1]
                epochs_for_phase = current_progress['epochs_per_phase'][current_progress['current_phase'] - 1]
                print(f"   Phase: {current_progress['current_phase']}/{len(current_progress['phase_sizes'])} "
                      f"({phase_size} examples)")
                print(f"   Epoch: {current_progress['current_epoch']}/{epochs_for_phase}")
                print(f"   Example: {current_progress['current_example']}/{phase_size}")
            else:
                print(f"   🎉 Training Complete!")

class ProgressiveULDTrainer:
    """Enhanced Progressive ULD trainer with per-dataset progress tracking and GPU optimization"""
    
    def __init__(self, student_model_name: str = "Qwen/Qwen3-4B-Thinking-2507"):
        # Initialize device detection and GPU optimization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")
        
        if self.device.type == "cuda":
            print(f"🚀 GPU Details:")
            print(f"   GPU Name: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"   CUDA Version: {torch.version.cuda}")
            
            # Enable memory optimization
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_allocated'):
                print(f"   Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        self.checkpoint = DistillationCheckpoint()
        self.student_model_name = student_model_name
        
        # Migrate legacy state if needed
        if "current_dataset" in self.checkpoint.state:
            self.checkpoint.migrate_legacy_state()
        
        self.model, self.tokenizer, self.optimizer = self.checkpoint.load_model_checkpoint(
            student_model_name, self.device
        )
        
        # Ensure model is on correct device
        if not next(self.model.parameters()).device == self.device:
            print(f"📦 Moving model to {self.device}...")
            self.model = self.model.to(self.device)
        
        print(f"✅ Model loaded on {next(self.model.parameters()).device}")
        
        self.checkpoint.current_optimizer = self.optimizer
        
        # Store vocab size for use in logits parsing
        self.vocab_size = self.tokenizer.vocab_size
        print(f"📝 Vocab Size: {self.vocab_size}")
        
        print("🎯 Enhanced Progressive ULD Trainer with GPU Acceleration")
        self.checkpoint.print_status()
    
    def validate_model(self) -> Dict:
        """Validate model generation quality with GPU support"""
        test_prompts = [
            "What is software architecture?",
            "Explain microservices briefly.",
            "How do you design scalable systems?"
        ]

        self.model.eval()
        results = {"coherent": 0, "repetitive": 0, "failed": 0}
    
        with torch.no_grad():  # Disable gradients for validation
            for prompt in test_prompts:
                try:
                    inputs = self.tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        return_attention_mask=True,
                        padding=True
                    )
                
                    # Move inputs to GPU - keep as BatchEncoding object
                    inputs.input_ids = inputs.input_ids.to(self.device)
                    inputs.attention_mask = inputs.attention_mask.to(self.device)
                
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
                
                    words = generated.split()[:20]
                    if len(set(words)) < len(words) * 0.5:
                        results["repetitive"] += 1
                    elif len(generated.strip()) > 10:
                        results["coherent"] += 1
                    else:
                        results["failed"] += 1
                    
                except Exception as e:
                    print(f"⚠️ Validation failed for prompt: {e}")
                    results["failed"] += 1
    
        return results
    
    def train_step(self, prompt: str, target: str) -> Tuple[Optional[float], str]:
        """Single training step with ULD and GPU optimization"""
        try:
            teacher_response = self.checkpoint.get_teacher_response_cached(prompt)
            teacher_text, teacher_logits = self.parse_teacher_logits(teacher_response)
            
            # Move teacher logits to GPU
            teacher_logits = teacher_logits.to(self.device)
            
            full_text = prompt + " " + teacher_text
            inputs = self.tokenizer(
                full_text, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True, 
                padding=True,
                return_attention_mask=True
            )
            
            # Move inputs to GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            prompt_inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                return_attention_mask=True
            )
            # Move prompt inputs to GPU
            prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}
            prompt_len = prompt_inputs["input_ids"].shape[1]
            
            self.model.train()
            outputs = self.model(**inputs)
            student_logits = outputs.logits
            
            student_gen_logits = student_logits[0, prompt_len-1:-1, :]
            
            min_len = min(teacher_logits.shape[0], student_gen_logits.shape[0])
            if min_len <= 0:
                return None, "NO_TOKENS"
            
            teacher_aligned = teacher_logits[:min_len].unsqueeze(0)
            student_aligned = student_gen_logits[:min_len].unsqueeze(0)
            
            uld_loss = self.compute_simple_uld_loss(teacher_aligned, student_aligned)
            
            labels = inputs["input_ids"][0, prompt_len:prompt_len+min_len]
            ce_loss = F.cross_entropy(
                student_aligned.view(-1, student_aligned.size(-1)), 
                labels.view(-1)
            )
            
            total_loss = 0.4 * ce_loss + 0.6 * uld_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Clear cache to prevent memory buildup
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            return total_loss.item(), "ULD"
            
        except Exception as e:
            print(f"⚠️ Training step failed: {e}")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return None, "FAILED"
    
    def compute_simple_uld_loss(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
        """Simplified ULD loss computation with GPU optimization"""
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)
        
        teacher_sorted, _ = torch.sort(teacher_probs, dim=-1, descending=True)
        student_sorted, _ = torch.sort(student_probs, dim=-1, descending=True)
        
        max_vocab = max(teacher_sorted.size(-1), student_sorted.size(-1))
        teacher_padded = F.pad(teacher_sorted, (0, max_vocab - teacher_sorted.size(-1)))
        student_padded = F.pad(student_sorted, (0, max_vocab - student_sorted.size(-1)))
        
        return torch.mean(torch.abs(teacher_padded - student_padded))
    
    def parse_teacher_logits(self, response_data: Dict) -> Tuple[str, torch.Tensor]:
        """Parse Together AI teacher response to extract logits with GPU support"""
        choice = response_data["choices"][0]
        generated_text = choice["message"]["content"]
        
        if "logprobs" not in choice or not choice["logprobs"]:
            raise RuntimeError("No logprobs in Together AI teacher response")
        
        logprobs_data = choice["logprobs"]
        
        if "token_logprobs" in logprobs_data and "tokens" in logprobs_data:
            token_logprobs = logprobs_data["token_logprobs"]
            tokens = logprobs_data["tokens"]
            
            logits_list = []
            vocab_size = self.vocab_size
            
            for i, (token, logprob) in enumerate(zip(tokens, token_logprobs)):
                if logprob is None:
                    continue
                    
                # Create logits tensor on the correct device
                logits = torch.full((vocab_size,), -10.0, device=self.device)
                logits[i % vocab_size] = logprob
                
                if "top_logprobs" in logprobs_data and i < len(logprobs_data["top_logprobs"]):
                    top_logprobs = logprobs_data["top_logprobs"][i]
                    if top_logprobs:
                        for j, (top_token, top_logprob) in enumerate(top_logprobs.items()):
                            if j < vocab_size:
                                logits[j] = top_logprob
                
                logits_list.append(logits)
        
        elif "content" in logprobs_data:
            content_logprobs = logprobs_data["content"]
            logits_list = []
            vocab_size = self.vocab_size
            
            for token_data in content_logprobs:
                # Create logits tensor on the correct device
                logits = torch.full((vocab_size,), -10.0, device=self.device)
                
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
    
    def run_training(self, examples: List[Dict], dataset_name: str, dataset_key: str):
        """Run complete progressive training with selected dataset and GPU optimization"""
        print(f"🚀 Starting Progressive ULD Training on {dataset_name}")
        
        # Switch to this dataset and load its progress
        self.checkpoint.update_dataset_config(dataset_key, dataset_name, len(examples))
        
        while not self.checkpoint.is_training_complete():
            self.checkpoint.print_status()
            
            current_examples = self.checkpoint.get_current_examples(examples)
            
            if not current_examples:
                print("❌ No examples for current phase")
                break
            
            phase_success = self.train_phase(current_examples)
            
            if not phase_success:
                print("❌ Phase training failed - stopping")
                break
            
            current_progress = self.checkpoint.get_current_progress()
            print(f"✅ Phase {current_progress['current_phase']} completed successfully")
        
        if self.checkpoint.is_training_complete():
            print("🎉 Training completed successfully!")
            self.save_final_model()
    
    def train_phase(self, examples: List[Dict]) -> bool:
        """Train on current phase examples with GPU optimization"""
        current_progress = self.checkpoint.get_current_progress()
        start_idx = current_progress.get("current_example", 0)
        
        for i in range(start_idx, len(examples)):
            example = examples[i]
            prompt = example.get("input", "")
            target = example.get("output", "")
            
            if len(prompt.strip()) < 10:
                self.checkpoint.advance_progress()
                continue
            
            loss, mode = self.train_step(prompt, target)
            
            if loss is not None:
                self.checkpoint.log_progress(loss, mode)
                print(f"📈 Step {i}: Loss={loss:.4f}, Mode={mode}")
                
                # Show GPU memory usage periodically
                if self.device.type == "cuda" and i % 10 == 0:
                    memory_used = torch.cuda.memory_allocated() / 1e9
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"🖥️ GPU Memory: {memory_used:.2f}/{memory_total:.1f} GB")
            
            if self.checkpoint.should_validate():
                validation = self.validate_model()
                self.checkpoint.log_progress(loss or 0, mode, validation)
                print(f"🔍 Validation: {validation}")
                
                if validation["repetitive"] > validation["coherent"]:
                    print("❌ Model degradation detected - stopping phase")
                    return False
            
            if self.checkpoint.should_save_checkpoint():
                self.checkpoint.save_model_checkpoint(self.model, self.tokenizer, self.optimizer, self.device)
                self.checkpoint.save_state()
            
            self.checkpoint.advance_progress()
            self.checkpoint.save_state()
        
        return True
    
    def save_final_model(self):
        """Save final trained model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        current_progress = self.checkpoint.get_current_progress()
        dataset_clean = current_progress.get("dataset_name", "unknown").replace(" ", "_")
        final_dir = f"./qwen3-4b-uld-{dataset_clean}-{timestamp}"
        os.makedirs(final_dir, exist_ok=True)
        
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        
        summary = {
            "teacher_model": "moonshotai/Kimi-K2-Instruct-0905",
            "teacher_provider": "Together AI",
            "dataset_used": current_progress.get("dataset_name", "unknown"),
            "dataset_size": current_progress.get("dataset_size", 0),
            "training_completed": datetime.now().isoformat(),
            "device_used": str(self.device),
            "global_stats": self.checkpoint.state["global_stats"],
            "datasets_trained": {k: v for k, v in self.checkpoint.state["datasets"].items()},
            "final_validation": self.validate_model()
        }
        
        with open(f"{final_dir}/training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Final model saved to {final_dir}")
        print(f"📊 Training Summary: {summary}")

if __name__ == "__main__":
    print("🚀 Enhanced ULD Trainer with Per-Dataset Progress Tracking and GPU Acceleration")
    print("=" * 70)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA is available!")
        print(f"   Device: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ CUDA not available, will use CPU")
    
    if not os.getenv('TOGETHER_API_KEY'):
        print("❌ Please set TOGETHER_API_KEY environment variable")
        print("   export TOGETHER_API_KEY='your_api_key_here'")
        exit(1)
    
    trainer = ProgressiveULDTrainer()
    
    if Path("./distillation_checkpoints/training_state.json").exists():
        print("\n📂 Existing training progress found!")
        trainer.checkpoint.print_status()
        print("Each dataset maintains separate progress while sharing model weights.\n")
    
    dataset_choice = DatasetManager.show_dataset_menu()
    examples, dataset_name, dataset_key = DatasetManager.load_dataset(dataset_choice)
    
    if not examples:
        print("❌ No examples loaded. Exiting.")
        exit(1)
    
    print(f"\n🎯 Selected Dataset: {dataset_name}")
    print(f"📊 Total Examples Available: {len(examples):,}")
    
    trainer.run_training(examples, dataset_name, dataset_key)
