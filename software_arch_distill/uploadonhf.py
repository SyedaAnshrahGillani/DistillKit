"""
Upload your distilled model to Hugging Face Hub - CORRECTED VERSION
"""

import os
import json
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Configuration
HF_USERNAME = "Anshrah"  # Replace with your Hugging Face username
MODEL_NAME = "qwen3-4b-software-arch-distilled_04"  # Choose your model name
LOCAL_MODEL_PATH = "./distillation_checkpoints/model"  # Path where your model is saved
CHECKPOINT_STATE_PATH = "./distillation_checkpoints/training_state.json"  # Training state file

# Full repository name
REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"

def load_training_stats():
    """Load actual training statistics from checkpoint"""
    try:
        with open(CHECKPOINT_STATE_PATH, 'r') as f:
            state = json.load(f)
        return state
    except Exception as e:
        print(f"Warning: Could not load training stats: {e}")
        return None

def upload_model_to_hf():
    """Upload the distilled model to Hugging Face Hub"""
    
    # Load actual training stats
    training_state = load_training_stats()
    
    # Initialize Hugging Face API
    api = HfApi()
    
    print(f"Uploading model to: {REPO_ID}")
    
    # Step 1: Create repository (if it doesn't exist)
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="model",
            private=False,  # Set to True if you want private repo
            exist_ok=True   # Don't fail if repo already exists
        )
        print(f"✓ Repository {REPO_ID} created/verified")
    except Exception as e:
        print(f"Repository creation: {e}")
    
    # Step 2: Create README.md with CORRECT model details
    if training_state:
        # Use actual training stats
        current_phase = training_state.get("current_phase", "Unknown")
        total_processed = training_state.get("total_examples_processed", 0)
        api_calls = training_state.get("api_calls_made", 0)
        estimated_cost = training_state.get("estimated_cost", 0.0)
        avg_loss = training_state.get("total_loss", 0) / max(training_state.get("successful_batches", 1), 1)
        
        training_stats_section = f"""- **Current Phase**: {current_phase}/{len(training_state.get('phase_sizes', []))}
- **Examples Processed**: {total_processed}
- **API Calls Made**: {api_calls}
- **Estimated Training Cost**: ${estimated_cost:.2f}
- **Average Loss**: {avg_loss:.4f}
- **Training Status**: {"Complete" if training_state.get("current_phase", 1) > len(training_state.get("phase_sizes", [])) else "In Progress"}"""
    else:
        training_stats_section = "- **Training Stats**: Not available (checkpoint state file missing)"
    
    readme_content = f"""---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Thinking-2507
tags:
- knowledge-distillation
- software-architecture
- qwen3
- distilled
- uld
- together-ai
- kimi
language:
- en
pipeline_tag: text-generation
---

# {MODEL_NAME}

This model is a knowledge-distilled version of Qwen/Qwen3-4B-Thinking-2507, trained on software architecture content using **ULD (Unified Logits Distillation)** from **Kimi K2** via Together AI.

## Model Details

- **Base Model**: Qwen/Qwen3-4B-Thinking-2507
- **Teacher Model**: moonshotai/Kimi-K2-Instruct-0905 (via Together AI)
- **Training Method**: ULD (Unified Logits Distillation) + Cross-Entropy fallback
- **Dataset**: ajibawa-2023/Software-Architecture
- **Training Framework**: Progressive phases with comprehensive checkpointing

## Training Statistics

{training_stats_section}

## Training Process

The model was trained using a progressive ULD (Unified Logits Distillation) approach:

1. **Progressive Phases**: [10, 50, 200, 1000] examples with [1, 2, 2, 3] epochs each
2. **Teacher API**: Together AI's Kimi K2 model generates responses with logprobs
3. **ULD Loss**: Student learns to match teacher's probability distributions
4. **Fallback**: Cross-entropy loss when logprobs unavailable
5. **Combined Loss**: 40% Cross-Entropy + 60% ULD (optimal ratio)

## Key Features

- ✅ **Comprehensive Checkpointing**: Resume training from any point
- ✅ **API Cost Optimization**: Cached teacher responses
- ✅ **Progressive Training**: Start small, scale up gradually  
- ✅ **Quality Monitoring**: Automatic validation every 20 examples
- ✅ **Gradient Stability**: Clipping and optimal learning rates

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{REPO_ID}")
model = AutoModelForCausalLM.from_pretrained("{REPO_ID}")

# Generate text
prompt = "What are the key principles of microservices architecture?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Configuration

- **Learning Rates**: [5e-6, 3e-6, 2e-6, 1e-6] (decreasing per phase)
- **Max Tokens**: 512 (optimal for cost vs quality)
- **Batch Size**: 1
- **Optimizer**: AdamW with weight decay (1e-5)
- **Precision**: bfloat16
- **Gradient Clipping**: max_norm=1.0

## Intended Use

This model is optimized for software architecture discussions, explanations, and Q&A. It should perform better than the base model on architecture-related topics while maintaining general capabilities.

## Limitations

- Trained on software architecture domain (may have domain bias)
- Uses simplified ULD approximation (not full ULD implementation)
- Teacher model biases inherited from Kimi K2
- Performance on non-architecture topics may vary

## Training Infrastructure

- **Teacher API**: Together AI (Kimi K2 model)
- **Checkpointing**: Complete state management with resume capability
- **Cost Optimization**: Response caching to minimize API calls
- **Quality Control**: Automatic validation and degradation detection

## Citation

If you use this model, please cite:

```bibtex
@misc{{{MODEL_NAME.replace('-', '_')},
  title={{{MODEL_NAME}: ULD Knowledge Distilled Qwen3-4B for Software Architecture}},
  author={{Anshrah}},
  year={{2025}},
  url={{https://huggingface.co/{REPO_ID}}},
  note={{Distilled using Together AI Kimi K2 with ULD methodology}}
}}
```

## Acknowledgments

- Base model: Qwen team
- Teacher model: Moonshot AI (Kimi K2)
- API provider: Together AI
- Dataset: ajibawa-2023
- Method: ULD (Unified Logits Distillation)
"""
    
    # Save README
    readme_path = os.path.join(LOCAL_MODEL_PATH, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("✓ README.md created with correct information")
    
    # Step 3: Copy training summary if available
    if training_state:
        summary_path = os.path.join(LOCAL_MODEL_PATH, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(training_state, f, indent=2)
        print("✓ Training summary added")
    
    # Step 4: Upload the entire folder
    try:
        api.upload_folder(
            folder_path=LOCAL_MODEL_PATH,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Upload ULD distilled Qwen3-4B model (Kimi K2 teacher)"
        )
        print(f"✓ Model uploaded successfully!")
        print(f"🎉 Your model is available at: https://huggingface.co/{REPO_ID}")
        
    except Exception as e:
        print(f"Upload failed: {e}")
        print("Make sure you're logged in: huggingface-cli login")

def verify_model_locally():
    """Test loading the model locally before upload"""
    print("Testing local model loading...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, torch_dtype="auto")
        print("✓ Model loads successfully")
        
        # Test generation
        test_prompt = "What is software architecture?"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50, do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Test generation: {response[:100]}...")
        
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Preparing to upload ULD distilled model to Hugging Face...")
    
    # Check if logged in
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Logged in as: {user_info['name']}")
    except Exception:
        print("❌ Not logged in to Hugging Face")
        print("Please run: huggingface-cli login")
        exit(1)
    
    # Verify model exists
    if not Path(LOCAL_MODEL_PATH).exists():
        print(f"❌ Model path not found: {LOCAL_MODEL_PATH}")
        print("Make sure training has saved at least one checkpoint")
        exit(1)
    
    # Verify model works locally
    if not verify_model_locally():
        print("❌ Model verification failed. Check your model files.")
        exit(1)
    
    # Upload model
    upload_model_to_hf()
    
    print("\n🎯 Next steps:")
    print("1. Visit your model page and test it")
    print("2. Update the model card with examples")
    print("3. Consider making it featured if performance is good")
    print(f"4. Share: https://huggingface.co/{REPO_ID}")
