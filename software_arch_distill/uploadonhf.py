"""
Upload your distilled model to Hugging Face Hub
"""

import os
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
HF_USERNAME = "Anshrah"  # Replace with your Hugging Face username
MODEL_NAME = "qwen3-4b-software-arch-distilled_02"  # Choose your model name
LOCAL_MODEL_PATH = "./qwen3-4b-kd-finetuned"  # Path where your model is saved

# Full repository name
REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"

def upload_model_to_hf():
    """Upload the distilled model to Hugging Face Hub"""
    
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
    
    # Step 2: Create README.md with model details
    readme_content = f"""---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Thinking-2507
tags:
- knowledge-distillation
- software-architecture
- qwen3
- distilled
language:
- en
pipeline_tag: text-generation
---

# {MODEL_NAME}

This model is a knowledge-distilled version of Qwen/Qwen3-4B-Thinking-2507, trained on software architecture content using logits distillation from GPT-4o-mini.

## Model Details

- **Base Model**: Qwen/Qwen3-4B-Thinking-2507
- **Teacher Model**: GPT-4o-mini (via OpenRouter)
- **Training Method**: Logits distillation with KL divergence
- **Dataset**: Software Architecture examples (3,000 samples)
- **Training Stats**:
  - 3 epochs
  - 2,986 KL divergence batches (99.5%)
  - 14 cross-entropy fallback batches (0.5%)
  - Average loss: 9.35

## Training Process

The model was trained using knowledge distillation where:
1. GPT-4o-mini generates responses with logprobs for software architecture questions
2. Student model learns to match the teacher's probability distributions using KL divergence
3. Fallback to cross-entropy loss when logprobs unavailable

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

- Learning rate: 1e-6
- Max tokens: 64
- Batch size: 1
- Optimizer: Adam with weight decay (1e-5)
- Precision: bfloat16

## Intended Use

This model is optimized for software architecture discussions, explanations, and Q&A. It should perform better than the base model on architecture-related topics while maintaining general capabilities.

## Limitations

- Trained on a specific subset of software architecture data
- May have biases from the teacher model (GPT-4o-mini)
- Performance on non-architecture topics may vary

## Citation

If you use this model, please cite:

```
@misc{{{MODEL_NAME.replace('-', '_')},
  title={{{MODEL_NAME}: Knowledge Distilled Qwen3-4B for Software Architecture}},
  author={{Your Name}},
  year={{2025}},
  url={{https://huggingface.co/{REPO_ID}}}
}}
```
"""
    
    # Save README
    readme_path = os.path.join(LOCAL_MODEL_PATH, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("✓ README.md created")
    
    # Step 3: Upload the entire folder
    try:
        api.upload_folder(
            folder_path=LOCAL_MODEL_PATH,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Initial upload of distilled Qwen3-4B model"
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
        model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH)
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
    print("🚀 Preparing to upload distilled model to Hugging Face...")
    
    # Check if logged in
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Logged in as: {user_info['name']}")
    except Exception:
        print("❌ Not logged in to Hugging Face")
        print("Please run: huggingface-cli login")
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
