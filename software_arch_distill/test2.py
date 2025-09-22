"""
Improved test app for your distilled model with CUDA info and better error handling
"""

import time
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== CONFIG ==========
MODEL_ID = "Anshrah/qwen3-4b-software-arch-distilled_03"  # Your distilled model
HISTORY_TURNS = 4
REQUESTED_MAX_NEW_TOKENS = 8192
SAFETY_MARGIN_TOKENS = 32
TEMPERATURE = 0.8
TOP_P = 0.9
REPETITION_PENALTY = 1.1
# ============================

# Senior Cloud Architect System Prompt
SYSTEM_PROMPT = """You are a senior cloud architect with 15+ years of experience.
You provide deeply knowledgeable, practical, and cloud-agnostic answers.

Guidelines:
- Always explain the reasoning clearly, not just the final solution.
- Mention relevant services across multiple cloud providers (AWS, Azure, GCP, Oracle Cloud).
- Always include at least one open-source or cloud-agnostic alternative (e.g., MinIO for object storage, Ceph, PostgreSQL, Kafka, Kubernetes).
- When relevant, illustrate architectures with a simple Mermaid or ASCII diagram.
- Include trade-offs: managed vs. open source, vendor lock-in vs. portability, cost vs. scalability.
- Highlight best practices and common pitfalls.
- Use natural expert language; structure the answer logically but adapt to the question. 
  (You don't always need fixed headings like "Concept / Implementation / Pitfalls" — use them only when it improves clarity.)
- Assume the reader is a mid/senior engineer preparing for real-world projects or certifications."""

# === Device and CUDA info ===
def get_device_info():
    """Get comprehensive device information"""
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "device_name": "CPU",
        "memory_info": "N/A"
    }
    
    if torch.cuda.is_available():
        device_info["current_device"] = torch.cuda.current_device()
        device_info["device_name"] = torch.cuda.get_device_name()
        
        # Get memory info
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        device_info["memory_info"] = f"{memory_allocated:.1f}GB/{total_memory:.1f}GB allocated"
    
    return device_info

# === Load model & tokenizer once ===
def load_model_and_tokenizer():
    device_info = get_device_info()
    print(f"🖥️  Device Info: {device_info['device_name']} | CUDA: {device_info['cuda_available']}")
    
    print(f"Loading distilled model: {MODEL_ID}")
    
    try:
        # Load directly as a full model (not PEFT)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,  # Use bfloat16 as trained
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Distilled model loaded successfully!")
        print(f"   📊 Parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"   🖥️  Device: {model.device if hasattr(model, 'device') else 'Multiple devices'}")
        
        return model, tokenizer, True, device_info
        
    except Exception as e:
        print(f"❌ Failed to load distilled model: {e}")
        print("🔄 Falling back to base model...")
        
        # Fallback to base model
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507", trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-4B-Thinking-2507",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model.eval()
            
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✅ Base model loaded as fallback!")
            print(f"   📊 Parameters: {total_params:,}")
            
            return model, tokenizer, False, device_info
            
        except Exception as e2:
            print(f"❌ Failed to load base model too: {e2}")
            raise RuntimeError("Could not load any model")

# Load model
try:
    model, tokenizer, MODEL_LOADED, device_info = load_model_and_tokenizer()
    MODEL_MAX_TOKENS = getattr(tokenizer, "model_max_length", 4096) or 4096
except Exception as e:
    print(f"❌ Critical error: {e}")
    exit(1)

# === Utility ===
def count_tokens(text: str) -> int:
    return len(tokenizer(text, return_tensors="pt", truncation=True, max_length=MODEL_MAX_TOKENS)["input_ids"][0])

def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved"
    return "CPU mode"

# === Chat backend with thinking extraction ===
def chat(user_message: str, history, progress=gr.Progress()):
    if not user_message or not user_message.strip():
        return gr.update(), history, gr.update(value=""), gr.update(value="")

    history = list(history or [])
    trimmed = history[-HISTORY_TURNS:] if len(history) > HISTORY_TURNS else list(history)
    
    # Build conversation for Qwen chat template
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add history (only user/assistant pairs, no thinking)
    for user_msg, assistant_msg, _ in trimmed:
        messages.append({"role": "user", "content": user_msg})
        # Clean assistant message (remove timing info)
        clean_msg = assistant_msg.split("\n\n🧠")[0].split("\n\n📚")[0]
        messages.append({"role": "assistant", "content": clean_msg})
    
    messages.append({"role": "user", "content": user_message})

    try:
        # Apply chat template
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompt_tokens = count_tokens(text_input)
        allowed_prompt_tokens = MODEL_MAX_TOKENS - REQUESTED_MAX_NEW_TOKENS - SAFETY_MARGIN_TOKENS

        # Trim if too long
        while prompt_tokens > allowed_prompt_tokens and trimmed:
            trimmed.pop(0)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for user_msg, assistant_msg, _ in trimmed:
                messages.append({"role": "user", "content": user_msg})
                clean_msg = assistant_msg.split("\n\n🧠")[0].split("\n\n📚")[0]
                messages.append({"role": "assistant", "content": clean_msg})
            messages.append({"role": "user", "content": user_message})
            text_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_tokens = count_tokens(text_input)

        effective_new_tokens = min(REQUESTED_MAX_NEW_TOKENS, MODEL_MAX_TOKENS - prompt_tokens - SAFETY_MARGIN_TOKENS)
        inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(effective_new_tokens),
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        end = time.time()

        output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()

        # Extract thinking if model supports it (Qwen thinking models)
        thinking_content = ""
        content = ""
        
        try:
            # Try to find thinking token (</think> token id is 151668 in Qwen)
            if 151668 in output_ids:
                index = len(output_ids) - output_ids[::-1].index(151668)
                thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
                content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
            else:
                # No thinking tokens found, treat as regular response
                content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                thinking_content = "No thinking process detected"
        except (ValueError, IndexError):
            # Fallback if thinking extraction fails
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            thinking_content = "Thinking extraction failed"

        # Add timing and model info
        model_info = "🧠 Distilled Model" if MODEL_LOADED else "📚 Base Model"
        memory_info = get_memory_usage()
        final_reply = content + f"\n\n{model_info} | ⏱ Time: {end - start:.2f}s | {memory_info}"

        # Store in history (user_msg, assistant_reply, thinking)
        history.append((user_message, final_reply, thinking_content))
        if len(history) > HISTORY_TURNS:
            history = history[-HISTORY_TURNS:]

        # Build display history with collapsible thinking
        display_history = []
        for user_msg, reply, thinking in history:
            if thinking and thinking != "No thinking process detected" and thinking != "Thinking extraction failed":
                formatted_reply = f"<details><summary>💭 Thinking Process</summary><pre>{thinking}</pre></details>\n\n{reply}"
            else:
                formatted_reply = reply
            display_history.append((user_msg, formatted_reply))

        return display_history, history, gr.update(value=""), gr.update(value=thinking_content)
        
    except Exception as e:
        error_msg = f"❌ Generation failed: {str(e)}"
        history.append((user_message, error_msg, ""))
        return [history[-1]], history, gr.update(value=""), gr.update(value="")

def clear_history():
    """Clear chat history and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [], [], gr.update(value=""), gr.update(value="")

def test_model():
    """Test the model with a cloud architecture question"""
    test_question = "What are the key considerations for designing a multi-region cloud architecture?"
    print(f"🧪 Testing model with: {test_question}")
    
    try:
        inputs = tokenizer([test_question], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Model test successful: {response[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

# Test model on startup
print("🧪 Testing model...")
model_works = test_model()

# === Gradio UI ===
model_status = "🧠 Distilled Model Loaded" if MODEL_LOADED else "📚 Base Model (Fallback)"
device_status = f"🖥️ {device_info['device_name']}" if device_info['cuda_available'] else "💻 CPU"

with gr.Blocks(title="Senior Cloud Architect Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {model_status} {device_status}")
    gr.Markdown("**Senior Cloud Architect Assistant** - Fine-tuned through ULD distillation with 15+ years of expertise")
    
    with gr.Row():
        gr.Markdown(f"**Model:** `{MODEL_ID}`")
        gr.Markdown(f"**Max Tokens:** {MODEL_MAX_TOKENS:,}")
        gr.Markdown(f"**History:** {HISTORY_TURNS} turns")
        if device_info['cuda_available']:
            gr.Markdown(f"**GPU Memory:** {device_info['memory_info']}")

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Architecture Discussion",
                height=600,
                show_label=True,
                avatar_images=("👤", "🤖"),
                bubble_full_width=False
            )
            
            with gr.Row():
                user_input = gr.Textbox(
                    label="Ask about cloud architecture...", 
                    placeholder="e.g., How do I design a multi-cloud disaster recovery strategy?",
                    lines=2,
                    scale=4
                )
                send_btn = gr.Button("Send 🚀", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Clear History 🗑️", variant="secondary")
                
        with gr.Column(scale=1):
            with gr.Accordion("🧠 Model's Internal Thinking", open=False):
                reasoning_output = gr.Textbox(
                    label="Thinking Process",
                    lines=15,
                    interactive=False,
                    placeholder="The model's reasoning process will appear here if available...",
                    show_copy_button=True
                )

    # Example questions for a senior cloud architect
    gr.Markdown("### 💡 Expert Cloud Architecture Questions:")
    examples = [
        "Design a multi-cloud disaster recovery strategy with RTO < 5 minutes",
        "Compare serverless vs containerized approaches for event-driven architecture",
        "What's the best way to implement zero-downtime database migrations at scale?",
        "How do I design a cost-optimized data lake architecture across AWS, Azure, and GCP?",
        "Explain the trade-offs between service mesh solutions (Istio, Linkerd, Consul Connect)"
    ]
    
    with gr.Row():
        for example in examples[:3]:
            gr.Button(example, size="sm").click(
                lambda x=example: x, 
                outputs=user_input
            )
    
    with gr.Row():
        for example in examples[3:]:
            gr.Button(example, size="sm").click(
                lambda x=example: x, 
                outputs=user_input
            )

    state = gr.State([])

    # Event handlers
    user_input.submit(
        chat, 
        [user_input, state], 
        [chatbot, state, user_input, reasoning_output]
    )
    send_btn.click(
        chat, 
        [user_input, state], 
        [chatbot, state, user_input, reasoning_output]
    )
    clear_btn.click(
        clear_history, 
        None, 
        [chatbot, state, user_input, reasoning_output]
    )

    # Footer
    gr.Markdown("---")
    gr.Markdown(f"**Status:** {'✅ Model working' if model_works else '❌ Model issues detected'} | **Device:** {device_status}")

if __name__ == "__main__":
    print(f"🚀 Launching Gradio interface...")
    print(f"   Model: {'Distilled' if MODEL_LOADED else 'Base (fallback)'}")
    print(f"   Device: {device_info['device_name']}")
    print(f"   CUDA: {device_info['cuda_available']}")
    
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
