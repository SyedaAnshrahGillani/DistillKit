"""
MAXIMUM GPU UTILIZATION VERSION - Optimized for 80GB NVIDIA GPU (A100/H100)
"""

import time
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.backends.cudnn as cudnn
import os

# ========== HIGH-PERFORMANCE CONFIG FOR 80GB GPU ==========
MODEL_ID = "Anshrah/qwen3-4b-software-arch-distilled_02"  # Your distilled model
HISTORY_TURNS = 4
REQUESTED_MAX_NEW_TOKENS = 4096  # Increased for your powerful GPU
SAFETY_MARGIN_TOKENS = 32
TEMPERATURE = 0.8
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# GPU optimization flags
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for massive speedup on A100/H100
torch.backends.cudnn.allow_tf32 = True
cudnn.benchmark = True  # Optimize cudnn for consistent input sizes
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # Better memory management
# ============================

SYSTEM_PROMPT = """You are an expert software architecture assistant. You have been fine-tuned on software architecture knowledge through distillation from GPT-4o-mini. 

Provide detailed, practical guidance on:
- System design patterns and principles
- Scalability and performance considerations  
- Cloud architecture best practices
- Microservices, APIs, and distributed systems
- Technology trade-offs and recommendations

Format your responses with clear structure and actionable advice."""

# === MAXIMUM GPU UTILIZATION MODEL LOADING ===
def load_model_and_tokenizer():
    print("Loading model with MAXIMUM 80GB GPU utilization...")
    
    # Verify GPU setup
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! This setup requires GPU.")
    
    device_capability = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"🚀 GPU: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    print(f"⚡ Compute Capability: {device_capability}")
    
    # Force optimal settings for A100/H100
    if gpu_memory > 70:  # 80GB GPU detected
        dtype = torch.bfloat16  # Best for A100/H100
        print("🎯 Using bfloat16 for maximum performance on 80GB GPU")
    else:
        dtype = torch.float16
        print("Using float16")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # MAXIMUM PERFORMANCE LOADING
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",  # Auto-distribute across available GPUs
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=True,
            # Force Flash Attention 2 for maximum speed
            attn_implementation="flash_attention_2",
            # Additional optimizations for large GPU
            use_safetensors=True,
            variant="bf16" if dtype == torch.bfloat16 else "fp16"
        )
        
        # AGGRESSIVE COMPILATION for maximum speed
        print("🔥 Compiling model with aggressive optimizations...")
        try:
            # Use max-autotune mode for best performance on powerful hardware
            model = torch.compile(
                model, 
                mode="max-autotune",  # Most aggressive optimization
                fullgraph=True,       # Compile entire model as one graph
                dynamic=False         # Static shapes for better optimization
            )
            print("✅ Model compiled with max-autotune mode!")
        except Exception as e:
            print(f"Max-autotune failed, trying reduce-overhead: {e}")
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("✅ Model compiled with reduce-overhead mode!")
            except Exception as e2:
                print(f"Torch compile failed: {e2}")
        
        model.eval()
        
        # AGGRESSIVE WARMUP for CUDA kernel optimization
        print("🔥 Performing aggressive GPU warmup...")
        warmup_texts = [
            "Hello world",
            "This is a longer warmup text to optimize CUDA kernels for various sequence lengths.",
            "An even longer warmup sequence to ensure all CUDA kernels are properly optimized and cached for maximum inference performance during actual usage."
        ]
        
        for i, text in enumerate(warmup_texts):
            dummy_input = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                _ = model.generate(
                    **dummy_input, 
                    max_new_tokens=50, 
                    do_sample=False,
                    use_cache=True
                )
            print(f"   Warmup {i+1}/3 complete")
        
        # Memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("✅ MAXIMUM PERFORMANCE MODEL LOADED!")
        print(f"📊 GPU Memory Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"📊 GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        return model, tokenizer, True
        
    except Exception as e:
        print(f"❌ Failed to load full fine-tuned model: {e}")
        raise e  # Don't fallback for 80GB GPU setup

# Load model
model, tokenizer, MODEL_LOADED = load_model_and_tokenizer()
MODEL_MAX_TOKENS = getattr(tokenizer, "model_max_length", 32768) or 32768  # Use full context

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === ULTRA-FAST UTILITIES ===
@torch.jit.script  # JIT compile for speed
def fast_count_tokens(input_ids: torch.Tensor) -> int:
    return input_ids.shape[-1]

@torch.no_grad()
def count_tokens(text: str) -> int:
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    return fast_count_tokens(input_ids)

# === MAXIMUM PERFORMANCE CHAT BACKEND ===
@torch.no_grad()
def chat(user_message: str, history, progress=gr.Progress()):
    if not user_message:
        return gr.update(), history, gr.update(value=""), gr.update(value="")

    # Pre-allocate and optimize
    torch.cuda.empty_cache()  # Clear cache before generation
    
    history = list(history or [])
    trimmed = history[-HISTORY_TURNS:] if len(history) > HISTORY_TURNS else list(history)
    
    # Build conversation for Qwen chat template
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add history (only user/assistant pairs, no thinking)
    for user_msg, assistant_msg, _ in trimmed:
        messages.append({"role": "user", "content": user_msg})
        # Clean assistant message (remove timing info)
        clean_msg = assistant_msg.split("\n\n🚀 Time taken:")[0]
        messages.append({"role": "assistant", "content": clean_msg})
    
    messages.append({"role": "user", "content": user_message})

    # Apply chat template
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_tokens = count_tokens(text_input)
    allowed_prompt_tokens = MODEL_MAX_TOKENS - REQUESTED_MAX_NEW_TOKENS - SAFETY_MARGIN_TOKENS

    # Trim if too long (optimized)
    while prompt_tokens > allowed_prompt_tokens and trimmed:
        trimmed.pop(0)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user_msg, assistant_msg, _ in trimmed:
            messages.append({"role": "user", "content": user_msg})
            clean_msg = assistant_msg.split("\n\n🚀 Time taken:")[0]
            messages.append({"role": "assistant", "content": clean_msg})
        messages.append({"role": "user", "content": user_message})
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = count_tokens(text_input)

    effective_new_tokens = min(REQUESTED_MAX_NEW_TOKENS, MODEL_MAX_TOKENS - prompt_tokens - SAFETY_MARGIN_TOKENS)
    
    # MAXIMUM PERFORMANCE TOKENIZATION
    inputs = tokenizer(
        text_input,  # Single string, not list
        return_tensors="pt",
        padding=False,  # No padding needed for single input
        truncation=True,
        max_length=MODEL_MAX_TOKENS - effective_new_tokens
    ).to(model.device, non_blocking=True)  # Async GPU transfer

    # GPU memory pre-allocation for generation
    torch.cuda.synchronize()
    start = time.time()
    
    # MAXIMUM PERFORMANCE GENERATION
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=int(effective_new_tokens),
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,  # Critical for speed
        num_beams=1,     # Sampling is faster than beam search
        # Additional optimizations for large GPU
        output_scores=False,
        return_dict_in_generate=False,
        # Memory optimizations
        low_memory=False,  # We have plenty of memory, use it!
    )
    
    torch.cuda.synchronize()  # Ensure generation is complete
    end = time.time()

    # OPTIMIZED TOKEN EXTRACTION
    output_ids = outputs[0][inputs.input_ids.shape[-1]:].tolist()

    # ULTRA-FAST THINKING EXTRACTION
    thinking_content = ""
    content = ""
    
    # Optimized thinking token detection
    THINK_TOKEN_ID = 151668  # </think> token in Qwen
    
    if THINK_TOKEN_ID in output_ids:
        try:
            # Use list.index for forward search (faster than reverse)
            think_idx = output_ids.index(THINK_TOKEN_ID)
            thinking_content = tokenizer.decode(output_ids[:think_idx], skip_special_tokens=True).strip()
            content = tokenizer.decode(output_ids[think_idx+1:], skip_special_tokens=True).strip()
        except (ValueError, IndexError):
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            thinking_content = "Thinking extraction failed"
    else:
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        thinking_content = "No thinking process detected"

    # Performance metrics
    tokens_per_second = len(output_ids) / (end - start) if (end - start) > 0 else 0
    gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
    
    # Add performance info
    model_info = "🚀 80GB GPU MAXIMIZED"
    final_reply = content + f"\n\n{model_info} | ⚡ Time: {end - start:.2f}s | 🎯 Speed: {tokens_per_second:.1f} tok/s | 💾 GPU: {gpu_memory:.1f}GB"

    # Store in history
    history.append((user_message, final_reply, thinking_content))
    if len(history) > HISTORY_TURNS:
        history = history[-HISTORY_TURNS:]

    # Build display history with collapsible thinking
    display_history = []
    for user_msg, reply, thinking in history:
        if thinking and thinking not in ["No thinking process detected", "Thinking extraction failed"]:
            formatted_reply = f"<details><summary>💭 Thinking Process</summary><pre>{thinking}</pre></details>\n\n{reply}"
        else:
            formatted_reply = reply
        display_history.append((user_msg, formatted_reply))

    # Clean up GPU memory
    torch.cuda.empty_cache()
    
    return display_history, history, gr.update(value=""), gr.update(value=thinking_content)

def clear_history():
    torch.cuda.empty_cache()  # Clean memory on clear
    return [], [], gr.update(value=""), gr.update(value="")

def test_model():
    """High-performance model test"""
    test_question = "What are the key principles of microservices architecture?"
    print(f"🧪 Testing 80GB GPU performance with: {test_question}")
    
    try:
        start = time.time()
        inputs = tokenizer(test_question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        end = time.time()
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_generated = outputs[0].shape[-1] - inputs.input_ids.shape[-1]
        speed = tokens_generated / (end - start)
        
        print(f"✅ Test successful!")
        print(f"⚡ Speed: {speed:.1f} tokens/second")
        print(f"📝 Response preview: {response[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

# Perform startup test
print("🚀 Testing maximum performance setup...")
test_model()

# === HIGH-PERFORMANCE GRADIO UI ===
with gr.Blocks(title="80GB GPU MAXIMIZED Architecture Assistant") as demo:
    model_status = "🚀 80GB GPU FULLY UTILIZED" if MODEL_LOADED else "❌ GPU Setup Failed"
    
    gr.Markdown(f"## {model_status}")
    gr.Markdown("**MAXIMUM PERFORMANCE** Software Architecture Assistant - Full fine-tuned model with 80GB GPU optimization")
    gr.Markdown(f"🎯 Model: `{MODEL_ID}`")
    gr.Markdown(f"⚡ Max context: **{MODEL_MAX_TOKENS:,}** tokens | Generation: **{REQUESTED_MAX_NEW_TOKENS:,}** tokens | History: **{HISTORY_TURNS}** turns")
    gr.Markdown(f"🔥 Optimizations: TF32, Flash Attention 2, Max-Autotune, JIT Compilation")

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="🚀 Ultra-Fast Architecture Discussion",
                height=500,
                show_label=True
            )
            
            with gr.Row():
                user_input = gr.Textbox(
                    label="Ask about software architecture (80GB GPU power ready)...", 
                    placeholder="e.g., Design a microservices system handling 1M+ requests/second",
                    lines=2,
                    scale=4
                )
                send_btn = gr.Button("🚀 SEND", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                
        with gr.Column(scale=1):
            with gr.Accordion("🧠 Model's Internal Thinking", open=False):
                reasoning_output = gr.Textbox(
                    label="Thinking Process",
                    lines=10,
                    interactive=False,
                    placeholder="High-speed thinking process will appear here..."
                )

    # High-performance example questions
    gr.Markdown("### 🎯 High-Performance Architecture Questions:")
    examples = [
        "Design a system to handle 10 million concurrent users",
        "Create a microservices architecture for real-time trading",
        "Build a globally distributed content delivery system",
        "Design fault-tolerant event-driven architecture",
        "Optimize database sharding for petabyte-scale data"
    ]
    
    with gr.Row():
        for example in examples[:3]:
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

if __name__ == "__main__":
    print("🚀 Launching MAXIMUM PERFORMANCE Gradio interface...")
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        # Additional optimizations for high-performance serving
        max_threads=40,  # Handle more concurrent requests
        show_error=True
    )
