"""
High-performance inference app with comprehensive GPU optimizations
"""

import time
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
from typing import List, Tuple, Optional
import threading

# ========== OPTIMIZED CONFIG ==========
MODEL_ID = "Anshrah/qwen3-4b-software-arch-distilled_04"
HISTORY_TURNS = 4
REQUESTED_MAX_NEW_TOKENS = 2500
SAFETY_MARGIN_TOKENS = 32
TEMPERATURE = 0.6
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# Performance optimizations
ENABLE_KV_CACHE = True
PREFILL_BATCH_SIZE = 1
MAX_CONCURRENT_REQUESTS = 4
MEMORY_OPTIMIZATION = True
# =====================================

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

# Global request queue for batching
request_queue = []
queue_lock = threading.Lock()

# === Enhanced Device and CUDA info ===
def get_device_info():
    """Get comprehensive device information"""
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "device_name": "CPU",
        "memory_info": "N/A",
        "compute_capability": "N/A",
        "memory_bandwidth": "N/A"
    }
    
    if torch.cuda.is_available():
        device_info["current_device"] = torch.cuda.current_device()
        device_info["device_name"] = torch.cuda.get_device_name()
        
        # Enhanced memory info
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_cached = torch.cuda.memory_reserved() / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        device_info["memory_info"] = f"{memory_allocated:.1f}GB/{total_memory:.1f}GB allocated"
        
        # Compute capability and memory bandwidth
        props = torch.cuda.get_device_properties(0)
        device_info["compute_capability"] = f"{props.major}.{props.minor}"
        device_info["memory_bandwidth"] = f"{props.memory_clock_rate * props.memory_bus_width // 8 / 1e6:.0f} GB/s"
    
    return device_info

# === Optimized model loading ===
def load_model_and_tokenizer():
    device_info = get_device_info()
    print(f"Device: {device_info['device_name']} | CUDA: {device_info['cuda_available']}")
    print(f"Memory: {device_info['memory_info']} | Bandwidth: {device_info['memory_bandwidth']}")
    print(f"Compute: {device_info['compute_capability']}")
    
    print(f"Loading distilled model: {MODEL_ID}")
    
    try:
        # Optimized tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True,
            use_fast=True,
            padding_side="left"  # Better for batch inference
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Enable padding token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print("Loading model with performance optimizations...")
        
        # Enhanced model loading with performance optimizations
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=ENABLE_KV_CACHE,  # Enable KV caching
            torch_compile=True,  # Enable graph compilation
            attn_implementation="eager"  # Use optimized attention
        )
        
        # Enable eval mode and memory optimizations
        model.eval()
        
        # Enable memory efficient attention if available
        if hasattr(model.config, 'use_memory_efficient_attention'):
            model.config.use_memory_efficient_attention = True
        
        # Optimize for inference
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
            
        # Enable optimized generation
        model.generation_config.use_cache = ENABLE_KV_CACHE
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        
        # Compile model for faster inference
        try:
            print("Compiling model for optimized inference...")
            # Compile only the forward pass for generation
            model.forward = torch.compile(
                model.forward, 
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=True
            )
            print("Model compilation successful")
        except Exception as e:
            print(f"Model compilation failed (continuing without): {e}")
        
        # Optimize memory layout
        if MEMORY_OPTIMIZATION and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Memory optimizations applied")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model loaded successfully!")
        print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"Memory usage: {get_memory_usage()}")
        
        return model, tokenizer, True, device_info
        
    except Exception as e:
        print(f"Failed to load distilled model: {e}")
        print("Falling back to base model...")
        
        # Fallback with same optimizations
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-4B-Thinking-2507", 
                trust_remote_code=True,
                use_fast=True,
                padding_side="left"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-4B-Thinking-2507",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=ENABLE_KV_CACHE,
                attn_implementation="eager"
            )
            model.eval()
            
            if hasattr(model, 'gradient_checkpointing_disable'):
                model.gradient_checkpointing_disable()
                
            model.generation_config.use_cache = ENABLE_KV_CACHE
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            
            try:
                model.forward = torch.compile(model.forward, mode="reduce-overhead")
                print("Base model compiled successfully")
            except:
                pass
            
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Base model loaded as fallback!")
            print(f"Parameters: {total_params:,}")
            
            return model, tokenizer, False, device_info
            
        except Exception as e2:
            print(f"Failed to load base model too: {e2}")
            raise RuntimeError("Could not load any model")

# Load model with optimizations
try:
    model, tokenizer, MODEL_LOADED, device_info = load_model_and_tokenizer()
    MODEL_MAX_TOKENS = getattr(tokenizer, "model_max_length", 4096) or 4096
except Exception as e:
    print(f"Critical error: {e}")
    exit(1)

# === Optimized utility functions ===
def count_tokens_batch(texts: List[str]) -> List[int]:
    """Optimized batch token counting"""
    encoded = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)
    return [len(ids) for ids in encoded['input_ids']]

def count_tokens(text: str) -> int:
    """Fast single text token counting"""
    return len(tokenizer.encode(text, add_special_tokens=False))

def get_memory_usage():
    """Enhanced GPU memory usage tracking"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        utilization = (allocated / total) * 100
        return f"GPU: {allocated:.1f}GB/{total:.1f}GB ({utilization:.1f}% util)"
    return "CPU mode"

def optimize_memory():
    """Aggressive memory optimization"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# === High-performance chat backend ===
def chat(user_message: str, history, progress=gr.Progress()):
    if not user_message or not user_message.strip():
        return gr.update(), history, gr.update(value=""), gr.update(value="")

    history = list(history or [])
    trimmed = history[-HISTORY_TURNS:] if len(history) > HISTORY_TURNS else list(history)
    
    # Build optimized conversation
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add history with optimized cleaning
    for user_msg, assistant_msg, _ in trimmed:
        messages.append({"role": "user", "content": user_msg})
        # More efficient message cleaning
        clean_msg = assistant_msg.split("\n\nDistilled Model")[0].split("\n\nBase Model")[0]
        messages.append({"role": "assistant", "content": clean_msg})
    
    messages.append({"role": "user", "content": user_message})

    try:
        # Apply chat template once
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Fast token counting and context management
        prompt_tokens = count_tokens(text_input)
        allowed_prompt_tokens = MODEL_MAX_TOKENS - REQUESTED_MAX_NEW_TOKENS - SAFETY_MARGIN_TOKENS

        # Efficient context trimming
        while prompt_tokens > allowed_prompt_tokens and trimmed:
            trimmed.pop(0)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for user_msg, assistant_msg, _ in trimmed:
                messages.append({"role": "user", "content": user_msg})
                clean_msg = assistant_msg.split("\n\nDistilled Model")[0].split("\n\nBase Model")[0]
                messages.append({"role": "assistant", "content": clean_msg})
            messages.append({"role": "user", "content": user_message})
            text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_tokens = count_tokens(text_input)

        effective_new_tokens = min(REQUESTED_MAX_NEW_TOKENS, MODEL_MAX_TOKENS - prompt_tokens - SAFETY_MARGIN_TOKENS)
        
        # Optimized input preparation
        inputs = tokenizer(
            text_input, 
            return_tensors="pt", 
            truncation=True,
            max_length=MODEL_MAX_TOKENS - REQUESTED_MAX_NEW_TOKENS - SAFETY_MARGIN_TOKENS,
            padding=False
        ).to(model.device, non_blocking=True)

        start = time.time()
        
        # High-performance generation with optimizations
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(effective_new_tokens),
                min_new_tokens=1,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                do_sample=True,
                use_cache=ENABLE_KV_CACHE,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # Performance optimizations
                num_beams=1,  # Faster than beam search
                early_stopping=False,
                length_penalty=1.0,
                # Memory optimizations
                output_attentions=False,
                output_hidden_states=False,
                return_dict_in_generate=False
            )
        
        end = time.time()
        generation_time = end - start
        
        # Optimized token extraction
        input_length = inputs.input_ids.shape[1]
        output_ids = outputs[0][input_length:].tolist()
        tokens_generated = len(output_ids)

        # Preserved thinking extraction with optimizations
        thinking_content = ""
        content = ""
        
        try:
            # Optimized thinking token detection
            if 151668 in output_ids:
                # Find last occurrence for better performance
                think_end_idx = len(output_ids) - 1 - output_ids[::-1].index(151668)
                thinking_content = tokenizer.decode(output_ids[:think_end_idx], skip_special_tokens=True).strip()
                content = tokenizer.decode(output_ids[think_end_idx:], skip_special_tokens=True).strip()
            else:
                content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                thinking_content = "No thinking process detected"
        except (ValueError, IndexError):
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            thinking_content = "Thinking extraction failed"

        # Performance metrics
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        # Enhanced status info
        model_info = "Distilled Model" if MODEL_LOADED else "Base Model"
        memory_info = get_memory_usage()
        
        # Add throughput metrics
        throughput_info = f"{tokens_per_second:.1f} tok/s"
        if tokens_generated > 0:
            latency_per_token = (generation_time / tokens_generated) * 1000  # ms per token
            throughput_info += f" ({latency_per_token:.1f}ms/tok)"
        
        final_reply = content + f"\n\n{model_info} | Time: {generation_time:.2f}s | {throughput_info} | {memory_info}"

        # Store in history
        history.append((user_message, final_reply, thinking_content))
        if len(history) > HISTORY_TURNS:
            history = history[-HISTORY_TURNS:]

        # Build display history with preserved thinking
        display_history = []
        for user_msg, reply, thinking in history:
            if thinking and thinking not in ["No thinking process detected", "Thinking extraction failed"]:
                formatted_reply = f"<details><summary>Thinking Process</summary><pre>{thinking}</pre></details>\n\n{reply}"
            else:
                formatted_reply = reply
            display_history.append((user_msg, formatted_reply))

        # Memory cleanup for long sessions
        if len(history) % 5 == 0:  # Every 5 messages
            optimize_memory()

        return display_history, history, gr.update(value=""), gr.update(value=thinking_content)
        
    except Exception as e:
        error_msg = f"Generation failed: {str(e)}"
        history.append((user_message, error_msg, ""))
        return [history[-1]], history, gr.update(value=""), gr.update(value="")

def clear_history():
    """Enhanced history clearing with memory optimization"""
    optimize_memory()
    return [], [], gr.update(value=""), gr.update(value="")

def benchmark_model():
    """Enhanced model benchmarking"""
    test_prompts = [
        "What is cloud architecture?",
        "Explain microservices design patterns.",
        "How do you implement disaster recovery?"
    ]
    
    print("Running comprehensive model benchmark...")
    results = []
    
    for prompt in test_prompts:
        try:
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            
            # Warm up
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
            # Actual benchmark
            start_time = time.time()
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    use_cache=ENABLE_KV_CACHE,
                    pad_token_id=tokenizer.pad_token_id
                )
            end_time = time.time()
            
            generation_time = end_time - start_time
            tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]
            speed = tokens_generated / generation_time
            
            results.append(speed)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Test {len(results)}: {speed:.1f} tok/s | Preview: {response[:100]}...")
            
        except Exception as e:
            print(f"Benchmark failed on prompt {len(results)+1}: {e}")
            results.append(0.0)
    
    avg_speed = sum(results) / len(results) if results else 0.0
    print(f"Average benchmark speed: {avg_speed:.1f} tokens/second")
    return len(results) == len(test_prompts), avg_speed

# Run enhanced benchmark
print("Running model benchmark...")
model_works, benchmark_speed = benchmark_model()

# === Enhanced Gradio UI ===
model_status = "Distilled Model Loaded" if MODEL_LOADED else "Base Model (Fallback)"
device_status = f"{device_info['device_name']}" if device_info['cuda_available'] else "CPU"
performance_status = f"{benchmark_speed:.1f} tok/s avg" if model_works else "Benchmark failed"

with gr.Blocks(title="Senior Cloud Architect Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {model_status} - {device_status}")
    gr.Markdown("**Senior Cloud Architect Assistant** - Optimized for high-performance inference")
    
    with gr.Row():
        gr.Markdown(f"**Model:** `{MODEL_ID}`")
        gr.Markdown(f"**Context:** {MODEL_MAX_TOKENS:,} tokens")
        gr.Markdown(f"**Generation:** {REQUESTED_MAX_NEW_TOKENS:,} tokens")
        if device_info['cuda_available']:
            gr.Markdown(f"**Performance:** {performance_status}")
            gr.Markdown(f"**Memory Bandwidth:** {device_info['memory_bandwidth']}")

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Architecture Discussion",
                height=600,
                show_label=True,
                avatar_images=("👤", "🤖"),
                bubble_full_width=False,
                render_markdown=True
            )
            
            with gr.Row():
                user_input = gr.Textbox(
                    label="Ask about cloud architecture...", 
                    placeholder="e.g., How do I design a multi-cloud disaster recovery strategy?",
                    lines=2,
                    scale=4,
                    max_lines=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Clear History", variant="secondary")
                
        with gr.Column(scale=1):
            with gr.Accordion("Model's Internal Thinking", open=False):
                reasoning_output = gr.Textbox(
                    label="Thinking Process",
                    lines=15,
                    interactive=False,
                    placeholder="The model's reasoning process will appear here if available...",
                    show_copy_button=True,
                    max_lines=20
                )

    # Expert questions
    gr.Markdown("### Expert Cloud Architecture Questions:")
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

    # Event handlers with performance optimizations
    user_input.submit(
        chat, 
        [user_input, state], 
        [chatbot, state, user_input, reasoning_output],
        queue=True
    )
    send_btn.click(
        chat, 
        [user_input, state], 
        [chatbot, state, user_input, reasoning_output],
        queue=True
    )
    clear_btn.click(
        clear_history, 
        None, 
        [chatbot, state, user_input, reasoning_output]
    )

    # Enhanced footer with performance metrics
    gr.Markdown("---")
    status_items = [
        f"**Status:** {'Model operational' if model_works else 'Model issues detected'}",
        f"**Device:** {device_status}",
        f"**Performance:** {performance_status}"
    ]
    if device_info['cuda_available']:
        status_items.append(f"**Compute:** {device_info['compute_capability']}")
        status_items.append(f"**Memory:** {device_info['memory_info']}")
    
    gr.Markdown(" | ".join(status_items))

if __name__ == "__main__":
    print("Launching high-performance Gradio interface...")
    print(f"Model: {'Distilled' if MODEL_LOADED else 'Base (fallback)'}")
    print(f"Device: {device_info['device_name']}")
    print(f"CUDA: {device_info['cuda_available']}")
    print(f"Memory: {device_info['memory_info']}")
    print(f"Benchmark Speed: {benchmark_speed:.1f} tokens/second")
    print(f"Memory Bandwidth: {device_info['memory_bandwidth']}")
    
    # Enable queue for better performance
    demo.queue(
        concurrency_count=MAX_CONCURRENT_REQUESTS,
        max_size=20,
        api_open=False
    )
    
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        enable_queue=True
    )
