"""
Fixed test app for your full fine-tuned model (not PEFT)
"""

import time
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== CONFIG ==========
MODEL_ID = "Anshrah/qwen3-4b-software-arch-distilled_04"  # Your distilled model
HISTORY_TURNS = 4
REQUESTED_MAX_NEW_TOKENS = 8192 #2500 previous
SAFETY_MARGIN_TOKENS = 32
TEMPERATURE = 0.8
TOP_P = 0.9
REPETITION_PENALTY = 1.1
# ============================

SYSTEM_PROMPT = """You are an expert software architecture assistant. You have been fine-tuned on software architecture knowledge through distillation from GPT-4o-mini. 

Provide detailed, practical guidance on:
- System design patterns and principles
- Scalability and performance considerations  
- Cloud architecture best practices
- Microservices, APIs, and distributed systems
- Technology trade-offs and recommendations

Format your responses with clear structure and actionable advice."""

SYSTEM_PROMPT = "You are a helpful AI assistant. Think through problems step by step before answering."

# === Load model & tokenizer once ===
def load_model_and_tokenizer():
    print("Loading your distilled model...")
    
    # Load directly as a full model (not PEFT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,  # Use bfloat16 as trained
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("✓ Model loaded successfully!")
    return model, tokenizer

# Load model
try:
    model, tokenizer = load_model_and_tokenizer()
    MODEL_MAX_TOKENS = getattr(tokenizer, "model_max_length", 4096) or 4096
    MODEL_LOADED = True
except Exception as e:
    print(f"Failed to load model: {e}")
    print("Falling back to base model...")
    
    # Fallback to base model if distilled model fails
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Thinking-2507",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    MODEL_MAX_TOKENS = getattr(tokenizer, "model_max_length", 4096) or 4096
    MODEL_LOADED = False

# === Utility ===
def count_tokens(text: str) -> int:
    return len(tokenizer(text, return_tensors="pt")["input_ids"][0])

# === Chat backend with thinking extraction ===
def chat(user_message: str, history, progress=gr.Progress()):
    if not user_message:
        return gr.update(), history, gr.update(value=""), gr.update(value="")

    history = list(history or [])
    trimmed = history[-HISTORY_TURNS:] if len(history) > HISTORY_TURNS else list(history)
    
    # Build conversation for Qwen chat template
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add history (only user/assistant pairs, no thinking)
    for user_msg, assistant_msg, _ in trimmed:
        messages.append({"role": "user", "content": user_msg})
        # Clean assistant message (remove timing info)
        clean_msg = assistant_msg.split("\n\n⏱ Time taken:")[0]
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

    # Trim if too long
    while prompt_tokens > allowed_prompt_tokens and trimmed:
        trimmed.pop(0)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user_msg, assistant_msg, _ in trimmed:
            messages.append({"role": "user", "content": user_msg})
            clean_msg = assistant_msg.split("\n\n⏱ Time taken:")[0]
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
            pad_token_id=tokenizer.eos_token_id
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
    final_reply = content + f"\n\n{model_info} | ⏱ Time: {end - start:.2f}s"

    # Store in history (user_msg, assistant_reply, thinking)
    history.append((user_message, final_reply, thinking_content))
    if len(history) > HISTORY_TURNS:
        history = history[-HISTORY_TURNS:]

    # Build display history with collapsible thinking
    display_history = []
    for user_msg, reply, thinking in history:
        if thinking and thinking != "No thinking process detected":
            formatted_reply = f"<details><summary>💭 Thinking Process</summary><pre>{thinking}</pre></details>\n\n{reply}"
        else:
            formatted_reply = reply
        display_history.append((user_msg, formatted_reply))

    return display_history, history, gr.update(value=""), gr.update(value=thinking_content)

def clear_history():
    return [], [], gr.update(value=""), gr.update(value="")

def test_model():
    """Test the model with a simple architecture question"""
    test_question = "What are the key principles of microservices architecture?"
    print(f"Testing model with: {test_question}")
    
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
        print(f"✓ Model test successful: {response[:200]}...")
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

# Test model on startup
print("Testing model...")
test_model()

# === Gradio UI ===
with gr.Blocks(title="Software Architecture Assistant") as demo:
    model_status = "🧠 Distilled Model Loaded" if MODEL_LOADED else "📚 Base Model (Fallback)"
    
    gr.Markdown(f"## {model_status}")
    gr.Markdown("**Software Architecture Assistant** - Fine-tuned on architecture knowledge through distillation")
    gr.Markdown(f"Model: `{MODEL_ID}` | Max tokens: **{MODEL_MAX_TOKENS}** | History: **{HISTORY_TURNS}** turns")

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Architecture Discussion",
                height=500,
                show_label=True
            )
            
            with gr.Row():
                user_input = gr.Textbox(
                    label="Ask about software architecture...", 
                    placeholder="e.g., How do I design a scalable microservices system?",
                    lines=2,
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Clear History", variant="secondary")
                
        with gr.Column(scale=1):
            with gr.Accordion("Model's Internal Thinking", open=False):
                reasoning_output = gr.Textbox(
                    label="Thinking Process",
                    lines=10,
                    interactive=False,
                    placeholder="Thinking process will appear here..."
                )

    # Example questions
    gr.Markdown("### Example Questions:")
    examples = [
        "What are the key principles of microservices architecture?",
        "How do I design a scalable API gateway?",
        "What's the difference between horizontal and vertical scaling?",
        "Explain the CAP theorem and its implications for distributed systems",
        "What are the trade-offs between SQL and NoSQL databases?"
    ]
    
    with gr.Row():
        for i, example in enumerate(examples[:3]):
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
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )
