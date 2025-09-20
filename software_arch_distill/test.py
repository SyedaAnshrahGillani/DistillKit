##### supposed to render thinking..

import time
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ========== CONFIG ==========
MODEL_ID = "Anshrah/qwen3-4b-software-arch-distilled"
BASE_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
HISTORY_TURNS = 4
REQUESTED_MAX_NEW_TOKENS = 8192 #2500 previous
SAFETY_MARGIN_TOKENS = 32
TEMPERATURE = 0.8
TOP_P = 0.9
REPETITION_PENALTY = 1.1
# ============================

SYSTEM_PROMPT = """
You are an expert, mentor-style assistant tuned to help users become senior, cloud-agnostic architects and advanced cloud practitioners. 
Your job: explain concepts deeply, make high-quality architecture recommendations, and produce creative, actionable, vendor-neutral guidance. 
Response rules:
- Always lead with a one-line scope header: "Scope: Cloud Architecture | Tone: Mentor | Format: Sections, Tables, Examples".
- Use clear sectioning: Context → Recommendation(s) → Trade-offs → Implementation sketch → Example(s) → Next steps.
- Present 3 labelled options: Recommended ✅, Alternatives ⚠️, When to avoid ⛔.
- Use Markdown for structure, tables, emojis sparingly ✅⚠️⛔.
- Aim for long-form answers while keeping paragraphs readable (≤6 lines).
- Respect safety & privacy: never invent private keys or secrets.
"""

SYSTEM_PROMPT = "You are a helpful AI assistant. Think through problems step by step before answering."

# === Load model & tokenizer once ===
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, MODEL_ID)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()
MODEL_MAX_TOKENS = getattr(tokenizer, "model_max_length", 4096) or 4096

# === Utility ===
def count_tokens(text: str) -> int:
    return len(tokenizer(text, return_tensors="pt")["input_ids"][0])

# === Chat backend with thinking extraction and collapsible display ===
def chat(user_message: str, history, progress=gr.Progress()):
    if not user_message:
        return gr.update(), history, gr.update(value="")

    history = list(history or [])
    trimmed = history[-HISTORY_TURNS:] if len(history) > HISTORY_TURNS else list(history)
    base_prompt = SYSTEM_PROMPT.strip() + "\n\n"

    # Build conversation for Qwen chat template
    messages = [{"role": "system", "content": base_prompt}]
    for u, a, _ in trimmed:  # unpack thinking as _
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_message})

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
        # Rebuild messages
        messages = [{"role": "system", "content": base_prompt}]
        for u, a in trimmed:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
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
        )
    end = time.time()

    output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()

    # Extract thinking (</think> token id is 151668 in Qwen)
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    final_reply = content + f"\n\n⏱ Time taken: {end - start:.2f} s"

    # Store thinking along with the response
    history.append((user_message, final_reply, thinking_content))
    if len(history) > HISTORY_TURNS:
        history = history[-HISTORY_TURNS:]

    # Build display history with collapsible thinking at top
    display_history = []
    for user_msg, reply, thinking in history:
        formatted_reply = f"<details><summary>💭 Thinking</summary>{thinking}</details>\n\n{reply}"
        display_history.append((user_msg, formatted_reply))

    return display_history, history, gr.update(value=""), gr.update(value=thinking_content)

def clear_history():
    return [], [], gr.update(value=""), gr.update(value="")

# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## ☁️ Cloud Mentor Chatbot (Thinking-enabled)")
    gr.Markdown(f"- History turns: **{HISTORY_TURNS}** | Requested reply tokens: **{REQUESTED_MAX_NEW_TOKENS}** | Model max tokens: **{MODEL_MAX_TOKENS}**")

    chatbot = gr.Chatbot(label="Conversation")
    user_input = gr.Textbox(label="Your question", placeholder="Ask about cloud architecture...", lines=1)
    send_btn = gr.Button("Send")
    clear_btn = gr.Button("Clear")

    reasoning_section = gr.Accordion("Model's Reasoning (Internal Thoughts)", open=False)
    reasoning_output = gr.Textbox(label="Thinking Output", interactive=False)

    state = gr.State([])

    user_input.submit(chat, [user_input, state], [chatbot, state, user_input, reasoning_output])
    send_btn.click(chat, [user_input, state], [chatbot, state, user_input, reasoning_output])
    clear_btn.click(clear_history, None, [chatbot, state, user_input, reasoning_output])

demo.launch(share=True)
