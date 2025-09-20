# -------------------------------
# 1️⃣ Imports & Setup
# -------------------------------
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------------------
# 2️⃣ Load Dataset
# -------------------------------
dataset = load_dataset("ajibawa-2023/Software-Architecture", split="train")

def extract_fields(example):
    return {
        "input": example["input"],
        "output": example["output"]
    }

processed_data = dataset.map(extract_fields)
processed_data = processed_data.remove_columns([col for col in processed_data.column_names if col not in ["input", "output"]])

# -------------------------------
# 3️⃣ Tokenizer
# -------------------------------
student_model_name = "Qwen/Qwen3-4B-Thinking-2507"
teacher_model_name = "Kimi-K2-0905"

tokenizer = AutoTokenizer.from_pretrained(student_model_name)

MAX_LENGTH = 512

# -------------------------------
# 4️⃣ Dataset Class
# -------------------------------
class ArchitectureDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        output_text = item['output']

        # Tokenize input and output
        inputs = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        outputs = self.tokenizer(output_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")

        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': outputs.input_ids.squeeze()
        }

train_dataset = ArchitectureDataset(processed_data, tokenizer, MAX_LENGTH)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # adjust batch size based on GPU

# -------------------------------
# 5️⃣ Load Models
# -------------------------------
teacher = AutoModelForCausalLM.from_pretrained(teacher_model_name, output_hidden_states=True).to(device)
student = AutoModelForCausalLM.from_pretrained(student_model_name, output_hidden_states=True).to(device)
teacher.eval()

# Optional projection if hidden sizes differ
teacher_hidden_size = teacher.config.hidden_size
student_hidden_size = student.config.hidden_size
projection = nn.Linear(student_hidden_size, teacher_hidden_size).to(device) if student_hidden_size != teacher_hidden_size else nn.Identity()

# Optimizer
optimizer = optim.Adam(student.parameters(), lr=1e-5)

# -------------------------------
# 6️⃣ Training Loop
# -------------------------------
EPOCHS = 3

loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    total_loss = 0.0

    for step, batch in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Teacher forward
        with torch.no_grad():
            teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
            teacher_hidden = teacher_outputs.hidden_states[-1]

        # Student forward
        student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
        student_hidden = student_outputs.hidden_states[-1]

        # Project student hidden states if needed
        student_proj = projection(student_hidden)

        # Compute distillation loss
        distillation_loss = loss_fn(student_proj, teacher_hidden.detach())

        # Backpropagation
        optimizer.zero_grad()
        distillation_loss.backward()
        optimizer.step()

        total_loss += distillation_loss.item()
        if step % 50 == 0:
            print(f"Step {step}, Loss: {distillation_loss.item():.4f}")

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

# -------------------------------
# 7️⃣ Save Fine-Tuned Student Model
# -------------------------------
student.save_pretrained("./qwen3-4b-finetuned")
tokenizer.save_pretrained("./qwen3-4b-finetuned")

print("✅ Fine-tuning complete. Model saved at ./qwen3-4b-finetuned")
