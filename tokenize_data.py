from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID = "PORTULAN/gervasio-8b-portuguese-ptpt-decoder"
INPUT_FILE = r"C:\Users\afons\Desktop\Afonso\Masters_Thesis\Data\FineTuning_Gervasio\Re__dados_para_fine-tuning\Arte\no_failed_state\pt_no_failed_state_final.jsonl"
OUTPUT_FILE = r"C:\Users\afons\Desktop\Afonso\Masters_Thesis\Data\FineTuning_Gervasio\Re__dados_para_fine-tuning\Arte\no_failed_state\tokenized\pt_no_failed_state_final.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def format(row):
    conversation = row["messages"]
    formatted_text = tokenizer.apply_chat_template(conversation, tokenize=False)
    return {"text": formatted_text}

dataset = load_dataset("json", data_files=INPUT_FILE, split="train")

dataset = dataset.map(format)

print(f"Saving to {OUTPUT_FILE}...")
dataset.select_columns(["text"]).to_json(OUTPUT_FILE)

print("\n--- SUCCESS! Preview of the formatted text: ---")
print(dataset[0]["text"])

