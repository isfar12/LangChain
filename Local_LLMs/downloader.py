from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel

model_name = "sentence-transformers/all-MiniLM-L6-v2"
save_dir = "./models/sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
