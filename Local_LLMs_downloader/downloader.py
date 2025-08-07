from sentence_transformers import SentenceTransformer

model_name = "intfloat/e5-base-v2"  # change this to any from the list
model = SentenceTransformer(model_name)
model.save(f"./models/{model_name.split('/')[-1]}")
