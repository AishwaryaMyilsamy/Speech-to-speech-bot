from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss

model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

knowledge_base = [
    "Hi! I'm Rachel, your motivational chatbot.",
    "I'm here to brighten your day and help you stay positive!",
    "Always believe in yourself, and remember, you are capable of amazing things.",
    "Tell me how you're feeling, and I'll do my best to support you.",
    "You're doing great, and I'm here to remind you of your potential.",
    "If you're feeling stuck, don't worry—every problem has a solution!",
    "It's okay to take a break. Even small steps make a big difference over time."
]

knowledge_embeddings = embedding_model.encode(knowledge_base)
faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
faiss_index.add(knowledge_embeddings)

def retrieve_context(user_input):
  
    user_embedding = embedding_model.encode([user_input])
    distances, indices = faiss_index.search(user_embedding, 1)
    return knowledge_base[indices[0][0]]

def generate_response(user_input):
  
    context = retrieve_context(user_input)


    prompt = f"""
Rachel is a motivational chatbot with a positive and friendly personality.
Context: {context}
User: {user_input}
Rachel:"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Rachel:" in response:
        response = response.split("Rachel:")[-1].strip()
    return response
