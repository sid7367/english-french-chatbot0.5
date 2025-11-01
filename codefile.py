from transformers import MarianMTModel, MarianTokenizer

# Load pre-trained model and tokenizer for English→French
model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name) # converts text to tokens
model = MarianMTModel.from_pretrained(model_name)  # performs the translation

# Sample English sentences
sentences = [
    "Hello, how are you?",
    "I am learning Python programming.",
    "The weather is beautiful today.",
    "Let's build a translation chatbot."
]

# Translate sentences
for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt", padding=True)
    translated_tokens = model.generate(**inputs)
    french_translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    print(f"English: {sentence}")
    print(f"French : {french_translation}\n")
