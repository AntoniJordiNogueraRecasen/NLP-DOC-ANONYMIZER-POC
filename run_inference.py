import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import joblib  # For loading the LabelEncoder

def run_inference(input_text):
    # Load the trained model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained("./ner_model")
    tokenizer = AutoTokenizer.from_pretrained("./ner_model")

    # Load the LabelEncoder used during training
    label_encoder = joblib.load("./ner_model/label_encoder.pkl")

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)

    # Decode predictions
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Map predicted labels to their category names
    predicted_categories = [
        label_encoder.inverse_transform([label_id.item()])[0] if label_id.item() in label_encoder.classes_ else "UNKNOWN"
        for label_id in predictions[0]
    ]

    # Combine tokens and categories
    result = list(zip(tokens, predicted_categories))

    # Filter out special tokens and format the output
    filtered_result = [(token, category) for token, category in result if token not in ["[CLS]", "[SEP]", "[PAD]"]]
    print("Inference result:")
    for token, category in filtered_result:
        print(f"Token: {token}, Category: {category}")

if __name__ == "__main__":
    input_text = "¡Hola! Mi nombre es John Doe y soy un ingeniero de software."
    run_inference(input_text)