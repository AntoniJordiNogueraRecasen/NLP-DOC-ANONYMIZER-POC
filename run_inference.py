import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def run_inference(input_text):
    # Load the trained model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained("./ner_model")
    tokenizer = AutoTokenizer.from_pretrained("./ner_model")

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)

    # Decode predictions
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predicted_labels = [model.config.id2label[label_id.item()] for label_id in predictions[0]]

    # Combine tokens and labels
    result = list(zip(tokens, predicted_labels))
    return result

if __name__ == "__main__":
    input_text = "Enter your text here for inference."
    result = run_inference(input_text)
    print("Inference result:", result)