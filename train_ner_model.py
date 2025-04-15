import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, TrainerCallback
from sklearn.preprocessing import LabelEncoder

def train_model():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset from CSV
    df = pd.read_csv("./database/Resume.csv")  # Replace with your CSV file path

    # Encode the Category column into numerical labels
    label_encoder = LabelEncoder()
    df["Category_encoded"] = label_encoder.fit_transform(df["Category"])

    # Convert the DataFrame to a Hugging Face Dataset
    dataset = Dataset.from_pandas(df[["Resume_str", "Category_encoded"]])

    # Split the dataset into train and validation sets
    dataset = dataset.train_test_split(test_size=0.2)

    # Load tokenizer and model
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = len(label_encoder.classes_)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

    # Tokenize dataset
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["Resume_str"], truncation=True, padding="max_length", max_length=512)
        tokenized_inputs["labels"] = examples["Category_encoded"]
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

    # Define metrics
    from evaluate import load
    metric = load("accuracy")

    def compute_metrics(p):
        predictions, labels = p
        predictions = predictions.argmax(axis=1)
        accuracy = metric.compute(predictions=predictions, references=labels)
        return {"accuracy": accuracy["accuracy"]}

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("./ner_model")
    tokenizer.save_pretrained("./ner_model")
    print("Model training complete and saved to './ner_model'.")

if __name__ == "__main__":
    train_model()