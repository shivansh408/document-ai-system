import os
import fitz 
import pandas as pd
import requests
import logging
from concurrent.futures import ThreadPoolExecutor
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

# Setting up logging to keep track (just adding a feature)
logging.basicConfig(filename='document_ai.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def extract_text_from_pdf(pdf_path):
    logging.info(f"Starting text extraction from {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        logging.info(f"Completed text extraction from {pdf_path}")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return None

def call_document_ai_api(extracted_text):
    try:
        # Replace with your Hugging Face API and model
        api_url = "https://api-inference.huggingface.co/models/your-model"
        headers = {"Authorization": f"Bearer YOUR_HUGGINGFACE_API_TOKEN"}
        data = {"inputs": extracted_text}
        
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        
        logging.info("Successfully received response from Document AI API")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return None

def save_to_excel(data, output_excel_path):
    try:
        with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
            for file_name, details in data.items():
                df = pd.DataFrame([details])
                df.to_excel(writer, sheet_name=os.path.basename(file_name)[:31], index=False)
        logging.info(f"Data successfully saved to {output_excel_path}")
    except Exception as e:
        logging.error(f"Error saving data to Excel: {e}")

def process_pdf(pdf_path):
    extracted_text = extract_text_from_pdf(pdf_path)
    if extracted_text:
        extracted_details = call_document_ai_api(extracted_text)
        return extracted_details
    return None

# Main function to handle multiple PDFs
def main(pdf_dir, output_excel_path):
    try:
        pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            logging.warning("No PDF files found in the specified directory")
            return

        logging.info(f"Found {len(pdf_files)} PDF files to process")
        
        extracted_data = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(process_pdf, pdf_files)

        for pdf_file, result in zip(pdf_files, results):
            if result:
                extracted_data[pdf_file] = result

        if extracted_data:
            save_to_excel(extracted_data, output_excel_path)
        else:
            logging.warning("No data extracted from PDFs")

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")

#change path_to_your_pdf_directory into your dataset pdf directory
if __name__ == "__main__":
    pdf_dir = "path_to_your_pdf_directory"
    output_excel_path = "output_file.xlsx"
    main(pdf_dir, output_excel_path)

# Load dataset (DeepForm is recommended from me)
dataset = load_dataset('path_to_your_dataset')

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

def preprocess_data(examples):
    images = [image.convert("RGB") for image in examples['image']]
    words = examples["words"]
    boxes = examples["bbox"]
    word_labels = examples["ner_tags"]

    encoding = processor(images, words, boxes=boxes, word_labels=word_labels, truncation=True, padding="max_length")

    return encoding

encoded_dataset = dataset.map(preprocess_data, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
)

trainer.train()

model.save_pretrained("path_to_save_your_model")
processor.save_pretrained("path_to_save_your_processor")
