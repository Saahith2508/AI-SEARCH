import os
import spacy
import json
import random
from flask import Flask, request, jsonify, render_template
from transformers import BertForTokenClassification, BertTokenizer, Trainer, TrainingArguments, pipeline
from spacy.training.example import Example
from datasets import load_dataset
import pysolr
from werkzeug.utils import secure_filename
from flask_cors import CORS
from flask import Flask, request, jsonify
import spacy
import re
import json
from transformers import pipeline
from spellchecker import SpellChecker
from typing import Dict, Any
import pysolr
import os

app = Flask(__name__)

# Paths for saving models
MODEL_SAVE_PATH = "models"

# Enable CORS for all routes and origins
CORS(app)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load BERT-based NER model
bert_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Solr client setup
SOLR_URL = "https://c-solr.copart.com/solr/#/c_lots_us"
solr = pysolr.Solr(SOLR_URL, timeout=10)

# Directory for saving uploaded training data
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'json'}



# Initialize spell checker
spell = SpellChecker()

# Define dictionaries for specific categories
CAR_BRANDS_BY_REGION = {
    "german": ["BMW", "Mercedes", "Audi", "Volkswagen", "Porsche"],
    "american": ["Ford", "Chevrolet", "Tesla", "Dodge"],
    "japanese": ["Toyota", "Honda", "Nissan", "Mazda", "Subaru"],
}

PRICE_CATEGORIES = {
    "cheap": 10000,
    "expensive": 50000,
}

DAMAGE_TYPES = ["water damaged", "fire damaged", "salvage"]

CAR_TYPES = ["suv", "sedan", "truck", "convertible", "coupe", "hatchback"]

LOCATION_DATA = [
    {
        "location": "Los Angeles",
        "latitude": 34.0522,
        "longitude": -118.2437,
        "total_spent": 95000,
        "most_bought_car": {"make": "BMW", "model": "X5", "count": 3},
        "frequent_cars": ["BMW X5", "Tesla Model 3", "Chevrolet Malibu"]
    },
    {
        "location": "New York",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "total_spent": 120000,
        "most_bought_car": {"make": "Tesla", "model": "Model 3", "count": 5},
        "frequent_cars": ["Tesla Model 3", "Ford Mustang", "Audi A4"]
    }
]


# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to train spaCy NER model
def train_spacy_ner(training_data, output_dir):
    # Load pre-trained spaCy model
    nlp = spacy.load("en_core_web_trf")  # Transformer-based model

    # Disable unneeded components to speed up training
    nlp.disable_pipes("ner", "parser", "textcat")

    # Add new labels to the NER pipeline
    ner = nlp.get_pipe("ner")
    for label in training_data['labels']:
        ner.add_label(label)

    # Prepare training data
    train_examples = []
    for text, annot in training_data['data']:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annot)
        train_examples.append(example)

    # Training loop
    optimizer = nlp.begin_training()
    for i in range(10):  # Epochs
        random.shuffle(train_examples)
        for example in train_examples:
            nlp.update([example], drop=0.5)  # Dropout for regularization

    # Save the fine-tuned model
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")

# Function to train BERT model for NER
def train_bert_ner(training_data, output_dir):
    # Load pre-trained BERT model and tokenizer
    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(training_data['labels']))
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Prepare dataset (in BIO format)
    def tokenize_and_align_labels(example):
        tokenized_inputs = tokenizer(example["tokens"], truncation=True, padding="max_length", is_split_into_words=True)
        labels = example["ner_tags"]
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Load your custom dataset (replace with your file handling logic)
    dataset = load_dataset('json', data_files=training_data['data'])
    dataset = dataset.map(tokenize_and_align_labels)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )

    # Start training
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


# Load buyer data from a file
def load_buyer_data(file_path='buyers.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        return {"buyers": []}

# Function to get the closest car make/model
def get_closest_car_make_model(word: str) -> str:
    closest_make = spell.correction(word)
    if closest_make in sum(CAR_BRANDS_BY_REGION.values(), []):
        return closest_make
    return word

# Function to extract price from text
def extract_price(price_text: str) -> int:
    price_match = re.search(r"\d+", price_text)
    return int(price_match.group(0)) if price_match else None

CAR_COLORS = ["black", "white", "red", "blue", "silver", "gray", "green", "yellow", "orange", "brown", "purple"]
YEAR_CATEGORIES = {
    "under": lambda x: f"lot_year:[* TO {x - 1}]",
    "before": lambda x: f"lot_year:[* TO {x - 1}]",
    "after": lambda x: f"lot_year:[{x + 1} TO *]",
    "since": lambda x: f"lot_year:[{x + 1} TO *]",
}

# Update the `parse_query` function to extract lot year and color
def parse_query(query: str, buyer_data: Dict[str, Any]) -> Dict[str, Any]:
    structured_data = {
        "make": [],
        "model": None,
        "type": None,
        "price": {"min": None, "max": None},
        "location": None,
        "damage": None,
        "buyer_spent": None,
        "most_bought_car": None,
        "color": None,
        "lot_year": None
    }

    corrected_query = " ".join([get_closest_car_make_model(word) for word in query.split()])
    doc = nlp(corrected_query)

    for ent in doc.ents:
        if ent.label_ == "ORG" and ent.text not in structured_data["make"]:
            structured_data["make"].append(ent.text)
        elif ent.label_ == "PRODUCT":
            structured_data["model"] = ent.text
        elif ent.label_ == "MONEY":
            price = extract_price(ent.text)
            if price:
                structured_data["price"]["max"] = price
        elif ent.label_ == "GPE":
            structured_data["location"] = ent.text

    # Parsing for car colors
    for color in CAR_COLORS:
        if color in corrected_query.lower():
            structured_data["color"] = color

    # Parsing for lot year (under, before, after, since)
    for keyword, lambda_func in YEAR_CATEGORIES.items():
        if keyword in corrected_query.lower():
            year_match = re.search(rf"{keyword}\s?(\d{4})", corrected_query)
            if year_match:
                structured_data["lot_year"] = lambda_func(int(year_match.group(1)))

    if "under" in corrected_query or "below" in corrected_query:
        price_match = re.search(r"(?:under|below)\s?(\d+)", corrected_query, re.IGNORECASE)
        if price_match:
            structured_data["price"]["max"] = int(price_match.group(1))

    for region, brands in CAR_BRANDS_BY_REGION.items():
        if region in corrected_query.lower():
            structured_data["make"].extend(brands)

    for category, max_price in PRICE_CATEGORIES.items():
        if category in corrected_query.lower():
            structured_data["price"]["max"] = max_price

    for damage in DAMAGE_TYPES:
        if damage in corrected_query.lower():
            structured_data["damage"] = damage

    for word in corrected_query.lower().split():
        if word in CAR_TYPES:
            structured_data["type"] = word

    if buyer_data["buyers"]:
        for buyer in buyer_data["buyers"]:
            structured_data["buyer_spent"] = buyer["total_spent"]
            structured_data["most_bought_car"] = buyer["most_bought_car"]

    return structured_data

# Update Solr query generation to include color and lot year
def generate_solr_query(parsed_data: Dict[str, Any]) -> str:
    query = []

    location_boost = 1.0
    if parsed_data["location"]:
        for location in LOCATION_DATA:
            if parsed_data["location"] == location["location"]:
                location_boost = location["total_spent"] / 100000.0

    if parsed_data["make"]:
        query.append(f"lot_make_desc:({' OR '.join(parsed_data['make'])})")

    if parsed_data["model"]:
        query.append(f"lot_model_group:{parsed_data['model']}")

    if parsed_data["price"]["max"]:
        query.append(f"price:[* TO {parsed_data['price']['max']}]")

    if parsed_data["location"]:
        query.append(f"yard_name:{parsed_data['location']}^2")

    if parsed_data["damage"]:
        query.append(f"damage:{parsed_data['damage']}")

    if parsed_data["color"]:
        query.append(f"lot_color:{parsed_data['color']}")

    if parsed_data["lot_year"]:
        query.append(f"lot_year:{parsed_data['lot_year']}")

    if parsed_data["most_bought_car"]:
        query.append(f"lot_model_group:{parsed_data['most_bought_car']}^2")

    return " AND ".join(query) + f"^ {location_boost}"


# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for training models
@app.route('/train', methods=['POST'])
def train_model():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            with open(file_path, 'r') as f:
                training_data = json.load(f)

            model_type = request.form.get("model")
            output_dir = os.path.join(MODEL_SAVE_PATH, filename.rsplit('.', 1)[0])

            if model_type == "spacy":
                train_spacy_ner(training_data, output_dir)
            elif model_type == "bert":
                train_bert_ner(training_data, output_dir)
            else:
                return jsonify({"error": "Invalid model type"}), 400

            return jsonify({"message": f"Model trained and saved at {output_dir}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route for generating Solr query
@app.route('/generate-query', methods=['POST'])
def generate_query():
    try:
        data = request.get_json()

        query = data.get("query", "")
        buyer_id = data.get("buyer_id", "")

        # Load buyer data from JSON file
        buyer_data = load_buyer_data()

        # Find buyer data for the given buyer_id
        selected_buyer = next((buyer for buyer in buyer_data["buyers"] if buyer["buyer_id"] == buyer_id), None)

        if not selected_buyer:
            return jsonify({"error": "Buyer not found"}), 404

        # Parse the query and buyer data
        structured_query = parse_query(query, {"buyers": [selected_buyer]})
        solr_query = generate_solr_query(structured_query)

        print(solr_query)
        # Query Solr with the generated query
        solr_results = solr.search(solr_query)

        return jsonify({"solr_query": solr_query, "results": solr_results.docs})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)



