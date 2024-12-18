# import torch
# from transformers import BertForTokenClassification, BertTokenizer
# from transformers import pipeline
#
# # Load the saved BERT model and tokenizer
# model = BertForTokenClassification.from_pretrained("./bert_car_model")
# tokenizer = BertTokenizer.from_pretrained("./bert_car_model")
#
# # Create a pipeline for token classification
# nlp_bert = pipeline("ner", model=model, tokenizer=tokenizer)
#
# # Sample text for testing
# test_text = "The 2020 Chevrolet Silverado with 60,000 miles is available at the Copart Houston auction."
#
# # Run the model on the test text
# entities = nlp_bert(test_text)
#
# # Print the entities identified by the model
# print(f"Entities in the text: {entities}")


import spacy

# Load the saved spaCy model
nlp_spacy = spacy.load("./spacy_car_model")

# Sample text for testing
test_text = "The 2021 Audi A8 sedan with 60,000 miles is available at the Copart-Houston auction."

# Process the text with the spaCy model
doc = nlp_spacy(test_text)

# Print the entities identified by the model
print(f"Entities in the text: {[(ent.text, ent.label_) for ent in doc.ents]}")



# import requests
# import json
#
# url = 'https://c-solr9-dev.copart.com/solr/c_dev4_onsale_lots_c/select'
#
# query_json = {
#     "query": "*:*",
#     "filter": ["location_city:dallas"],
#     "params": {
#         "q.op": "OR",
#         "defType": "edismax",
#         "qf": "lot_make_desc^2 location_city",
#         "bq": "lot_make_desc:(BMW OR Mercedes OR Audi OR Volkswagen OR Porsche) AND lot_model_group:BMW^2"
#     }
# }
#
# params = {
#     'q': '*:*',
#     'indent': 'true',
#     'json': json.dumps(query_json)
# }
#
# response = requests.get(url, params=params)
# print(response.status_code)
# print(response.text)
#
#
