# French Medical NER with DrBERT 🧬

A Streamlit application for Named Entity Recognition (NER) in French medical texts using a fine-tuned DrBERT model.

## Overview

This project provides a web interface for extracting medical entities from French clinical text using a transformer model fine-tuned on the QUAERO medical corpus. The application supports real-time entity extraction and provides interactive visualizations of the results.


the ner_project.ipynb notebook demonstrates the development and evaluation of  various approaches, from traditional machine learning to transformer-based models.


## Features

- 🔍 Real-time medical entity extraction
- 📊 Interactive visualizations:
  - Entity distribution charts
  - Confidence score distributions
  - Entity cards with confidence indicators
- 📥 Export results to CSV
- 🎯 Entity types supported:
  - DISO (Disorders)
  - PROC (Procedures)
  - ANAT (Anatomical structures)
  - CHEM (Chemicals & Drugs)
  - And more...

## Installation



1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. Open the application in your web browser
2. Enter or paste French medical text in the input area
3. Click "Extract Entities" to analyze the text
4. View results in different visualization modes:
   - Entity Cards
   - Entity Distribution
  
5. Download results as CSV if needed

## Model Details

- Base model: DrBERT (French medical BERT)
- Fine-tuned on: QUAERO medical corpus
- Hosted on: Hugging Face Hub
- Model ID: `abdel132/ner-drbert-quaero`

## Dependencies

- streamlit
- transformers
- pandas
- plotly
- torch

