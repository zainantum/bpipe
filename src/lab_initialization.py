import spacy
import json
import torch
import requests
from transformers import pipeline
from argostranslate import translate as _translate
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from huggingface_hub import hf_hub_download
import logging

def initialize_models(device):
    logging.info("[TAGGING] Initializing models to be pre-ready for batch processing:")
    models = {}
    
    logging.info("[TAGGING] Loading model: MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33")
    models['zs_pipe'] = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
        device=device
    )
    logging.info("[TAGGING] Loading model: sentence-transformers/all-MiniLM-L6-v2")
    models['sentence_transformer'] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    text_classification_models = [
        ("Emotion", "SamLowe/roberta-base-go_emotions"),
        ("Irony", "cardiffnlp/twitter-roberta-base-irony"),
        ("TextType", "marieke93/MiniLM-evidence-types"),
    ]
    for col_name, model_name in text_classification_models:
        logging.info(f"[TAGGING] Loading model: {model_name}")
        models[col_name] = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device=device,
            max_length=512,
            padding=True,
        )
    
    logging.info("[TAGGING] Loading model: bert-large-uncased")
    models['bert_tokenizer'] = AutoTokenizer.from_pretrained("bert-large-uncased")
    logging.info("[TAGGING] Loading model: vaderSentiment")
    models['sentiment_analyzer'] = SentimentIntensityAnalyzer()
    try:
        emoji_lexicon = hf_hub_download(
            repo_id="ExordeLabs/SentimentDetection",
            filename="emoji_unic_lexicon.json",
        )
        loughran_dict = hf_hub_download(
            repo_id="ExordeLabs/SentimentDetection", filename="loughran_dict.json"
        )
        logging.info("[TAGGING] Loading Loughran_dict & unic_emoji_dict for sentiment_analyzer.")
        with open(emoji_lexicon) as f:
            unic_emoji_dict = json.load(f)
        with open(loughran_dict) as f:
            Loughran_dict = json.load(f)
        models['sentiment_analyzer'].lexicon.update(Loughran_dict)
        models['sentiment_analyzer'].lexicon.update(unic_emoji_dict)
    except Exception as e:
        logging.info("[TAGGING] Error loading Loughran_dict & unic_emoji_dict for sentiment_analyzer. Doing without.")
    
    logging.info("[TAGGING] Loading model: mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    models['fdb_tokenizer'] = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    logging.info("[TAGGING] Loading model: mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    models['fdb_model'] = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    models['fdb_pipe'] = pipeline(
        "text-classification",
        model=models['fdb_model'],
        tokenizer=models['fdb_tokenizer'],
        top_k=None, 
        max_length=512,
        padding=True,
    )
    
    logging.info("[TAGGING] Loading model: lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    models['gdb_tokenizer'] = AutoTokenizer.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    logging.info("[TAGGING] Loading model: lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    models['gdb_model'] = AutoModelForSequenceClassification.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    models['gdb_pipe'] = pipeline(
        "text-classification",
        model=models['gdb_model'],
        tokenizer=models['gdb_tokenizer'],
        top_k=None, 
        max_length=512,
        padding=True,
    )
    logging.info("[TAGGING] Models loaded successfully.")
    
    return models



def lab_initialization():
    device = torch.cuda.current_device() if torch.cuda.is_available() else -1
    mappings = {
        "Gender": {0: "Female", 1: "Male"},
        "Age": {0: "<20", 1: "20<30", 2: "30<40", 3: ">=40"},
        # "HateSpeech": {0: "Hate speech", 1: "Offensive", 2: "None"},
    }
    try:
        nlp = spacy.load("en_core_web_trf")
    except Exception as err:
        logging.exception("Could not load en_core_web_trf")
        raise err
    installed_languages = _translate.get_installed_languages()
    models = initialize_models(device)
    labels = requests.get(
        "https://raw.githubusercontent.com/exorde-labs/TestnetProtocol/main/targets/class_names.json"
    ).json()
    return {
        "labeldict": labels,
        "device": device, # bpipe & upipe
        "mappings": mappings, #bpipe
        "nlp": nlp, #bpipe
        "max_depth": 2,
        "remove_stopwords": False,
        "models": models
    }
