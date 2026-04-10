import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import numpy as np

def get_page_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

def compute_jaccard(text1, text2):
    vectorizer = CountVectorizer(binary=True, stop_words='english')
    vectors = vectorizer.fit_transform([text1, text2]).toarray()
    return jaccard_score(vectors[0], vectors[1])

def analyze_url(url):
    page_text = get_page_text(url)
    sample_text = "social media"  # 

    result = {
        "ranking": 0, 
        "mld_res": 0, 
        "mld.ps_res": 0, 
        "card_rem": len(set(page_text.split())), 
        "ratio_Rrem": 0.5, 
        "ratio_Arem": 0.3,  
        "jaccard_RR": compute_jaccard(page_text, sample_text),
        "jaccard_RA": compute_jaccard(page_text[::-1], sample_text),
        "jaccard_AR": compute_jaccard(sample_text, page_text),
        "jaccard_AA": compute_jaccard(sample_text, sample_text),
        "jaccard_ARrd": compute_jaccard(sample_text.upper(), page_text),
        "jaccard_ARrem": compute_jaccard(sample_text.lower(), page_text),
    }

    return result

url = "https://en.wikipedia.org/wiki/Social_media"
values = analyze_url(url)

for k, v in values.items():
    print(f"{k}: {v}")