from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
stop_tokens = set(stopwords.words("english"))

def clean_token(tok):
  return tok.lstrip("Ġ").lower()
