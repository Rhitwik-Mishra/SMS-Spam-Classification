# SMS Spam Classifier (Streamlit)

## Run locally
1. Create & activate a virtual environment.
2. Install dependencies:
pip install -r requirements.txt
3. Run:
streamlit run app.py

## Notes
- Add `nltk.download('punkt', quiet=True)` and `nltk.download('stopwords', quiet=True)` at top of app.py so NLTK works on cloud.
- If model files are large, consider Git LFS or external storage.
