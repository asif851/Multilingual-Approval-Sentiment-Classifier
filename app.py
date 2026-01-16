from flask import Flask, request, jsonify, render_template_string
import pickle
import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Flask app

app = Flask(__name__)


# Load pickles 

try:
    nb_model = pickle.load(open("log_model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    print("Model loaded successfully!")
    print(f"Model expects features: {nb_model.n_features_in_}")
    print(f"TF-IDF vocab size: {len(tfidf.vocabulary_)}")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please ensure 'nb_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory")
    nb_model = None
    tfidf = None

# ----------------------------
# NLTK setup
# ----------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Stopwords
stop_en = set(stopwords.words('english'))

# Bangla stopwords
try:
    stop_bn = set(stopwords.words('bengali'))
except:
    stop_bn = set()

# Romanized Bangla stopwords (dataset-specific)
stop_rn = {
    'ami', 'amra', 'tumi', 'tomra', 'se', 'tara', 'amar', 'tomar', 'tar', 'tader',
    'eta', 'ota', 'ekta', 'onek', 'sob', 'kisu', 'kichu', 'ke', 'kar', 'kare',
    'jodi', 'jodiw', 'karon', 'tai', 'ekhono', 'age', 'por', 'ekhon',
    'ar', 'kintu', 'ba', 'ta', 'tobe', 'tore', 'nay', 'na', 'noy', 'ache', 'chilo', 'thake',
    'jonno', 'sate', 'diyese', 'bolse', 'bole', 'korse', 'korche', 'korse', 'korte',
}

stemmer = PorterStemmer()

# ----------------------------
# Text Cleaning Function
# ----------------------------
def clean_text(text):
    """Clean text: remove URLs, hashtags, punctuation, normalize spaces."""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)    # remove URLs
    text = re.sub(r'@\w+|#\w+', ' ', text)                # remove mentions, hashtags
    text = re.sub(r'[^\u0980-\u09FFA-Za-z\s]', ' ', text) # keep Bangla + English
    text = re.sub(r'\s+', ' ', text).strip()              # normalize spaces
    return text

# ----------------------------
# Preprocessing Function
# ----------------------------
def preprocess_text(text, lang=None):
    """Tokenize, remove stopwords for English/Bangla/Romanized, stem English."""
    text = clean_text(text)
    tokens = word_tokenize(text)
    lang = (lang or '').upper()

    if lang == 'BN':  # Bangla
        tokens = [t for t in tokens if t not in stop_bn and len(t) > 1]

    elif lang == 'EN':  # English
        tokens = [stemmer.stem(t) for t in tokens if t not in stop_en and len(t) > 2]

    elif lang == 'RN':  # Romanized Bangla
        tokens = [t for t in tokens if t not in stop_rn and len(t) > 1]

    else:  # fallback for unknown language
        tokens = [t for t in tokens if len(t) > 1]

    return ' '.join(tokens)

# ----------------------------
# Language Detection Function (Simple)
# ----------------------------
def detect_language(text):
    """Simple language detection based on character ranges."""
    text = str(text)
    
    # Check for Bangla characters
    if re.search(r'[\u0980-\u09FF]', text):
        return 'BN'
    
    # Check for Romanized Bangla words
    romanized_words = ['ami', 'tumi', 'se', 'amar', 'tomar', 'ekta', 'onek', 'kisu']
    text_lower = text.lower()
    if any(word in text_lower for word in romanized_words):
        return 'RN'
    
    # Default to English
    return 'EN'

# ----------------------------
# Fix Feature Dimension Mismatch
# ----------------------------
def fix_feature_dimensions(X_transformed, expected_features):
    """Fix feature dimension mismatch between TF-IDF output and model expectations."""
    from scipy.sparse import hstack, csr_matrix
    
    if X_transformed.shape[1] == expected_features:
        return X_transformed
    
    # If we have more features than expected (shouldn't happen with same vectorizer)
    if X_transformed.shape[1] > expected_features:
        # Take only the first expected_features columns
        return X_transformed[:, :expected_features]
    
    # If we have fewer features, pad with zeros
    missing_features = expected_features - X_transformed.shape[1]
    zeros = csr_matrix((X_transformed.shape[0], missing_features))
    return hstack([X_transformed, zeros])

# ----------------------------
# HTML Template
# ----------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Classification - Approval Prediction</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 800px;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 1.1rem;
        }
        
        .content {
            padding: 40px;
        }
        
        .input-section {
            margin-bottom: 30px;
        }
        
        .input-section h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5rem;
        }
        
        textarea {
            width: 100%;
            min-height: 200px;
            padding: 20px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .language-selection {
            margin-top: 15px;
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        .language-selection label {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            padding: 10px 15px;
            background: #f5f5f5;
            border-radius: 8px;
            transition: background 0.3s;
        }
        
        .language-selection label:hover {
            background: #e8e8e8;
        }
        
        .language-selection input[type="radio"] {
            margin: 0;
        }
        
        .buttons {
            display: flex;
            gap: 15px;
            margin-top: 20px;
        }
        
        button {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        #predictBtn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        #clearBtn {
            background: #f5f5f5;
            color: #666;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .result-section {
            margin-top: 40px;
            display: none;
        }
        
        .result-section.active {
            display: block;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-box {
            background: #f9f9f9;
            border-radius: 10px;
            padding: 25px;
            border-left: 5px solid;
            transition: border-color 0.3s;
        }
        
        .result-box.approved {
            border-color: #28a745;
        }
        
        .result-box.not-approved {
            border-color: #dc3545;
        }
        
        .prediction {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .prediction-label {
            font-size: 1.8rem;
            font-weight: bold;
            color: #333;
        }
        
        .prediction-status {
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.9rem;
            letter-spacing: 1px;
        }
        
        .status-approved {
            background: #d4edda;
            color: #155724;
        }
        
        .status-not-approved {
            background: #f8d7da;
            color: #721c24;
        }
        
        .confidence-meter {
            margin: 20px 0;
        }
        
        .confidence-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-weight: 500;
        }
        
        .confidence-bar {
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 5px;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        
        .confidence-fill.approved {
            background: linear-gradient(90deg, #28a745, #5cb85c);
        }
        
        .confidence-fill.not-approved {
            background: linear-gradient(90deg, #dc3545, #e35d6a);
        }
        
        .confidence-percentages {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            color: #666;
        }
        
        .details {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-top: 15px;
        }
        
        .detail-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        
        .detail-item:last-child {
            border-bottom: none;
        }
        
        .detail-label {
            font-weight: 600;
            color: #666;
        }
        
        .detail-value {
            color: #333;
            font-family: monospace;
            max-width: 60%;
            overflow-wrap: break-word;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .prediction-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }
        
        .prediction-card {
            background: white;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .prediction-card.approved {
            border-top: 3px solid #28a745;
        }
        
        .prediction-card.not-approved {
            border-top: 3px solid #dc3545;
        }
        
        .prediction-card h4 {
            margin-bottom: 10px;
            color: #333;
        }
        
        .prediction-card .percentage {
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .prediction-card.approved .percentage {
            color: #28a745;
        }
        
        .prediction-card.not-approved .percentage {
            color: #dc3545;
        }
        
        @media (max-width: 768px) {
            .container {
                margin: 10px;
                padding: 0;
            }
            
            .content {
                padding: 20px;
            }
            
            .header {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .buttons {
                flex-direction: column;
            }
            
            .language-selection {
                flex-direction: column;
                align-items: stretch;
            }
            
            .language-selection label {
                justify-content: center;
            }
            
            .prediction-details {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📝 Text Approval Classifier</h1>
            <p>AI-powered content moderation system supporting English, Bangla, and Romanized Bangla</p>
        </div>
        
        <div class="content">
            <div class="input-section">
                <h2>Enter Text to Analyze</h2>
                <textarea id="textInput" placeholder="Type or paste your text here... (Supports English, Bangla, and Romanized Bangla)"></textarea>
                
                <div class="language-selection">
                    <h3 style="margin-right: 15px;">Language:</h3>
                    <label>
                        <input type="radio" name="language" value="auto" checked> Auto-detect
                    </label>
                    <label>
                        <input type="radio" name="language" value="EN"> English
                    </label>
                    <label>
                        <input type="radio" name="language" value="BN"> Bangla
                    </label>
                    <label>
                        <input type="radio" name="language" value="RN"> Romanized Bangla
                    </label>
                </div>
                
                <div class="buttons">
                    <button id="predictBtn" onclick="predictText()">🔍 Predict Approval</button>
                    <button id="clearBtn" onclick="clearText()">🗑️ Clear Text</button>
                </div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing text...</p>
            </div>
            
            <div class="result-section" id="resultSection">
                <h2>Prediction Result</h2>
                <div class="result-box" id="resultBox">
                    <!-- Results will be inserted here -->
                </div>
            </div>
        </div>
    </div>

    <script>
        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }
        
        function showResult(show) {
            const resultSection = document.getElementById('resultSection');
            if (show) {
                resultSection.classList.add('active');
            } else {
                resultSection.classList.remove('active');
            }
        }
        
        async function predictText() {
            const textInput = document.getElementById('textInput').value.trim();
            if (!textInput) {
                alert('Please enter some text to analyze.');
                return;
            }
            
            const language = document.querySelector('input[name="language"]:checked').value;
            
            showLoading(true);
            showResult(false);
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: textInput,
                        language: language === 'auto' ? null : language
                    })
                });
                
                const data = await response.json();
                showLoading(false);
                displayResults(data);
            } catch (error) {
                showLoading(false);
                alert('Error: ' + error.message);
            }
        }
        
        function displayResults(data) {
            const resultBox = document.getElementById('resultBox');
            
            const isApproved = data.prediction === 'Approved';
            const approvedPercent = Math.round(data.probabilities['Approved'] * 100);
            const notApprovedPercent = Math.round(data.probabilities['Not Approved'] * 100);
            
            // Set result box class based on prediction
            resultBox.className = 'result-box ' + (isApproved ? 'approved' : 'not-approved');
            
            resultBox.innerHTML = `
                <div class="prediction">
                    <div class="prediction-label">Prediction:</div>
                    <div class="prediction-status ${isApproved ? 'status-approved' : 'status-not-approved'}">
                        ${data.prediction}
                    </div>
                </div>
                
                <div class="confidence-meter">
                    <div class="confidence-label">
                        <span>Confidence Level:</span>
                        <span>${Math.round(data.confidence * 100)}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill ${isApproved ? 'approved' : 'not-approved'}" 
                             style="width: ${Math.round(data.confidence * 100)}%">
                        </div>
                    </div>
                </div>
                
                <div class="prediction-details">
                    <div class="prediction-card approved">
                        <h4>Approved Probability</h4>
                        <div class="percentage">${approvedPercent}%</div>
                        <small>Class 0</small>
                    </div>
                    
                    <div class="prediction-card not-approved">
                        <h4>Not Approved Probability</h4>
                        <div class="percentage">${notApprovedPercent}%</div>
                        <small>Class 1</small>
                    </div>
                </div>
                
                <div class="details">
                    <div class="detail-item">
                        <div class="detail-label">Detected Language:</div>
                        <div class="detail-value">${data.detected_language || 'Not detected'}</div>
                    </div>
                    
                    <div class="detail-item">
                        <div class="detail-label">Processed Language:</div>
                        <div class="detail-value">${data.processed_language || 'Auto'}</div>
                    </div>
                    
                    <div class="detail-item">
                        <div class="detail-label">Cleaned Text:</div>
                        <div class="detail-value">${data.cleaned_text || 'None'}</div>
                    </div>
                    
                    <div class="detail-item">
                        <div class="detail-label">Features Used:</div>
                        <div class="detail-value">${data.features_used || 0}</div>
                    </div>
                    
                    ${data.raw_prediction !== undefined ? `
                    <div class="detail-item">
                        <div class="detail-label">Raw Prediction (0/1):</div>
                        <div class="detail-value">${data.raw_prediction}</div>
                    </div>
                    ` : ''}
                    
                    ${data.error ? `
                    <div class="detail-item">
                        <div class="detail-label">Error:</div>
                        <div class="detail-value" style="color: #dc3545;">${data.error}</div>
                    </div>
                    ` : ''}
                </div>
            `;
            
            showResult(true);
            
            // Scroll to results
            resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
        
        function clearText() {
            document.getElementById('textInput').value = '';
            showResult(false);
        }
        
        // Allow pressing Ctrl+Enter to submit
        document.getElementById('textInput').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                predictText();
            }
        });
    </script>
</body>
</html>
"""

# ----------------------------
# Home Route
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

# ----------------------------
# API Prediction Endpoint
# ----------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    if nb_model is None or tfidf is None:
        return jsonify({
            "error": "Model not loaded. Please check server logs.",
            "prediction": "Error",
            "confidence": 0.0
        }), 500
    
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({
            "error": "text field is required",
            "prediction": "Error",
            "confidence": 0.0
        }), 400
    
    text = data["text"]
    user_language = data.get("language")
    
    # Detect language if not specified
    if not user_language:
        detected_lang = detect_language(text)
    else:
        detected_lang = user_language
    
    # Preprocess text
    cleaned_text = preprocess_text(text, detected_lang)
    
    # Check if we have meaningful text after preprocessing
    if not cleaned_text.strip():
        return jsonify({
            "input": text,
            "cleaned_text": cleaned_text,
            "detected_language": detected_lang,
            "processed_language": detected_lang,
            "prediction": "Not Approved",  # Default to Not Approved for empty text
            "confidence": 0.0,
            "raw_prediction": 1,
            "probabilities": {
                "Approved": 0.0,
                "Not Approved": 1.0
            },
            "error": "No meaningful text after preprocessing"
        })
    
    try:
        # Transform with TF-IDF
        X = tfidf.transform([cleaned_text])
        
        # Debug info
        print(f"\n=== PREDICTION DEBUG ===")
        print(f"Input text length: {len(text)}")
        print(f"Cleaned text: {cleaned_text}")
        print(f"Detected language: {detected_lang}")
        print(f"TF-IDF features: {X.shape[1]}")
        print(f"Model expects: {nb_model.n_features_in_}")
        
        # Fix dimension mismatch if needed
        if X.shape[1] != nb_model.n_features_in_:
            print(f"WARNING: Feature mismatch! Fixing dimensions...")
            X = fix_feature_dimensions(X, nb_model.n_features_in_)
            print(f"Fixed features: {X.shape[1]}")
        
        # Make prediction
        pred = nb_model.predict(X)[0]  # This will be 0 or 1
        proba = nb_model.predict_proba(X)[0]
        
        # IMPORTANT: Map predictions correctly
        # pred = 0 → Approved
        # pred = 1 → Not Approved
        prediction_label = "Approved" if pred == 0 else "Not Approved"
        
        # Get probabilities for both classes
        # Assuming sklearn order: [class_0, class_1] = [Approved, Not Approved]
        proba_approved = float(proba[0]) if len(proba) > 0 else 0
        proba_not_approved = float(proba[1]) if len(proba) > 1 else 0
        
        # Confidence is the probability of the predicted class
        confidence = proba_approved if pred == 0 else proba_not_approved
        
        result = {
            "input": text,
            "cleaned_text": cleaned_text,
            "detected_language": detected_lang,
            "processed_language": detected_lang,
            "prediction": prediction_label,
            "raw_prediction": int(pred),  # Include raw 0/1 prediction
            "confidence": confidence,
            "probabilities": {
                "Approved": proba_approved,
                "Not Approved": proba_not_approved
            },
            "features_used": X.getnnz()
        }
        
        print(f"Raw prediction: {pred} (0=Approved, 1=Not Approved)")
        print(f"Prediction: {prediction_label} (Confidence: {confidence:.2%})")
        print(f"Probabilities - Approved: {proba_approved:.2%}, Not Approved: {proba_not_approved:.2%}")
        print("=== END DEBUG ===\n")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            "input": text,
            "cleaned_text": cleaned_text,
            "detected_language": detected_lang,
            "error": str(e),
            "prediction": "Error",
            "confidence": 0.0,
            "raw_prediction": -1
        }), 500

# ----------------------------
# Health Check Endpoint
# ----------------------------
@app.route("/health", methods=["GET"])
def health_check():
    model_status = "loaded" if nb_model is not None else "not loaded"
    tfidf_status = "loaded" if tfidf is not None else "not loaded"
    
    return jsonify({
        "status": "healthy",
        "model": model_status,
        "tfidf": tfidf_status,
        "model_features": nb_model.n_features_in_ if nb_model else None,
        "tfidf_vocab_size": len(tfidf.vocabulary_) if tfidf else None,
        "label_mapping": {
            "0": "Approved",
            "1": "Not Approved"
        }
    })

# ----------------------------
# Test Endpoint for Debugging
# ----------------------------
@app.route("/api/debug", methods=["POST"])
def debug():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("language", "auto")
    
    if lang == "auto":
        detected_lang = detect_language(text)
    else:
        detected_lang = lang
    
    cleaned = preprocess_text(text, detected_lang)
    
    # Check which words are in vocabulary
    words_in_text = cleaned.split()
    vocab_matches = []
    if tfidf is not None:
        vocab = tfidf.vocabulary_
        vocab_matches = [word for word in words_in_text if word in vocab]
    
    return jsonify({
        "original_text": text,
        "detected_language": detected_lang,
        "cleaned_text": cleaned,
        "tokens": word_tokenize(cleaned),
        "words_in_vocab": vocab_matches,
        "vocab_match_count": len(vocab_matches),
        "total_words": len(words_in_text)
    })

# ----------------------------
# Sample Test Endpoint
# ----------------------------
@app.route("/api/test", methods=["GET"])
def test_samples():
    """Test with sample texts"""
    samples = [
        {"text": "This is a great product! I love it.", "language": "EN", "expected": "Approved"},
        {"text": "Very bad quality, waste of money.", "language": "EN", "expected": "Not Approved"},
        {"text": "আমি খুব খুশি এই পণ্যটি নিয়ে", "language": "BN", "expected": "Approved"},
        {"text": "খারাপ পণ্য, টাকা নষ্ট", "language": "BN", "expected": "Not Approved"},
        {"text": "ami khub valo lagse", "language": "RN", "expected": "Approved"},
        {"text": "kharap jinish", "language": "RN", "expected": "Not Approved"}
    ]
    
    return jsonify({
        "samples": samples,
        "note": "These are example texts for testing the classifier"
    })

# ----------------------------
# Run the app
# ----------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Text Classification Flask Application")
    print("="*50)
    print("Label Mapping: 0 = Approved, 1 = Not Approved")
    print("-"*50)
    
    if nb_model and tfidf:
        print(f"✓ Model loaded: Expects {nb_model.n_features_in_} features")
        print(f"✓ TF-IDF loaded: {len(tfidf.vocabulary_)} vocabulary terms")
        
        # Try to get class names if available
        try:
            if hasattr(nb_model, 'classes_'):
                print(f"✓ Model classes: {nb_model.classes_}")
                print(f"  (Index 0: Approved, Index 1: Not Approved)")
        except:
            pass
    else:
        print("✗ Warning: Model files not loaded")
        print("  Please ensure 'nb_model.pkl' and 'tfidf_vectorizer.pkl' exist")
    
    print(f"\n📱 Web Interface: http://localhost:5000")
    print(f"🔧 Health Check: http://localhost:5000/health")
    print(f"🐛 Debug API: POST http://localhost:5000/api/debug")
    print(f"🧪 Test Samples: http://localhost:5000/api/test")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)