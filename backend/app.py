from pathlib import Path
import os

from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from utils.predict import (
    CaptionModelError,
    InvalidImageError,
    PredictionService,
    compute_bleu
)

# ------------------ PATH SETUP ------------------
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / 'frontend'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}

# ------------------ FLASK APP ------------------
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024  # 6 MB limit

# ------------------ MODEL ------------------
predictor = PredictionService()

# ------------------ HELPERS ------------------
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------ ROUTES ------------------

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Check file
    if 'image' not in request.files:
        return jsonify({'message': 'No image file found in request.'}), 400

    image_file = request.files['image']

    # Get expected caption (optional)
    expected_caption = request.form.get("expected_caption")

    # Validate file
    if image_file.filename == '' or not allowed_file(image_file.filename):
        return jsonify({'message': 'Please upload a valid image file.'}), 400

    try:
        # Generate caption
        caption = predictor.predict(image_file)

        # BLEU Score (optional)
        bleu_score = None
        if expected_caption:
            bleu_score = compute_bleu(expected_caption, caption)

        return jsonify({
            'prediction': caption,
            'bleu': bleu_score
        }), 200

    except InvalidImageError as e:
        return jsonify({'message': str(e)}), 400

    except CaptionModelError as e:
        return jsonify({'message': str(e)}), 500

    except Exception:
        return jsonify({'message': 'Unexpected server error occurred.'}), 500


# ------------------ ERROR HANDLER ------------------
@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(_: RequestEntityTooLarge):
    return jsonify({
        'message': 'Image file is too large. Maximum size is 6 MB.'
    }), 413


# ------------------ MAIN ------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)