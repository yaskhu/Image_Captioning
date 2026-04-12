from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from utils.predict import CaptionModelError, InvalidImageError, PredictionService

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / 'frontend'
# MODEL_PATH = BASE_DIR / 'model' / 'model.h5'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024

predictor = PredictionService()

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return jsonify({'message': 'No image file found in request.'}), 400

    image_file = request.files['image']

    # ✅ ADD THIS LINE HERE
    expected_caption = request.form.get("expected_caption")

    if image_file.filename == '' or not allowed_file(image_file.filename):
        return jsonify({'message': 'Please upload a valid image file.'}), 400

    try:
        caption = predictor.predict(image_file)

        # ✅ ADD BLEU LOGIC HERE
        bleu_score = None
        if expected_caption:
            from utils.predict import compute_bleu
            bleu_score = compute_bleu(expected_caption, caption)

        return jsonify({
            'prediction': caption,
            'bleu': bleu_score
        }), 200

    except Exception:
        return jsonify({'message': 'Error occurred'}), 500

@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(_: RequestEntityTooLarge):
    return jsonify({'message': 'Image file is too large. Maximum size is 6 MB.'}), 413


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
