# AI Image Captioning

This repository provides a professional image captioning demo with a responsive frontend and a Flask backend. The backend loads a Keras model from `backend/model/model.h5` and processes image uploads to produce captions.

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Run the backend.

```powershell
python backend\app.py
```

4. Open the frontend.

Open `frontend/index.html` in a browser, or navigate to `http://127.0.0.1:5000/` if the backend is serving the static files.

## Notes

- The current backend includes a placeholder decoding path for model output. Replace the `_decode_prediction` implementation with your tokenizer and beam search logic for a fully operational captioning model.
- The frontend uses a friendly, accessible interface with preview support and error handling.

