MuseMate
========

Identify artworks from images and explore rich context around them. MuseMate lets users upload an image of an artwork; a TensorFlow model predicts the artwork title and returns contextual insights such as artist, style, an artist timeline, and recommendations for similar artworks and artists. A modern React frontend pairs with a Flask API and optional Supabase Edge Function.

Overview
--------
- Frontend: Vite + React + TypeScript + Tailwind + shadcn/ui (port 8080)
- Backend: Flask API serving a TensorFlow ResNet50V2 transfer-learning model (port 5000)
- Data: CSV metadata and images served as Flask static files
- Optional: Supabase Edge Function to analyze artwork via an AI gateway

Repository Layout
-----------------
- `MuseMate_Frontend/`: Vite React app (UI, routes, components)
- `MuseMate_Backend/`: Flask API, model training, evaluation, and data utilities
	- `src/app.py`: API server (`/api/predict`)
	- `src/model.py`: Transfer learning training pipeline (ResNet50V2)
	- `src/evaluate.py`: Evaluation on held-out test set
	- `src/merge_data.py`: Creates a master CSV joining training and metadata
	- `data/`: CSVs and images (`images/train/train/...`, `images/test/...`)
	- `artifacts_transfer/`: Trained model `transfer_model.keras` and `title_encoder.pkl`
- `MuseMate_Frontend/supabase/functions/recognize-artwork/`: Optional AI analysis function

How It Works
------------
1. User uploads an image in the frontend.
2. Frontend sends the image to the Flask API (`/api/predict`).
3. API preprocesses the image, runs the TensorFlow model, decodes the predicted class into an artwork title using the saved label encoder, and looks up contextual metadata in `master_artwork_info.csv`.
4. API returns:
	 - `predicted_title`, `confidence`, `artist`, `style`,
	 - `image_url` (served by Flask from `MuseMate_Backend/data/images/train/train/...`),
	 - `artist_timeline` (sorted by date for that artist),
	 - `similar_artworks` and `similar_artists` (sampled by style).

Quick Start (Windows, cmd.exe)
------------------------------

Backend (Flask API)
-------------------
Prerequisites: Python 3.10+ with pip, CUDA optional but recommended for TensorFlow GPU.

1) Create and activate a virtual environment, then install dependencies:

```
cd MuseMate_Backend
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install flask flask-cors tensorflow pandas numpy scikit-learn pillow matplotlib
```

2) Ensure artifacts and data are present:
- Model file: `MuseMate_Backend/artifacts_transfer/transfer_model.keras`
- Encoder: `MuseMate_Backend/artifacts_transfer/title_encoder.pkl`
- CSV: `MuseMate_Backend/data/CSV Files/master_artwork_info.csv`
- Images: `MuseMate_Backend/data/images/train/train/<filename>.jpg`

3) Run the API:

```
python src\app.py
```

The API listens on `http://localhost:5000`.

Frontend (Vite React)
---------------------
Prerequisites: Node.js 18+ and npm.

1) Install dependencies and run the dev server:

```
cd MuseMate_Frontend
npm install
npm run dev
```

The app listens on `http://localhost:8080`.

Testing the API
---------------
From the repository root (ensure the Flask server is running):

```
curl -X POST http://localhost:5000/api/predict -F "file=@MuseMate_Backend\data\images\train\train\68.jpg"
```

Expected JSON (fields may vary):

```
{
	"predicted_title": "The Starry Night",
	"confidence": 0.94,
	"artist": "Vincent van Gogh",
	"style": "Post-Impressionism",
	"image_url": "http://localhost:5000/train/train/68.jpg",
	"artist_timeline": [{ "title": "Sunflowers", "date": "1888", "image_url": "..." }],
	"similar_artworks": [{ "title": "Café Terrace at Night", "artist": "Vincent van Gogh", "image_url": "..." }],
	"similar_artists": [{ "artist_name": "Paul Gauguin", "image_url": "..." }]
}
```

Architecture
------------
- Backend: Flask + TensorFlow (ResNet50V2 transfer learning), Pandas for CSV lookups.
- Static images: Served by Flask via `app = Flask(__name__, static_folder='../data/images')`, exposed through `url_for('static', filename='train/train/<filename>')`.
- Frontend: Vite React app with shadcn/ui, Radix primitives, Tailwind styling.
- Optional AI path: Supabase Edge Function (`recognize-artwork`) that calls an AI gateway (requires `LOVABLE_API_KEY`). Current UI calls the Flask API directly in `src/pages/Recognize.tsx`.

Model Training & Evaluation
---------------------------
Training (transfer learning) is defined in `MuseMate_Backend/src/model.py` and will:
- Read `data/CSV Files/master_artwork_info.csv` and image files.
- Filter missing/corrupted images and rare classes.
- Train a ResNet50V2-based classifier in two phases (warmup + fine-tuning).
- Save artifacts to `artifacts_transfer/transfer_model.keras` and `artifacts_transfer/title_encoder.pkl`.

Run training (requires a proper dataset and adequate compute):

```
cd MuseMate_Backend
.venv\Scripts\activate
python src\model.py
```

Evaluate on test set (expects `data/CSV Files/all_data_info.csv` and images in `data/images/test/`):

```
cd MuseMate_Backend
.venv\Scripts\activate
python src\evaluate.py
```

Data & CSVs
-----------
- `data/CSV Files/train_info.csv` and `all_data_info.csv` provide artwork metadata.
- `src/merge_data.py` demonstrates how to generate `master_artwork_info.csv` by joining those CSVs.
- The API uses `master_artwork_info.csv` for metadata lookups by `title` and `artist`.

Environment Variables
---------------------
Frontend (optional Supabase integration): define in `MuseMate_Frontend/.env`:
- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_PUBLISHABLE_KEY`

Supabase Edge Function (`recognize-artwork`):
- `LOVABLE_API_KEY` (Supabase project environment variable)

API Endpoints
-------------
- `GET /`: Health message
- `POST /api/predict`: multipart/form-data with `file` image. Returns prediction JSON and contextual data.

Notes & Limitations
-------------------
- The classifier predicts among titles it was trained on. Unknown artworks may not be recognized or may map to the closest known title.
- Ensure the model artifacts and CSVs exist; otherwise the API will return a 500 with `Server artifacts not loaded`.
- Image URLs in responses assume images exist at `MuseMate_Backend/data/images/train/train/`.

Troubleshooting
---------------
- Import/TF errors: verify your Python version and that all pip packages installed successfully. CPU-only TensorFlow is slower; GPU requires compatible CUDA/cuDNN.
- 500 errors from API: confirm `artifacts_transfer/transfer_model.keras`, `artifacts_transfer/title_encoder.pkl`, and `data/CSV Files/master_artwork_info.csv` are present.
- CORS: Flask enables CORS via `flask-cors`. Frontend runs on port 8080; backend on 5000.

Scripts & Dev Tips
------------------
- Frontend: `npm run dev`, `npm run build`, `npm run preview` in `MuseMate_Frontend/`
- Backend: launch `python src\app.py` in `MuseMate_Backend/`
- Check an artwork filename in dataset: `python src\check.py`

