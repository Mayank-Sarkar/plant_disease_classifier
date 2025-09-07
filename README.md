# YOLOv8 Plant Disease Classifier

Classify plant leaf diseases using a trained YOLOv8 model.  
Each run produces annotated images and a `predictions.json` file with top-1 and top-5 predictions (absolute paths included).

## Setup

### Clone the repo
```bash
git clone <your-repo-url>
cd plant-disease-classifier

Create a virtual environment

Linux/macOS:

python3 -m venv venv
source venv/bin/activate

Windows (cmd):

python -m venv venv
venv\Scripts\activate

Install dependencies

CPU Users:

pip install -r requirements.txt

GPU Users (CUDA 12.1 example, one-liner Linux/macOS):

python3 -m venv venv && source venv/bin/activate && \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
pip install ultralytics numpy opencv-python tqdm

Windows (cmd):

python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics numpy opencv-python tqdm

Run predictions

Test on a folder:

python src/predict.py --source sample_images/

Test on a single image:

python src/predict.py --source sample_images/potato_healthy1.JPG

Output

Each run creates a folder dynamically based on the source:

runs/classify/test_<source_name>/
├── predictions.json          # absolute paths, top-1 & top-5 predictions
└── prediction_images/        # all annotated images

    If a folder with the same name exists, a suffix _2, _3, etc., is added automatically.


---
