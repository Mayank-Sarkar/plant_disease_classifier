# src/predict.py
import os
import json
import argparse
from ultralytics import YOLO
import shutil

def get_unique_folder(base_folder):
    """Return a unique folder path by appending _2, _3, etc. if needed."""
    if not os.path.exists(base_folder):
        return base_folder
    counter = 2
    while True:
        new_folder = f"{base_folder}_{counter}"
        if not os.path.exists(new_folder):
            return new_folder
        counter += 1

def main(args):
    # Load the trained model
    model_path = os.path.join("models", "yolov8n-cls.pt")
    model = YOLO(model_path)

    # Determine base name for folder
    source_name = os.path.basename(os.path.normpath(args.source))
    run_folder_base = os.path.join("runs", "classify", f"test_{source_name}")
    run_folder = get_unique_folder(run_folder_base)
    os.makedirs(run_folder, exist_ok=True)

    # Run prediction and save annotated images temporarily in this folder
    results = model.predict(source=args.source, save=True, save_dir=run_folder)

    # Create a subfolder for annotated images
    images_folder = os.path.join(run_folder, "prediction_images")
    os.makedirs(images_folder, exist_ok=True)

    # Move all annotated images into prediction_images/
    for r in results:
        src_file = r.path
        dst_file = os.path.join(images_folder, os.path.basename(src_file))
        shutil.move(src_file, dst_file)
        # update path in results to absolute path
        r.path = os.path.abspath(dst_file)

    # Prepare JSON output
    predictions = []
    for r in results:
        top1_index = int(r.probs.top1)
        top1_class = r.names[top1_index]
        top1_conf = float(r.probs.top1conf.item())

        # full top-5 predictions
        top5_list = []
        for i, prob in enumerate(r.probs.top5):
            top5_list.append({
                "class": r.names[int(prob)],
                "confidence": float(r.probs.data[int(prob)])
            })

        image_preds = {
            "image": r.path,  # absolute path now
            "top1_prediction": {
                "class": top1_class,
                "confidence": top1_conf
            },
            "top5_predictions": top5_list
        }
        predictions.append(image_preds)

    # Save JSON inside the run folder
    json_path = os.path.join(run_folder, "predictions.json")
    with open(json_path, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"✅ Predictions JSON saved at: {json_path}")
    print(f"✅ Annotated images saved in: {images_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Plant Disease Classifier")
    parser.add_argument(
        "--source", type=str, required=True,
        help="Path to an image or folder of images. Example: sample_images/test_leaf.jpg"
    )
    args = parser.parse_args()
    main(args)
