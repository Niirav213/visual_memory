import os
import shutil
import urllib.request
import zipfile
from ultralytics import YOLO

def train_model():
    print("=" * 60)
    print("  Visual Memory AI — YOLOv8 Model Training")
    print("=" * 60)
    print()

    # Create target directories
    os.makedirs("data/weights", exist_ok=True)
    os.makedirs("data/datasets", exist_ok=True)

    # 1. Download coco8 dataset zip programmatically
    dataset_url = "https://ultralytics.com/assets/coco8.zip"
    zip_dest = "data/datasets/coco8.zip"
    
    if not os.path.exists("data/datasets/coco8"):
        print("[*] Downloading coco8 dataset zip...")
        try:
            urllib.request.urlretrieve(dataset_url, zip_dest)
            print("[*] Extracting coco8 dataset zip...")
            with zipfile.ZipFile(zip_dest, 'r') as zip_ref:
                zip_ref.extractall("data/datasets")
            # Clean up zip file
            os.remove(zip_dest)
            print("[✓] Dataset downloaded and extracted successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to download/extract dataset: {e}")
            return
    else:
        print("[*] Dataset already exists locally.")

    # 2. Generate a custom YAML config with absolute path
    abs_project_path = os.path.abspath("data/datasets/coco8")
    yaml_content = f"""path: {abs_project_path.replace(chr(92), '/')} # Absolute path using forward slashes
train: images/train
val: images/val

names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
"""
    custom_yaml_path = "data/datasets/custom_coco8.yaml"
    with open(custom_yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[*] Generated custom absolute path YAML config at: {custom_yaml_path}")

    # 3. Load base model
    print("[*] Loading base model (yolov8n.pt)...")
    model = YOLO("yolov8n.pt")

    # 4. Start training on the custom local dataset config
    print("[*] Starting model fine-tuning...")
    try:
        model.train(
            data=custom_yaml_path,
            epochs=5,
            imgsz=640,
            project="data/runs",
            name="custom_train",
            exist_ok=True,
            verbose=True
        )
        print("[*] Training completed successfully!")
        
        # Locate and copy the best weights file
        best_weights_src = "data/runs/custom_train/weights/best.pt"
        best_weights_dst = "data/weights/custom_best.pt"
        
        # Check global user home directory as fallback on Windows
        if not os.path.exists(best_weights_src):
            home_src = os.path.expanduser("~/runs/detect/data/runs/custom_train/weights/best.pt")
            if os.path.exists(home_src):
                best_weights_src = home_src
        
        if os.path.exists(best_weights_src):
            shutil.copy(best_weights_src, best_weights_dst)
            print(f"[✓] Copying trained model weights to: {best_weights_dst}")
            print()
            print("Model is ready for use! You can now select it in the dashboard.")
        else:
            print(f"[ERROR] Could not find best weights at {best_weights_src}")
            
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        raise e

if __name__ == "__main__":
    train_model()
