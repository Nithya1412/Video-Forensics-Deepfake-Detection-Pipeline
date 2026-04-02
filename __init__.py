# deepfake_detection package
from .models.resnet50_detector import DeepfakeDetector
from .data.dataset import DeepfakeDataset, build_dataloaders
