
from Core_modules.vision_models import VisionModels

models = VisionModels()

print("Classification model:", models.classifier)
print("Detection model:", models.detector)
print("Hand model:", models.hand_landmarker)

models.close()
