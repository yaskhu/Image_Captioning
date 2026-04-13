import logging
from PIL import Image, UnidentifiedImageError

from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# BLEU imports
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# ------------------ CUSTOM EXCEPTIONS ------------------

class InvalidImageError(ValueError):
    pass


class CaptionModelError(RuntimeError):
    pass


# ------------------ PREDICTION SERVICE ------------------

class PredictionService:
    def __init__(self) -> None:
        try:
            logger.info("Loading BLIP model...")

            self.processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )

            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

            logger.info("BLIP model loaded successfully")

        except Exception as error:
            logger.exception("Failed to load BLIP model")
            raise CaptionModelError("Unable to load pretrained caption model") from error

    def _load_image(self, image_file) -> Image.Image:
        try:
            if hasattr(image_file, 'stream'):
                image = Image.open(image_file.stream)
            else:
                image = Image.open(image_file)

            return image.convert("RGB")

        except (UnidentifiedImageError, OSError) as error:
            raise InvalidImageError("Invalid image file") from error

    def predict(self, image_file) -> str:
        try:
            image = self._load_image(image_file)

            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            output = self.model.generate(**inputs)

            caption = self.processor.decode(output[0], skip_special_tokens=True)

            return caption

        except Exception as error:
            logger.exception("Prediction failed")
            raise CaptionModelError("Failed to generate caption") from error


# ------------------ BLEU SCORE FUNCTION ------------------

def compute_bleu(reference: str, prediction: str) -> float:
    """
    Compute BLEU score between expected caption and predicted caption
    """

    reference_tokens = [reference.lower().split()]
    prediction_tokens = prediction.lower().split()

    smoothie = SmoothingFunction().method4

    score = sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smoothie)

    return round(score, 3)