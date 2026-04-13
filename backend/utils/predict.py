import logging
from PIL import Image, UnidentifiedImageError

import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class InvalidImageError(ValueError):
    pass


class CaptionModelError(RuntimeError):
    pass


class PredictionService:
    def __init__(self) -> None:
        try:
            logger.info("Loading lightweight caption model...")

            self.model = VisionEncoderDecoderModel.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning"
            )
            self.processor = ViTImageProcessor.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning"
            )

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

            logger.info("Model loaded successfully")

        except Exception as error:
            logger.exception("Failed to load model")
            raise CaptionModelError("Unable to load caption model") from error

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

            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            output_ids = self.model.generate(pixel_values, max_length=16, num_beams=4)

            caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            return caption

        except Exception as error:
            logger.exception("Prediction failed")
            raise CaptionModelError("Failed to generate caption") from error


# ✅ BLEU FUNCTION (OUTSIDE CLASS)
def compute_bleu(reference: str, prediction: str) -> float:
    reference_tokens = [reference.lower().split()]
    prediction_tokens = prediction.lower().split()

    smoothie = SmoothingFunction().method4

    score = sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smoothie)

    return round(score, 3)