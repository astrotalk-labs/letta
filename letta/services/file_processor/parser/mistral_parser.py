import base64

try:
    from mistralai import Mistral, OCRPageObject, OCRResponse, OCRUsageInfo
    _MISTRAL_AVAILABLE = True
except ImportError:
    _MISTRAL_AVAILABLE = False
    Mistral = None  # type: ignore[assignment]
    OCRPageObject = None  # type: ignore[assignment]
    OCRResponse = None  # type: ignore[assignment]
    OCRUsageInfo = None  # type: ignore[assignment]

from letta.log import get_logger
from letta.services.file_processor.file_types import is_simple_text_mime_type
from letta.services.file_processor.parser.base_parser import FileParser
from letta.settings import settings

logger = get_logger(__name__)


class MistralFileParser(FileParser):
    """Mistral-based OCR extraction"""

    def __init__(self, model: str = "mistral-ocr-latest"):
        if not _MISTRAL_AVAILABLE:
            raise ImportError(
                "MistralFileParser requires the 'mistralai' package, which is not installed in this build. "
                "Install it via `pip install mistralai` (or a working source) to enable OCR file processing."
            )
        self.model = model

    # TODO: Make this return something general if we add more file parsers
    async def extract_text(self, content: bytes, mime_type: str) -> OCRResponse:
        """Extract text using Mistral OCR or shortcut for plain text."""
        try:
            logger.info(f"Extracting text using Mistral OCR model: {self.model}")

            # TODO: Kind of hacky...we try to exit early here?
            # TODO: Create our internal file parser representation we return instead of OCRResponse
            if is_simple_text_mime_type(mime_type):
                text = content.decode("utf-8", errors="replace")
                return OCRResponse(
                    model=self.model,
                    pages=[
                        OCRPageObject(
                            index=0,
                            markdown=text,
                            images=[],
                            dimensions=None,
                        )
                    ],
                    usage_info=OCRUsageInfo(pages_processed=1),  # You might need to construct this properly
                    document_annotation=None,
                )

            base64_encoded_content = base64.b64encode(content).decode("utf-8")
            document_url = f"data:{mime_type};base64,{base64_encoded_content}"

            async with Mistral(api_key=settings.mistral_api_key) as mistral:
                ocr_response = await mistral.ocr.process_async(
                    model="mistral-ocr-latest", document={"type": "document_url", "document_url": document_url}, include_image_base64=False
                )

            return ocr_response

        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            raise
