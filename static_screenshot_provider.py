"""Static screenshot provider for testing.

Provides a pre-captured screenshot image instead of taking a live screenshot.
Used by the test API to find patterns in provided screenshots.
"""

from PIL import Image

from qontinui.find.screenshot import ScreenshotProvider
from qontinui.model.element import Region


class StaticScreenshotProvider(ScreenshotProvider):
    """Screenshot provider that returns a static pre-loaded image.

    This is used for testing pattern matching against provided screenshots
    rather than capturing from the actual screen.
    """

    def __init__(self, image: Image.Image) -> None:
        """Initialize with a static screenshot image.

        Args:
            image: PIL Image to return when capture() is called
        """
        self._image = image

    def capture(self, region: Region | None = None) -> Image.Image:
        """Return the static screenshot (optionally cropped to region).

        Args:
            region: Optional region to crop to. If None, returns full image.

        Returns:
            PIL Image of the screenshot (full or cropped).
        """
        if region is None:
            return self._image

        # Crop to the specified region
        return self._image.crop((
            region.x,
            region.y,
            region.x + region.width,
            region.y + region.height
        ))
