"""
Frame acquisition (Sec. III-B): webcam, video file, or a folder of image
frames (e.g. URFD "-rgb" PNG sequences), resized to a fixed width.
"""

import glob
import os

import cv2

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def resize_to_width(frame, width):
    """Resize keeping aspect ratio. width <= 0 disables resizing."""
    if not width or width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == width:
        return frame
    new_h = int(round(h * width / w))
    interp = cv2.INTER_AREA if width < w else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, new_h), interpolation=interp)


def crop_frame(frame, crop):
    """crop = (x, y, w, h) in source pixels, or None."""
    if not crop:
        return frame
    x, y, w, h = crop
    return frame[y:y + h, x:x + w]


class FrameSource:
    """
    Iterates BGR frames from:
      - an int / digit string  -> webcam index
      - a directory            -> sorted image files inside it
      - anything else          -> passed to cv2.VideoCapture (file / URL)
    """

    def __init__(self, source, width=640, crop=None, default_fps=30.0):
        self.source = source
        self.width = width
        self.crop = crop
        self._cap = None
        self._images = None
        self.fps = default_fps

        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            self._cap = cv2.VideoCapture(int(source))
            self.is_live = True
        elif isinstance(source, str) and os.path.isdir(source):
            files = []
            for ext in IMAGE_EXTS:
                files.extend(glob.glob(os.path.join(source, f"*{ext}")))
            self._images = sorted(files)
            self.is_live = False
        else:
            self._cap = cv2.VideoCapture(source)
            self.is_live = False

        if self._cap is not None:
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 1:
                self.fps = fps

    def is_opened(self):
        if self._images is not None:
            return len(self._images) > 0
        return self._cap is not None and self._cap.isOpened()

    def __len__(self):
        if self._images is not None:
            return len(self._images)
        if self._cap is not None and not self.is_live:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

    def __iter__(self):
        if self._images is not None:
            for path in self._images:
                frame = cv2.imread(path)
                if frame is None:
                    continue
                yield self._prep(frame)
            return

        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield self._prep(frame)

    def _prep(self, frame):
        return resize_to_width(crop_frame(frame, self.crop), self.width)

    def release(self):
        if self._cap is not None:
            self._cap.release()
