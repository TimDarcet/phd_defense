# from https://gist.github.com/uwezi/faec101ed5d7c20222b33eee4b6c7d63

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2  # needs opencv-python https://pypi.org/project/opencv-python/
import moderngl
import numpy as np
from manimlib import ImageMobject, Mobject
from manimlib.constants import DL, DR, UL, UR
from manimlib.utils.bezier import inverse_interpolate
from manimlib.utils.images import get_full_raster_image_path
from manimlib.utils.iterables import listify, resize_with_interpolation
from PIL import Image, ImageOps

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manimlib.typing import Vect3


def change_to_rgba_array(image, dtype: str = "uint8"):
    """Converts an RGB array into RGBA with the alpha value opacity maxed."""
    pa = image
    if len(pa.shape) == 2:
        pa = pa.reshape([*list(pa.shape), 1])
    if pa.shape[2] == 1:
        pa = pa.repeat(3, axis=2)
    if pa.shape[2] == 3:
        alphas = 255 * np.ones(
            [*list(pa.shape[:2]), 1],
            dtype=dtype,
        )
        pa = np.append(pa, alphas, axis=2)
    return pa


@dataclass
class VideoStatus:
    video_object: cv2.VideoCapture
    time: float = 0

    def __deepcopy__(self, memo):
        return self


class PILImageMobject(Mobject):
    shader_folder: str = "image"
    data_dtype: Sequence[tuple[str, type, tuple[int]]] = [  # type:ignore
        ("point", np.float32, (3,)),
        ("im_coords", np.float32, (2,)),
        ("opacity", np.float32, (1,)),
    ]
    render_primitive: int = moderngl.TRIANGLES

    def __init__(self, image: Image.Image, height: float = 4.0, **kwargs):
        self.height = height
        self.image = image
        super().__init__(**kwargs)

    def init_data(self) -> None:  # type:ignore
        super().init_data(length=6)
        self.data["point"][:] = [UL, DL, UR, DR, UR, DL]
        self.data["im_coords"][:] = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 0), (0, 1)]
        self.data["opacity"][:] = self.opacity

    def init_points(self) -> None:
        size = self.image.size
        self.set_width(2 * size[0] / size[1], stretch=True)
        self.set_height(self.height)

    @Mobject.affects_data
    def set_opacity(self, opacity: float, recurse: bool = True):
        self.data["opacity"][:, 0] = resize_with_interpolation(np.array(listify(opacity)), self.get_num_points())
        return self

    def set_color(self, color, opacity=None, recurse=None):
        return self

    def point_to_rgb(self, point: Vect3) -> Vect3:
        x0, y0 = self.get_corner(UL)[:2]
        x1, y1 = self.get_corner(DR)[:2]
        x_alpha = inverse_interpolate(x0, x1, point[0])
        y_alpha = inverse_interpolate(y0, y1, point[1])
        if not (0 <= x_alpha <= 1) and (0 <= y_alpha <= 1):
            raise Exception("Cannot sample color from outside an image")

        pw, ph = self.image.size
        rgb = self.image.getpixel(
            (
                int((pw - 1) * x_alpha),
                int((ph - 1) * y_alpha),
            )
        )[:3]  # type:ignore
        return np.array(rgb) / 255  # type:ignore


class VideoMobject(PILImageMobject):
    """
    Following a discussion on Discord about animated GIF images.
    Modified for videos

    Parameters
    ----------
    filename
        the filename of the video file

    imageops
        (optional) possibility to include a PIL.ImageOps operation, e.g.
        PIL.ImageOps.mirror

    speed
        (optional) speed-up/slow-down the playback

    loop
        (optional) replay the video from the start in an endless loop

    https://discord.com/channels/581738731934056449/1126245755607339250/1126245755607339250
    2023-07-06 Uwe Zimmermann & Abulafia
    2024-03-09 Uwe Zimmermann
    """

    def __init__(self, filename: str, imageops: Callable | None = None, speed=1.0, loop=False, **kwargs):
        self.filename = filename
        self.imageops = imageops
        self.speed = speed
        self.loop = loop
        self._id = id(self)
        self.status = VideoStatus(video_object=cv2.VideoCapture(filename))

        self.status.video_object.set(cv2.CAP_PROP_POS_FRAMES, 1)
        ret, frame = self.status.video_object.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            if imageops is not None:
                img = imageops(img)
        else:
            img = Image.fromarray(
                np.array([[63, 0, 0, 0], [0, 127, 0, 0], [0, 0, 191, 0], [0, 0, 0, 255]], dtype=np.uint8)
            )
        super().__init__(img, **kwargs)
        if ret:
            self.add_updater(self.videoUpdater)

    def videoUpdater(self, mobj, dt):
        if dt == 0:
            return
        status = self.status
        status.time += 1000 * dt * mobj.speed
        self.status.video_object.set(cv2.CAP_PROP_POS_MSEC, status.time)
        ret, frame = self.status.video_object.read()
        if (not ret) and self.loop:
            status.time = 0
            self.status.video_object.set(cv2.CAP_PROP_POS_MSEC, status.time)
            ret, frame = self.status.video_object.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # needed here?
            img = Image.fromarray(frame)

            if mobj.imageops is not None:
                img = mobj.imageops(img)
            mobj.pixel_array = change_to_rgba_array(np.asarray(img), mobj.pixel_array_dtype)
