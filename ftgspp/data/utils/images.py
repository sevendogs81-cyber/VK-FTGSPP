# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

FRAME_INDEX_WIDTH = 6
CAMERA_INDEX_WIDTH = 3
SUFFIX = ".webp"


def to_image_name(frame: int | str, camera: int | str, suffix: str = SUFFIX) -> str:
    if isinstance(frame, int):
        frame = f"{frame:0{FRAME_INDEX_WIDTH}d}"
    if isinstance(camera, int):
        camera = f"{camera:0{CAMERA_INDEX_WIDTH}d}"

    return f"c{camera}-f{frame}{suffix}"
