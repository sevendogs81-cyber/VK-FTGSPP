# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import logging

from ftgspp.data.utils.mvdataset import MVDataset


def load_multiview_dataset(
    config,
    *,
    logger: logging.Logger | None = None,
):
    if logger is not None:
        logger.info("Using on-disk MVDataset")
    return MVDataset.from_config(config)
