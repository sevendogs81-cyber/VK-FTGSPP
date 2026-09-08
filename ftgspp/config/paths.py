from pathlib import Path

from ftgspp.config.schema import AppConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _absolute(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_paths(config: AppConfig) -> AppConfig:
    data = config.data
    data.root = _absolute(data.root)
    data.video_path = _absolute(data.video_path)
    data.calibration_path = _absolute(data.calibration_path)
    data.extracted_path = _absolute(data.extracted_path)

    config.init.points_path = _absolute(config.init.points_path)
    config.init.temporal_flow_path = _absolute(config.init.temporal_flow_path)

    if config.train.resume_checkpoint is not None:
        config.train.resume_checkpoint = _absolute(config.train.resume_checkpoint)

    config.run.experiment_root = _absolute(config.run.experiment_root)
    config.run.output_path = _absolute(config.run.output_path)
    config.run.log_dir = _absolute(config.run.log_dir)
    return config
