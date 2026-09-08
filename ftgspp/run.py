import hydra
from omegaconf import DictConfig

from ftgspp.config.register import register_configs


register_configs()


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    runtime = str(config.run.runtime)
    if runtime != "native":
        raise ValueError(
            f"run.runtime={runtime!r} is unsupported; "
            "ftgspp-vulkan only provides the native Vulkan training path "
            "(set run.runtime=native)"
        )
    from ftgspp.native_runtime.runner import NativeRunner

    NativeRunner.from_hydra(config).run()


if __name__ == "__main__":
    main()
