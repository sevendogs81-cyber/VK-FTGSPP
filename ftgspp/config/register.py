from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from ftgspp.config.schema import AppConfig


def register_resolvers() -> None:
    if not OmegaConf.has_resolver("ftgspp_min"):
        OmegaConf.register_new_resolver(
            "ftgspp_min",
            lambda left, right: min(int(left), int(right)),
        )


def register_configs() -> None:
    register_resolvers()
    ConfigStore.instance().store(
        group="schema",
        name="config",
        node=AppConfig,
        package="_global_",
    )
