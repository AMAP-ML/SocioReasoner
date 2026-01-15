import argparse

from dacite import from_dict
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.rlvr.rlvr_socioseg_vlm_pipeline_infer import SocioSegConfig, SocioSegInferPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="config")
    parser.add_argument(
        "--config_name", help="The name of the main configuration file (without extension).", default="sppo_config"
    )
    args = parser.parse_args()

    if GlobalHydra.instance().is_initialized():
        print("Hydra has been initialized. Now clearing it.")
        GlobalHydra.instance().clear()
        
    initialize(config_path=args.config_path, job_name="app")

    cfg = compose(config_name=args.config_name)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ppo_config = from_dict(data_class=SocioSegConfig, data=OmegaConf.to_container(cfg, resolve=True))
    

    init()

    pipeline = SocioSegInferPipeline(pipeline_config=ppo_config)
    
    pipeline.run()


if __name__ == "__main__":
    main()
