from lightning.pytorch.cli import LightningCLI
from models import SpectraDataModule, Lightning1DCNNModel

def main():
    cli = LightningCLI(
        model_class=Lightning1DCNNModel,
        datamodule_class=SpectraDataModule,
        save_config_kwargs={"overwrite": True}
    )

if __name__ == '__main__':
    main()