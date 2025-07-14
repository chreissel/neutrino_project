from lightning.pytorch.cli import LightningCLI

class CustomCLI(LightningCLI):
    def before_instantiate_classes(self):
        outdir = self.config['fit.trainer.default_root_dir']
        self.config['fit.trainer.logger.init_args.save_dir'] = outdir
        return

def cli_main():
    cli = CustomCLI(save_config_callback=None)

if __name__ == "__main__":
    cli_main()
