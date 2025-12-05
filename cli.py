import os
import sys
import shutil
from lightning.pytorch.cli import LightningCLI

class CustomCLI(LightningCLI):
    def before_instantiate_classes(self):
        config_file_path = None

        for i, arg in enumerate(sys.argv):
            if arg == '--config' and i + 1 < len(sys.argv):
                config_file_path = sys.argv[i + 1]
                break
    
        print(config_file_path)
        outdir = self.config['fit.trainer.default_root_dir']
        os.makedirs(outdir, exist_ok=True)
        self.config['fit.trainer.logger.init_args.save_dir'] = outdir
        destination_path = os.path.join(outdir, 'config.yaml')
        shutil.copy(config_file_path, destination_path)

def cli_main():
    cli = CustomCLI(save_config_callback=None)

if __name__ == "__main__":
    cli_main()
