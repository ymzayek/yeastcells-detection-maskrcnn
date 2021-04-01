import warnings
import importlib
import subprocess
tested_cuda_torch_versions = {('101', '1.8')}


def pip_install(*args):
  args = ['pip3', 'install'] + list(args)
  result = subprocess.run(args, capture_output=True)
  assert result.returncode == 0, (
    f"pip install failed with error code {result.returncode}: {' '.join(args)}\n"
    f"stdout:\n{result.stdout.decode('utf-8')}\n\n"
    f"stderr:\n{result.stderr.decode('utf-8')}"
  )


def check_colab():
  try:
    from google import colab
  except ImportError:
    warnings.warn(
        f"Setting up detectron2 was tested on Google Colab, "
        f"if this fails, please follow the instructions to install torch")


def check_torch():
  try:
    import torch, torchvision
  except ImportError as error:
    raise ModuleNotFoundError(
      f"Could not import {error.name}, please make sure toch and torchvision"
      "are installed, match the intended detectron2 version:\n"
      "https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md\n"
      "and match the installed cuda version:\n:https://pytorch.org/"
    )
    
  if not torch.cuda.is_available():
    warning.warn(
        "Torch could not detect a CUDA device, make sure your machine has "
        "a GPU and that CUDA is installed properly. If you intend to use a "
        "CPU, please realise detectron2 will be significantly slower, and "
        "make sure to use 'cpu' rather than 'cuda:0' as a device")
  for device in range(torch.cuda.device_count()):
    print(f"Found a CUDA device: {torch.cuda.get_device_name(0)}")


def install_detectron2():
  check_torch()
  try:
    import detectron2
  except ImportError as error:
    if error.name != 'detectron2':
        warning.warn(f"It seems detectron2 is improperly installed")
        raise error
    
    import torch
    torch_version, cuda_version = torch.__version__.split('+cu')
    torch_version = '.'.join(torch_version.split('.')[:2])
    print(f"Detected torch {torch_version} and CUDA {cuda_version}")
    if (cuda_version, torch_version) not in tested_cuda_torch_versions:
      warnings.warn(
          f"Automatic installation of detectron2 was not tested for torch "
          f"version {torch_version} and CUDA version {cuda_version}")

    # pip_install('-U', 'pyyaml') # solved in other setup.py with install_requires
    pip_install('detectron2', '-f',
                f"https://dl.fbaipublicfiles.com/detectron2/wheels/"
                f"cu{cuda_version}/torch{torch_version}/index.html")
    import detectron2


def reload_sklearn_classes():
    import sklearn.utils.fixes, sklearn.base
    for mod in [sklearn.utils.fixes, sklearn.base]:
      importlib.reload(mod)


def setup_colab():
  check_colab()
  install_detectron2()
  reload_sklearn_classes()
