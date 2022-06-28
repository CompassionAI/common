import os
import glob
import json


def get_local_ckpt(model_name, model_dir=False):
    """Convert the name of the model in the CAI data registry to a local checkpoint path.

    Args:
        model_name (:obj:`string`):
            The model name in the CAI data registry. If it starts with 'model_archive', it is assumed to be a path
            within the data registry. Otherwise, it is assumed to be a champion model inside the champion_models
            directory.

            If model name has no extension, and model_dir is False, it checks if there is only one .bin file in the path
            the model name resolves to. If there is more than one, it crashes, otherwise it returns the path to this
            .bin file.
        model_dir (:obj:`bool`): Return the model directory, not a candidate bin file. Useful for Hugging Face local
            loading using from_pretrained.

    Returns:
        The local directory name you can feed to AutoModel.from_pretrained.
    """

    data_base_path = os.environ['CAI_DATA_BASE_PATH']
    if not model_name.startswith('model_archive'):
        model_name = os.path.join('champion_models', model_name)
    model_name = os.path.join(data_base_path, model_name)
    if model_dir:
        return model_name
    if not '.' in model_name:
        candidates = glob.glob(os.path.join(model_name, "*.bin"))
        if len(candidates) == 0:
            raise FileNotFoundError(f"No .bin files found in {model_name}")
        if len(candidates) > 1:
            raise FileExistsError(f"Multiple .bin files in {model_name}, please specify which one to load by appending"
                                   "the .bin filename to the model name")
        model_name = candidates[0]
    return model_name


def get_cai_config(model_name):
    """Load the CompassionAI config for the name of the model in the CAI data registry.

    Args:
        model_name (:obj:`string`):
            The model name in the CAI data registry. Follows the same rules as get_local_ckpt.

    Returns:
        The loaded CompassionAI config JSON.
    """

    local_ckpt = get_local_ckpt(model_name)
    cfg_fn = os.path.splitext(local_ckpt)[0] + ".config_cai.json"
    with open(cfg_fn) as f:
        return json.load(f)
