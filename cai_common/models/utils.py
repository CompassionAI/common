import os
import glob
import json


def get_local_ckpt(model_name, model_dir=False, search_for_ext="bin"):
    """Convert the name of the model in the CAI data registry to a local checkpoint path.

    Args:
        model_name (:obj:`string`):
            The model name in the CAI data registry. If it starts with 'model_archive', it is assumed to be a path
            within the data registry. Otherwise, it is assumed to be a champion model inside the champion_models
            directory.

            If model name has no extension, and model_dir is False, it checks if there is only one file with the
            extension specified in search_for_ext in the path the model name resolves to. If there is more than one, it
            crashes, otherwise it returns the path to the unique file with the requested extension.
        model_dir (:obj:`bool`, `optional`): Return the model directory, not a candidate file. Useful for Hugging Face
            local loading using from_pretrained.
        search_for_ext (:obj:`string`, `optional`): What extension to search for. Defaults to 'bin'.

    Returns:
        The local directory name you can feed to AutoModel.from_pretrained.
    """

    data_base_path = os.environ['CAI_DATA_BASE_PATH']
    if not '/' in model_name:
        model_name = os.path.join('champion_models', model_name)
    model_name = os.path.join(data_base_path, model_name)
    if model_dir:
        return model_name
    if not '.' in model_name:
        candidates = glob.glob(os.path.join(model_name, "*." + search_for_ext))
        if len(candidates) == 0:
            raise FileNotFoundError(f"No .{search_for_ext} files found in {model_name}")
        if len(candidates) > 1:
            raise FileExistsError(f"Multiple .{search_for_ext} files in {model_name}, please specify which one to load "
                                   "by appending the .{search_for_ext} filename to the model name")
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

    cfg_fn = get_local_ckpt(model_name, search_for_ext="config_cai.json")
    if not cfg_fn[-len(".config_cai.json"):] == ".config_cai.json":
        cfg_fn = cfg_fn.split('.')[0] + ".config_cai.json"      # Deliberately take the first dot
    with open(cfg_fn) as f:
        return json.load(f)
