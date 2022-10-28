import os
import glob
import json
import yaml
import logging

from urllib.error import HTTPError
from tqdm.auto import tqdm
from torch.hub import download_url_to_file, get_dir
from cai_common.defaults import cai_s3


logger = logging.getLogger(__name__)

os_data_base_path = os.environ.get('CAI_DATA_BASE_PATH', None)
if os_data_base_path in {'', '***unset***'}:
    os_data_base_path = None


def _download_file(s3_loc, target_fn):
    try:
        download_url_to_file(s3_loc, target_fn)
    except HTTPError as e:
        logger.error(f"Failed to download {s3_loc} to {target_fn}")
        raise e


def get_local_model_dir(model_name, download_if_missing=True):
    """Load a model from the CAI S3 data registry.
    
    Args:
        model_name (:obj:`string`):
            The model name in the CAI data registry. If it starts with 'model_archive', it is assumed to be a path
            within the data registry. Otherwise, it is assumed to be a champion model inside the champion_models
            directory.
        download_if_missing (:obj:`bool`, `optional`): Download from the CompassionAI S3 repository if missing the local
            repository. This is expected in inference installations. Defaults to True.

    Returns:
        The local directory name you can feed to AutoModel.from_pretrained.
    """
    if model_name.startswith("https://"):
        s3_model_loc = model_name
        model_name = model_name[len(cai_s3):]
        model_dir = os.path.join(get_dir(), model_name)
    elif os.path.exists(model_name):
        model_dir = model_name
        s3_model_loc = None
    else:
        data_base_path = os_data_base_path if os_data_base_path is not None else cai_s3
        if data_base_path.startswith("https://"):
            model_dir = os.path.join(get_dir(), model_name)
            s3_model_loc = os.path.join(data_base_path, model_name)
        else:
            model_dir = os.path.join(data_base_path, model_name)
            s3_model_loc = None

    if s3_model_loc is not None:
        if not s3_model_loc.startswith(cai_s3):
            raise ValueError("Only downloads from the public CAI data registry are allowed")

        os.makedirs(model_dir, exist_ok=True)
        
        manifest_fn = os.path.join(model_dir, "manifest.yaml")
        if not os.path.exists(manifest_fn):
            if not download_if_missing:
                raise FileNotFoundError("Model not found locally and download if missing is set to False.")
            _download_file(f"{s3_model_loc}/manifest.yaml", manifest_fn)
        with open(manifest_fn, 'r') as manifest_f:
            manifest = yaml.safe_load(manifest_f)

        to_download = [
            fn for fn, f_loc in [(fn, os.path.join(model_dir, fn)) for fn in manifest['inference']]
                if not os.path.exists(f_loc)
        ]
        if len(to_download) > 0:
            if not download_if_missing:
                raise FileNotFoundError("Model not found locally and download if missing is set to False.")
            for model_file in tqdm(to_download, desc="Downloading model files"):
                _download_file(f"{s3_model_loc}/{model_file}", os.path.join(model_dir, model_file))

    return model_dir


def get_local_file(file_subpath):
    """Load a file from the CAI S3 data registry. Will download the file if needed.
    
    Args:
        file_subpath (:obj:`string`):
            The subpath of the file within the CAI data registry.

    Returns:
        The local file name.
    """
    if os.path.exists(file_subpath):
        return file_subpath
    file_dir = get_local_model_dir(os.path.dirname(file_subpath))
    return os.path.join(file_dir, os.path.basename(file_subpath))


def get_local_ckpt(model_name, model_dir=False, search_for_ext="bin", download_if_missing=True):
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
        download_if_missing (:obj:`bool`, `optional`): Download from the CompassionAI S3 repository if missing the local
            repository. This is expected in inference installations. Defaults to True.

    Returns:
        The local directory name you can feed to AutoModel.from_pretrained.
    """

    if os_data_base_path is None and not download_if_missing:
        raise FileNotFoundError("CAI data registry path not set and downloading is switched off")
    if not model_name.startswith('experiments'):
        model_name = os.path.join('champion_models', model_name)

    model_name = get_local_model_dir(model_name)

    if model_dir:
        return model_name
    if os.path.splitext(model_name)[1] == '':
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
