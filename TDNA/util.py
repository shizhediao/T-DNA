import re
import shutil
import tarfile
from collections import OrderedDict
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Tuple, Union
from zipfile import ZipFile, is_zipfile

import logging
import os
import threading
from logging import CRITICAL  # NOQA
from logging import DEBUG  # NOQA
from logging import ERROR  # NOQA
from logging import FATAL  # NOQA
from logging import INFO  # NOQA
from logging import NOTSET  # NOQA
from logging import WARN  # NOQA
from logging import WARNING  # NOQA
from typing import Optional
from config import glue_processors
from config import glue_output_modes


_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.WARNING


def _get_default_logging_level():
    """
    If TRANSFORMERS_VERBOSITY env var is set to one of the valid choices return that as the new default level.
    If it is not - fall back to ``_default_log_level``
    """
    env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option TRANSFORMERS_VERBOSITY={env_level_str}, "
                f"has to be one of: { ', '.join(log_levels.keys()) }"
            )
    return _default_log_level


def _get_library_name() -> str:

    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:

    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:

    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        library_root_logger.propagate = False


def _reset_library_root_logger() -> None:

    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return logging.getLogger(name)


def get_verbosity() -> int:
    """
    Return the current level for the 🤗 Transformers's root logger as an int.

    Returns:
        :obj:`int`: The logging level.

    .. note::

        🤗 Transformers has following logging levels:

        - 50: ``transformers.logging.CRITICAL`` or ``transformers.logging.FATAL``
        - 40: ``transformers.logging.ERROR``
        - 30: ``transformers.logging.WARNING`` or ``transformers.logging.WARN``
        - 20: ``transformers.logging.INFO``
        - 10: ``transformers.logging.DEBUG``
    """

    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    """
    Set the vebosity level for the 🤗 Transformers's root logger.

    Args:
        verbosity (:obj:`int`):
            Logging level, e.g., one of:

            - ``transformers.logging.CRITICAL`` or ``transformers.logging.FATAL``
            - ``transformers.logging.ERROR``
            - ``transformers.logging.WARNING`` or ``transformers.logging.WARN``
            - ``transformers.logging.INFO``
            - ``transformers.logging.DEBUG``
    """

    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


def set_verbosity_info():
    """Set the verbosity to the :obj:`INFO` level."""
    return set_verbosity(INFO)


def set_verbosity_warning():
    """Set the verbosity to the :obj:`WARNING` level."""
    return set_verbosity(WARNING)


def set_verbosity_debug():
    """Set the verbosity to the :obj:`DEBUG` level."""
    return set_verbosity(DEBUG)


def set_verbosity_error():
    """Set the verbosity to the :obj:`ERROR` level."""
    return set_verbosity(ERROR)


def disable_default_handler() -> None:
    """Disable the default handler of the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)


def enable_default_handler() -> None:
    """Enable the default handler of the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)


def disable_propagation() -> None:
    """Disable propagation of the library log outputs.
    Note that log propagation is disabled by default.
    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = False


def enable_propagation() -> None:
    """Enable propagation of the library log outputs.
    Please disable the HuggingFace Transformers's default handler to prevent double logging if the root logger has
    been configured.
    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = True


def enable_explicit_format() -> None:
    """
    Enable explicit formatting for every HuggingFace Transformers's logger. The explicit formatter is as follows:

    ::

        [LEVELNAME|FILENAME|LINE NUMBER] TIME >> MESSAGE

    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
        handler.setFormatter(formatter)


def reset_format() -> None:
    """
    Resets the formatting for HuggingFace Transformers's loggers.

    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        handler.setFormatter(None)

logger = get_logger(__name__)  # pylint: disable=invalid-name

try:
    from torch.hub import _get_torch_home

    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
    )

default_cache_path = os.path.join(torch_cache_home, "transformers")


PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)

WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
CONFIG_NAME = "config.json"
MODEL_CARD_NAME = "modelcard.json"


MULTIPLE_CHOICE_DUMMY_INPUTS = [
    [[0, 1, 0, 1], [1, 0, 0, 1]]
] * 2  # Needs to have 0s and 1s only since XLM uses it for langs too.
DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]

S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
CLOUDFRONT_DISTRIB_PREFIX = "https://cdn.huggingface.co"
PRESET_MIRROR_DICT = {
    "tuna": "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models",
    "bfsu": "https://mirrors.bfsu.edu.cn/hugging-face-models",
}

import math

import torch
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)

def _gelu_python(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))


ACT2FN = {
    "relu": F.relu,
    "swish": swish,
    "gelu": gelu,
    "tanh": torch.tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "mish": mish,
}


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(activation_string, list(ACT2FN.keys())))


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))



def _get_indent(t):
    """Returns the indentation in the first line of t"""
    search = re.search(r"^(\s*)\S", t)
    return "" if search is None else search.groups()[0]


def _convert_output_args_doc(output_args_doc):
    """Convert output_args_doc to display properly."""
    # Split output_arg_doc in blocks argument/description
    indent = _get_indent(output_args_doc)
    blocks = []
    current_block = ""
    for line in output_args_doc.split("\n"):
        # If the indent is the same as the beginning, the line is the name of new arg.
        if _get_indent(line) == indent:
            if len(current_block) > 0:
                blocks.append(current_block[:-1])
            current_block = f"{line}\n"
        else:
            # Otherwise it's part of the description of the current arg.
            # We need to remove 2 spaces to the indentation.
            current_block += f"{line[2:]}\n"
    blocks.append(current_block[:-1])

    # Format each block for proper rendering
    for i in range(len(blocks)):
        blocks[i] = re.sub(r"^(\s+)(\S+)(\s+)", r"\1- **\2**\3", blocks[i])
        blocks[i] = re.sub(r":\s*\n\s*(\S)", r" -- \1", blocks[i])

    return "\n".join(blocks)


from urllib.parse import urlparse
def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")
try:
    USE_TF = os.environ.get("USE_TF", "AUTO").upper()
    USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
    if USE_TORCH in ("1", "ON", "YES", "AUTO") and USE_TF not in ("1", "ON", "YES"):
        import torch

        _torch_available = True  # pylint: disable=invalid-name
        logger.info("PyTorch version {} available.".format(torch.__version__))
    else:
        logger.info("Disabling PyTorch because USE_TF is set")
        _torch_available = False
except ImportError:
    _torch_available = False  # pylint: disable=invalid-name
def is_torch_available():
    return _torch_available

try:
    USE_TF = os.environ.get("USE_TF", "AUTO").upper()
    USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()

    if USE_TF in ("1", "ON", "YES", "AUTO") and USE_TORCH not in ("1", "ON", "YES"):
        import tensorflow as tf

        assert hasattr(tf, "__version__") and int(tf.__version__[0]) >= 2
        _tf_available = True  # pylint: disable=invalid-name
        logger.info("TensorFlow version {} available.".format(tf.__version__))
    else:
        logger.info("Disabling Tensorflow because USE_TORCH is set")
        _tf_available = False
except (ImportError, AssertionError):
    _tf_available = False  # pylint: disable=invalid-name
def is_tf_available():
    return _tf_available
import sys
from tqdm import tqdm
def http_get(url, temp_file, proxies=None, resume_size=0, user_agent: Union[Dict, str, None] = None):
    ua = "transformers/{}; python/{}".format("3.2.0", sys.version.split()[0])
    if is_torch_available():
        ua += "; torch/{}".format(torch.__version__)
    if is_tf_available():
        ua += "; tensorflow/{}".format(tf.__version__)
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join("{}/{}".format(k, v) for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    headers = {"user-agent": ua}
    if resume_size > 0:
        headers["Range"] = "bytes=%d-" % (resume_size,)
    response = requests.get(url, stream=True, proxies=proxies, headers=headers)
    if response.status_code == 416:  # Range not satisfiable
        return
    content_length = response.headers.get("Content-Length")
    total = resume_size + int(content_length) if content_length is not None else None
    progress = tqdm(
        unit="B",
        unit_scale=True,
        total=total,
        initial=resume_size,
        desc="Downloading",
        disable=False,
    )
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()

def cached_path(
    url_or_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    extract_compressed_file=False,
    force_extract=False,
    local_files_only=False,
) -> Optional[str]:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-dowload the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incompletly recieved file is found.
        user_agent: Optional string or dict that will be appended to the user-agent on remote requests.
        extract_compressed_file: if True and the path point to a zip or tar file, extract the compressed
            file in a folder along the archive.
        force_extract: if True when extract_compressed_file is True and the archive was already extracted,
            re-extract the archive and overide the folder where it was extracted.
    Return:
        None in case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
        Local path (string) otherwise
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        output_path = get_from_cache(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            user_agent=user_agent,
            local_files_only=local_files_only,
        )
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))

    if extract_compressed_file:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
            return output_path

        # Path where we extract compressed archives
        # We avoid '.' in dir name and add "-extracted" at the end: "./model.zip" => "./model-zip-extracted/"
        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
        output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

        if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted) and not force_extract:
            return output_path_extracted

        # Prevent parallel extractions
        lock_path = output_path + ".lock"
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)
            os.makedirs(output_path_extracted)
            if is_zipfile(output_path):
                with ZipFile(output_path, "r") as zip_file:
                    zip_file.extractall(output_path_extracted)
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)
                tar_file.close()
            else:
                raise EnvironmentError("Archive format of {} could not be identified".format(output_path))

        return output_path_extracted

    return output_path

import requests
from hashlib import sha256

def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name
    so that TF 2.0 can identify it as a HDF5 file
    (see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    url_bytes = url.encode("utf-8")
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    if url.endswith(".h5"):
        filename += ".h5"

    return filename
from functools import partial
import fnmatch
import tempfile
def get_from_cache(
    url,
    cache_dir=None,
    force_download=False,
    proxies=None,
    etag_timeout=10,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    """
    Given a URL, look for the corresponding file in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    Return:
        None in case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
        Local path (string) otherwise
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    etag = None
    if not local_files_only:
        try:
            response = requests.head(url, allow_redirects=True, proxies=proxies, timeout=etag_timeout)
            if response.status_code == 200:
                etag = response.headers.get("ETag")
        except (EnvironmentError, requests.exceptions.Timeout):
            # etag is already None
            pass

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # etag is None = we don't have a connection, or url doesn't exist, or is otherwise inaccessible.
    # try to get the last downloaded one
    if etag is None:
        if os.path.exists(cache_path):
            return cache_path
        else:
            matching_files = [
                file
                for file in fnmatch.filter(os.listdir(cache_dir), filename + ".*")
                if not file.endswith(".json") and not file.endswith(".lock")
            ]
            if len(matching_files) > 0:
                return os.path.join(cache_dir, matching_files[-1])
            else:
                # If files cannot be found and local_files_only=True,
                # the models might've been found if local_files_only=False
                # Notify the user about that
                if local_files_only:
                    raise ValueError(
                        "Cannot find the requested files in the cached path and outgoing traffic has been"
                        " disabled. To enable model look-ups and downloads online, set 'local_files_only'"
                        " to False."
                    )
                return None

    # From now on, etag is not None.
    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):

        # If the download just completed while the lock was activated.
        if os.path.exists(cache_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return cache_path

        if resume_download:
            incomplete_path = cache_path + ".incomplete"

            def _resumable_file_manager():
                with open(incomplete_path, "a+b") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, dir=cache_dir, delete=False)
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info("%s not found in cache or force_download set to True, downloading to %s", url, temp_file.name)

            http_get(url, temp_file, proxies=proxies, resume_size=resume_size, user_agent=user_agent)

        logger.info("storing %s in cache at %s", url, cache_path)
        os.replace(temp_file.name, cache_path)

        logger.info("creating metadata file for %s", cache_path)
        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return cache_path



class cached_property(property):

    def __get__(self, obj, objtype=None):
        # See docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        cached = getattr(obj, attr, None)
        if cached is None:
            cached = self.fget(obj)
            setattr(obj, attr, cached)
        return cached


class ModelOutput(OrderedDict):

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have more than one required field."

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())




from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from tokenization import PreTrainedTokenizer
from tokenization import BatchEncoding


InputDataClass = NewType("InputDataClass", Any)

DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])


def default_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


@dataclass
class DataCollatorForLanguageModeling:
    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples_id = [e["input_ids"] for e in examples]   #input_ids is a list of list, each list contains 128 elements
            examples_attention_mask = [e["attention_mask"] for e in examples]  #attention_mask is a list of list, each list contains 128 elements
            examples_token_type_ids = [e["token_type_ids"] for e in examples]  #token_type_ids is a list of list, each list contains 128 elements
            examples_input_Ngram_ids = [e["input_Ngram_ids"] for e in examples]  #input_Ngram_ids is a list of list, each list contains 128 elements
            examples_Ngram_attention_mask = [e["Ngram_attention_mask"] for e in examples]  #Ngram_attention_mask is list of ndarray, each is size 128
            examples_Ngram_token_type_ids = [e["Ngram_token_type_ids"] for e in examples]  #Ngram_token_type_ids is a list of list, each list contains 128 elements
            examples_Ngram_position_matrix = [e["Ngram_position_matrix"] for e in examples]  #Ngram_position_matrix is a list of ndarray, each is (128, 128)

        batch = self._tensorize_batch(examples_id)  #torch.LongTensor [8,128]
        examples_attention_mask = self._tensorize_batch(examples_attention_mask)    #torch.LongTensor [8,128]
        examples_token_type_ids = self._tensorize_batch(examples_token_type_ids)    #torch.LongTensor [8,128]
        examples_input_Ngram_ids = self._tensorize_batch(examples_input_Ngram_ids)  #torch.LongTensor [8,128]
        examples_Ngram_attention_mask = torch.tensor(examples_Ngram_attention_mask)
        examples_Ngram_token_type_ids = self._tensorize_batch(examples_Ngram_token_type_ids)
        examples_Ngram_position_matrix = torch.tensor(examples_Ngram_position_matrix)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs,
                    'attention_mask': examples_attention_mask,
                    'token_type_ids': examples_token_type_ids,
                    'input_Ngram_ids': examples_input_Ngram_ids,
                    'Ngram_attention_mask': examples_Ngram_attention_mask,
                    'Ngram_token_type_ids': examples_Ngram_token_type_ids,
                    'Ngram_position_matrix': examples_Ngram_position_matrix,
                    "labels": labels}
        else:
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)  #128
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples) #True
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer.pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)  #[8, 256]
            probability_matrix.masked_fill_(padding_mask, value=0.0)  #[8, 256]
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


import dataclasses
import json
from dataclasses import dataclass
from typing import List, Optional, Union

@dataclass
class InputExample:
    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    input_Ngram_ids: List[int]
    Ngram_attention_mask: List[int]
    Ngram_token_type_ids: List[int]
    Ngram_position_matrix: List[int]
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


from enum import Enum
from typing import List, Optional, Union

def glue_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    Ngram_dict,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    return _glue_convert_examples_to_features(
        examples, tokenizer, Ngram_dict, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode
    )


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _glue_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    Ngram_dict,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    label_map = {label: i for i, label in enumerate(label_list)}
    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    num_tokens = 0
    num_unk = 0

    features = []
    for i in range(len(examples)):
        if i % 10000 == 0:
            logger.info("Writing example %d of %d" % (i, len(examples)))

        tokens_a = tokenizer.tokenize(examples[i].text_a)  # tokens_a is a python list
        tokens_b = None
        if len(tokens_a) > max_length - 2:
            tokens_a = tokens_a[:(max_length - 2)]

        tokens = ["<s>"] + tokens_a + ["</s>"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens) # list of token id (int)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        input_padding = [tokenizer.pad_token_id] * (max_length - len(input_ids))
        zero_padding = [0] * (max_length - len(input_ids))
        input_ids += input_padding
        input_mask += zero_padding
        segment_ids += zero_padding

        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(segment_ids) == max_length

        num_tokens += len(input_ids)

        num_unk += input_ids.count(3)

        ngram_matches = []
        #  Filter the word segment from 2 to 7 to check whether there is a word
        for p in range(2, 8):
            for q in range(0, len(tokens) - p + 1):
                character_segment = tokens[q:q + p]
                tmp_text = ''.join([tmp_x for tmp_x in character_segment])
                character_segment = tmp_text.replace('Ġ', ' ').strip()
                if character_segment in Ngram_dict.ngram_to_id_dict:
                    ngram_index = Ngram_dict.ngram_to_id_dict[character_segment]
                    ngram_matches.append([ngram_index, q, p, character_segment])

        max_word_in_seq_proportion = Ngram_dict.max_ngram_in_seq
        if len(ngram_matches) > max_word_in_seq_proportion:
            ngram_matches = ngram_matches[:max_word_in_seq_proportion]
        ngram_ids = [ngram[0] for ngram in ngram_matches]
        ngram_positions = [ngram[1] for ngram in ngram_matches]
        ngram_lengths = [ngram[2] for ngram in ngram_matches]
        ngram_tuples = [ngram[3] for ngram in ngram_matches]
        ngram_seg_ids = [0 if position < (len(tokens_a) + 2) else 1 for position in ngram_positions]

        import numpy as np
        ngram_mask_array = np.zeros(Ngram_dict.max_ngram_in_seq, dtype=np.bool)
        ngram_mask_array[:len(ngram_ids)] = 1

        # record the masked positions
        ngram_positions_matrix = np.zeros(shape=(max_length, Ngram_dict.max_ngram_in_seq), dtype=np.int32)
        for j in range(len(ngram_ids)):
            ngram_positions_matrix[ngram_positions[j]:ngram_positions[j] + ngram_lengths[j], j] = 1.0

        # Zero-pad up to the max word in seq length.
        padding = [0] * (Ngram_dict.max_ngram_in_seq - len(ngram_ids))
        ngram_ids += padding
        ngram_lengths += padding
        ngram_seg_ids += padding

        # 'Ngram_tuples': ngram_tuples,
        # 'Ngram_lengths': ngram_lengths,
        inputs = {'input_ids': input_ids,
                  'attention_mask': input_mask,
                  'token_type_ids': segment_ids,
                  'input_Ngram_ids': ngram_ids,
                  'Ngram_attention_mask': ngram_mask_array,
                  'Ngram_token_type_ids': ngram_seg_ids,
                  'Ngram_position_matrix': ngram_positions_matrix,
                  }
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    if num_unk > 0:
        print("there exists {num} [UNK] in input sentence".format(num=num_unk))

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


class OutputMode(Enum):
    classification = "classification"
    regression = "regression"


from torch.utils.data.dataset import Dataset

class LineByLineTextDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, Ngram_dict, file_path: str, block_size: int):
        max_length = 256  #add by @shizhe
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = []
        examples = lines

        for i in range(len(examples)):

            if i % 10000 == 0:
                logger.info("Writing example %d of %d" % (i, len(examples)))

            tokens_a = tokenizer.tokenize(examples[i])  # tokens_a is a python list
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_length - 2:
                tokens_a = tokens_a[:(max_length - 2)]

            # tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            tokens = ["<s>"] + tokens_a + ["</s>"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            input_padding = [tokenizer.pad_token_id] * (max_length - len(input_ids))
            zero_padding = [0] * (max_length - len(input_ids))
            input_ids += input_padding
            input_mask += zero_padding
            segment_ids += zero_padding

            assert len(input_ids) == max_length
            assert len(input_mask) == max_length
            assert len(segment_ids) == max_length

            ngram_matches = []
            #  Filter the word segment from 2 to 7 to check whether there is a word
            for p in range(2, 8):
                for q in range(0, len(tokens) - p + 1):
                    character_segment = tokens[q:q + p]
                    tmp_text = ''.join([tmp_x for tmp_x in character_segment])
                    character_segment = tmp_text.replace('Ġ', ' ').strip()
                    if character_segment in Ngram_dict.ngram_to_id_dict:
                        ngram_index = Ngram_dict.ngram_to_id_dict[character_segment]
                        ngram_matches.append([ngram_index, q, p, character_segment])

            max_word_in_seq_proportion = Ngram_dict.max_ngram_in_seq
            if len(ngram_matches) > max_word_in_seq_proportion:
                ngram_matches = ngram_matches[:max_word_in_seq_proportion]
            ngram_ids = [ngram[0] for ngram in ngram_matches]
            ngram_positions = [ngram[1] for ngram in ngram_matches]
            ngram_lengths = [ngram[2] for ngram in ngram_matches]
            ngram_tuples = [ngram[3] for ngram in ngram_matches]
            ngram_seg_ids = [0 if position < (len(tokens_a) + 2) else 1 for position in ngram_positions]

            import numpy as np
            ngram_mask_array = np.zeros(Ngram_dict.max_ngram_in_seq, dtype=np.bool)
            ngram_mask_array[:len(ngram_ids)] = 1

            # record the masked positions
            ngram_positions_matrix = np.zeros(shape=(max_length, Ngram_dict.max_ngram_in_seq), dtype=np.int32)
            for j in range(len(ngram_ids)):
                ngram_positions_matrix[ngram_positions[j]:ngram_positions[j] + ngram_lengths[j], j] = 1.0

            # Zero-pad up to the max word in seq length.
            padding = [0] * (Ngram_dict.max_ngram_in_seq - len(ngram_ids))
            ngram_ids += padding
            ngram_lengths += padding
            ngram_seg_ids += padding

            # 'Ngram_tuples': ngram_tuples,
            # 'Ngram_lengths': ngram_lengths,
            inputs = {'input_ids': input_ids,
                      'attention_mask': input_mask,
                      'token_type_ids': segment_ids,
                      'input_Ngram_ids': ngram_ids,
                      'Ngram_attention_mask': ngram_mask_array,
                      'Ngram_token_type_ids': ngram_seg_ids,
                      'Ngram_position_matrix': ngram_positions_matrix,
                      }
            self.examples.append(inputs)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


import os
import time
from enum import Enum

import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class GlueDataset(Dataset):

    def __init__(
        self,
        args,
        Ngram_dict,
        tokenizer,
        limit_length = None,
        mode = Split.train,
        cache_dir = None,
    ):
        self.args = args
        self.processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
            ),
        )
        label_list = self.processor.get_labels()
        self.label_list = label_list
        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    Ngram_dict,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list
