import typing as ty
from optim_esm_tools.config import config

_SECONDS_TO_YEAR: int = int(config['constants']['seconds_to_year'])
_FOLDER_FMT: ty.List[str] = config['CMIP_files']['folder_fmt'].split()
_CMIP_HANDLER_VERSION: str = config['versions']['cmip_handler']
_DEFAULT_MAX_TIME: ty.Tuple[int, ...] = tuple(
    int(s) for s in config['analyze']['max_time'].split()
)
