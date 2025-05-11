import logging
from psycopg import AsyncConnection
from psycopg.rows import dict_row, DictRow
import os
from psycopg_pool.pool_async import AsyncConnectionPool
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def getLogger(name: str):
    logger = logging.getLogger(name)
    return logger


_config = None
_pool = None


def _load_config() -> dict:
    possible_path = ["/etc/expert-kit/config.yaml"]
    for path in possible_path:
        if os.path.exists(path):
            config = yaml.safe_load(open(path, "r"))
            return config
    raise FileNotFoundError("can not found config.yaml in /etc/expert-kit/config.yaml")


def load_config_key(key: str) -> str:
    global _config
    if _config is None:
        _config = _load_config()
    val = _config[key]
    return val


async def get_db_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            load_config_key("db_dsn"),
            open=False,
            connection_class=AsyncConnection[DictRow],
            kwargs={"row_factory": dict_row},
        )
        await _pool.open()
    return _pool


ConnType = AsyncConnection[DictRow]
PoolType = AsyncConnectionPool[ConnType]
