import argparse
import asyncio

from splitter.plan import SplitPlan
from utils.dao import ModelDAO
from utils.storage import setup_storage, with_storage_args
from utils.base import get_db_pool, getLogger

logger = getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Split DeepSeek-R1 model weights")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )
    parser.add_argument("--split_plan", type=str, default="splitting_plan.json")
    with_storage_args(parser)

    args = parser.parse_args()
    return args


async def main():
    args = parse_args()
    storage = setup_storage(args)
    pool = await get_db_pool()
    async with pool.connection() as conn:
        modelQ = ModelDAO(conn)
        res = await modelQ.by_name(args.model_name)
        if res is not None and len(res) > 0:
            model = res
            logger.info(f"model found: {model}")
            return
        logger.info(f"model not found: {args.model_name}, creating a new one")
        plan = SplitPlan.load(storage=storage)
        logger.info(f"plan loaded, total chunk {plan.output_count()}")
        model_config = {"storage": {"type": storage.type(), "config": storage.config()}}
        model_name = args.model_name
        await modelQ.upsert(name=model_name, config=model_config)


if __name__ == "__main__":
    asyncio.run(main())
