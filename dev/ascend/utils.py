def long_banner(msg: str) -> None:
    banner(msg, width=50)


def short_banner(msg: str) -> None:
    banner(msg, width=40)


def banner(msg: str, width: int) -> None:
    print("=" * width + msg + "=" * width)
