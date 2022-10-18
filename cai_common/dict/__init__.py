from .numbers import tibetan_digits, tibetan_halves, translate_tibetan_number

try:
    from .dict import TibetanDict, TibetanEncoding
except ModuleNotFoundError:
    # For inference installations we don't currently support the data pulling utilities
    pass

__all__ = [
    "tibetan_digits",
    "tibetan_halves",
    "translate_tibetan_number",
    "TibetanEncoding",
    "TibetanDict"]
