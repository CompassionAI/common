tibetan_digits = [
    "\u0F20",
    "\u0F21",
    "\u0F22",
    "\u0F23",
    "\u0F24",
    "\u0F25",
    "\u0F26",
    "\u0F27",
    "\u0F28",
    "\u0F29"]
tibetan_halves = [
    "\u0F2A",
    "\u0F2B",
    "\u0F2C",
    "\u0F2D",
    "\u0F2E",
    "\u0F2F",
    "\u0F30",
    "\u0F31",
    "\u0F32",
    "\u0F33"]


def translate_tibetan_number(tibetan_number):
    """Translate Tibetan numericals to Western numerals.

    Args:
        tibetan_number (str): Unicode Tibetan numeral. Cannot contain half-digits.

    Returns:
        int: Integer represented by the input."""
    res = ""
    for digit in tibetan_number:
        try:
            res += str(tibetan_digits.index(digit))
        except ValueError as e:
            raise ValueError(f"Supposed to be Tibetan number consisting of whole digits but contains {digit}.") from e
    return int(res)
