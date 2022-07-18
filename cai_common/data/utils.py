def repl_split_commas_or_kill(match, take_second_item=False, test_mode=False):
    # Take a regex match of the form (...,...), or some other brackets, and return one of the items.
    #
    # Test mode returns the bracketed input and the result, for unit testing.
    match = match.group(0)
    if ',' in match:
        if take_second_item:
            if test_mode:
                return f"@{match}|{match.split(',')[1][:-1]}@"
            else:
                return match.split(',')[1][:-1]
        return match.split(',')[0][1:]
    # This is a different comma in unicode but it looks the same in most fonts
    if '‚' in match:
        if take_second_item:
            if test_mode:
                return f"@{match}|{match.split('‚')[1][:-1]}@"
            else:
                return match.split('‚')[1][:-1]
        return match.split('‚')[0][1:]
    return ''
