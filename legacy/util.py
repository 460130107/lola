"""
:Authors: - Wilker Aziz
"""
import re


def re_collect_groups(re_match, groups=[], repl=''):
    """
    Collect groups for re.sub
    :param re_match: re match object
    :param groups: a list used to return the groups matched
    :param repl: replacement string
    :return: repl string
    """
    groups.extend(re_match.groups())
    return repl


def re_sub(pattern, repl, string):
    """
    Wraps a call to re.sub in order to return both the resulting string and the matched groups.
    :param pattern:
    :param repl: a replacement string
    :param string:
    :return: the resulting string, matched groups
    """
    groups = []
    result = re.sub(pattern, lambda m: re_collect_groups(m, groups, repl), string)
    return result, groups


def re_key_value(key: str, string: str, separator='=', optional=True, default=None, dtype=None):
    """
    Matches a key-value pair for a given key and separator removing it from the input string.
    :param key: key to be matched
    :param string: string containing the key-value pair
    :param separator: separator between key and value
    :param optional: whether the key is optional
    :param default: a default value for optional keys
    :param dtype: if not None, convert to a data type, otherwise simply evaluate it literally with python's eval
    :return: remaining string, value (literal or converted)
    """
    result, groups = re_sub(r'{0}{1}([^ ]+)'.format(key, separator), '', string)
    if not optional and not groups:
        raise ValueError('Expected a key-value pair of the kind {0}{1}<value>: {2}'.format(key, separator, string))
    if groups:
        if dtype is None:
            try:
                value = eval(groups[0])
                return result, value
            except:
                raise ValueError('Cannot evaluate %s' % (groups[0]))
        else:
            try:
                value = dtype(groups[0])
                return result, value
            except:
                raise ValueError('Cannot convert %s to %s' % (groups[0], dtype))
    else:
        return result, default
