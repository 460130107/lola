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


def re_key_value(key, string, separator='=', repl='', optional=True):
    """
    Matches a key-value pair and replaces it by a given string.
    :param key:
    :param string:
    :param separator: separator of the key-value pair
    :param optional: if False, raises an exception in case no matches are found
    :return: resulting string, value (or None)
    """
    result, groups = re_sub(r'{0}{1}([^ ]+)'.format(key, separator), '', string)
    if not optional and not groups:
        raise ValueError('Expected a key-value pair of the kind {0}{1}<value>: {2}'.format(key, separator, string))
    if groups:
        return result, groups[0]
    else:
        return result, None
