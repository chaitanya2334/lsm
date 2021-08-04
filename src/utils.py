import functools
import io
import os
import pickle
import warnings
import zipfile
from string import digits
import numpy as np
import pandas as pd

import requests
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from nltk.parse.stanford import StanfordDependencyParser
import collections
import torch


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def printFail(text):
    print(Bcolors.FAIL)
    print(text)
    print(Bcolors.ENDC)


def remove_numbers(s):
    remove_digits = str.maketrans('', '', digits)
    return s.translate(remove_digits)


def _set_color(color):
    if color == "green":
        return parse_xml(r'<w:shd {} w:fill="2ECC40"/>'.format(nsdecls('w')))
    elif color == "red":
        return parse_xml(r'<w:shd {} w:fill="DDDDDD"/>'.format(nsdecls('w')))


def _create_table(document, data2d, color_rows, headers):
    table = document.add_table(rows=1, cols=len(headers))
    hdr_cells = table.rows[0].cells
    for i, header in enumerate(headers):
        hdr_cells[i].text = header

    for data1d, color in zip(data2d, color_rows):
        row_cells = table.add_row().cells

        for i, item in enumerate(data1d):
            row_cells[i].text = str(item)
            row_cells[i]._tc.get_or_add_tcPr().append(_set_color(color))

    document.add_page_break()
    return document


def pretty_write(document, coder1, coder2, p_name, c, mutual_list, data):
    data_copy = data.copy()
    data_copy.append(
        [
            "(line no:{})[{}]--->\n(line no:{})[{}]".format(
                a1.lineid, " ".join(a1.words), a2.lineid, " ".join(a2.words)
            ) for a1,
            a2 in mutual_list
        ]
    )
    data_copy = map(list, zip(*data_copy))
    print("{0}:{1:.4f}".format(p_name, c))
    data_copy = list(data_copy)
    document.add_heading("{0}".format(p_name), 0)
    color_rows = [
        "green" if rel1 == rel2 else "red" for rel1, rel2, _ in data_copy
    ]
    document = _create_table(
        document, data_copy, color_rows, [coder1, "answer key", 'Relation']
    )

    return document


def get_stanford_dep_parser(path_to_jar, version, path_to_models_jar, dirpath):
    def decorator(fn):
        def decorated(*args, **kwargs):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=DeprecationWarning
                    )
                    dep = StanfordDependencyParser(
                        path_to_jar=path_to_jar,
                        path_to_models_jar=path_to_models_jar,
                        java_options="-mx3000m"
                    )

            except (
                pickle.UnpicklingError,
                EOFError,
                FileNotFoundError,
                TypeError,
                LookupError
            ):
                r = requests.get(
                    "https://nlp.stanford.edu/software/" + version + ".zip"
                )
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(dirpath)
                z.close()
                print(dirpath)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=DeprecationWarning
                    )
                    dep = StanfordDependencyParser(
                        path_to_jar=path_to_jar,
                        path_to_models_jar=path_to_models_jar,
                        java_options="-mx3000m"
                    )

            kwargs['dep_parser'] = dep
            return fn(*args, **kwargs)

        return decorated

    return decorator


def save(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        savefile = kwargs['savefile']
        kwargs.pop('savefile', None)
        try:
            data = pickle.load(open(savefile, 'rb'))
        except (
            pickle.UnpicklingError,
            EOFError,
            FileNotFoundError,
            TypeError,
            AttributeError
        ):
            data = func(*args, **kwargs)
            if savefile:
                os.makedirs(os.path.dirname(savefile), exist_ok=True)
                pickle.dump(data, open(savefile, "wb"))

        return data

    return wrapper_decorator


def flatten_dict(d, divider='/'):
    def __flatten_dict(pyobj, keystring=''):

        if type(pyobj) is dict:
            if type(pyobj) is dict:
                keystring = (keystring + divider if keystring else keystring)
                for k in pyobj:
                    yield from __flatten_dict(pyobj[k], keystring + k)

            elif type(pyobj) is list:
                for lelm in pyobj:
                    yield from __flatten_dict(lelm, keystring)
        else:
            yield keystring, pyobj

    return {k: v for k, v in __flatten_dict(d)}


def flatten_list(l):
    for el in l:
        if (
            isinstance(el, collections.Iterable) and
            not isinstance(el, (str, bytes))
        ):
            yield from flatten_list(el)
        else:
            yield el


# From https://github.com/allenai/allennlp/blob/0e64b4d3281808fac0fe00cc5b56e5378dbb7615/allennlp/nn/util.py#L1981
def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is 
    used to avoid numerical issues such as division by zero.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def min_value_of_dtype(dtype: torch.dtype):
    """
    Returns the minimum value of a given PyTorch data type. 
    Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).min


def info_value_of_dtype(dtype: torch.dtype):
    """
    Returns the `finfo` or `iinfo` object of a given PyTorch data type. 
    Does not allow torch.bool.
    """
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    elif dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)


def multicol_table(df: pd.DataFrame, header1, header2):
    df.columns = [
        [
            ">",
            "ent",
            "<",
            ">",
            "FULL",
            "<",
            ">",
            "IAP",
            "<",
            ">",
            "CAPTAC",
            "<",
            ">",
            "intra_CAPTAC",
            "<",
            ">",
            "inter_CAPTAC",
            "<"
        ], ["P", "R", "F1"] * 6
    ]
