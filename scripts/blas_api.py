def arg_desc(ptr, nrows=None, ncols=None, output=None):
    obj = {}
    obj['ptr'] = ptr
    if nrows != None:
        obj['nrows'] = nrows
    if ncols != None:
        obj['ncols'] = ncols
    if output != None:
        obj['output'] = output
    return obj

arg_parsers = {\
        # level 1
        'ddot_': [ arg_desc(1, 0), arg_desc(3, 0) ],
        # level 2
        # level 3
        'sgemm_': [ arg_desc(6, 2, 4), arg_desc(8, 4, 3), arg_desc(11, 2, 3, True) ],
        'cblas_sgemm': [ arg_desc(7, 3, 5), arg_desc(9, 5, 4), arg_desc(12, 3, 4, True) ],
        'dgemm_': [ arg_desc(6, 2, 4), arg_desc(8, 4, 3), arg_desc(11, 2, 3, True) ],
        'cblas_dgemm': [ arg_desc(7, 3, 5), arg_desc(9, 5, 4), arg_desc(12, 3, 4, True) ],
        }

