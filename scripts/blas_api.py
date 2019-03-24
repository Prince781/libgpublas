class ArgDesc:
    def __init__(self, get_ptr, get_nrows=None, get_ncols=None, is_output=None, etype=None):
        self.get_ptr = (lambda args,ret: ret if get_ptr == -1 else args[get_ptr]) if isinstance(get_ptr, int) else get_ptr
        self.get_nrows = (lambda args,ret: args[get_nrows]) if isinstance(get_nrows, int) else get_nrows
        self.get_ncols = (lambda args,ret: args[get_ncols]) if isinstance(get_ncols, int) else get_ncols
        self.is_output = is_output if is_output != None else False
        self.elem_type = etype if etype != None else "B"    # default type is bytes

def ptr_info(argd, args, ret):
    return f"{argd.get_ptr(args,ret)} [{argd.get_nrows(args,ret)}x{argd.get_ncols(args,ret)} {argd.elem_type}]"

arg_parsers = {\
        # memory management
        'calloc': [ ArgDesc(-1, 0, 1) ],
        'malloc': [ ArgDesc(-1, lambda args,ret: 1, 0) ],
        # BLAS:
        # level 1
        'ddot_': [ ArgDesc(1, 0), ArgDesc(3, 0) ],
        # level 2
        # level 3
        'sgemm_': [ ArgDesc(6, 2, 4, etype="f32"), ArgDesc(8, 4, 3, etype="f32"), ArgDesc(11, 2, 3, True, etype="f32") ],
        'cblas_sgemm': [ ArgDesc(7, 3, 5, etype="f32"), ArgDesc(9, 5, 4, etype="f32"), ArgDesc(12, 3, 4, True, etype="f32") ],
        'dgemm_': [ ArgDesc(6, 2, 4, etype="f64"), ArgDesc(8, 4, 3, etype="f64"), ArgDesc(11, 2, 3, True, etype="f64") ],
        'cblas_dgemm': [ ArgDesc(7, 3, 5), ArgDesc(9, 5, 4), ArgDesc(12, 3, 4, True) ],
        }

