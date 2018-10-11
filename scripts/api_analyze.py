#!/bin/env python3
# Analyzes all function definitions in a header.

from parser import *

# see cgram.txt

class c_type:
    def __init__(self, name, const):
        self.name = name
        self.const = bool(const)

class c_typedef(c_type):
    def __init__(self, name, inner):
        c_type.__init__(self, name, False)
        self.inner = inner

class c_ref(c_type):
    def __init__(self, const, deref):
        c_type.__init__(self, None, const)
        self.deref = deref

class c_function(c_type):
    def __init__(self, name, args, ret):
        c_type.__init__(self, name, False)
        self.args = args
        self.ret = ret

class c_struct(c_type):
    def __init__(self, name, members):
        c_type.__init__(self, name, False)
        self.members = dict(members)

class c_enum(c_type):
    def __init__(self, name, members):
        c_type.__init__(self, name, False)
        self.members = dict(members)

class c_scope:
    def __init__(self, c_funs, c_types):
        self.funs = c_funs
        self.types = dict(c_types)

def make_ref(cref, idlist, last_name):
    return None     # TODO

# action routines: function(lhs: {}, rhs: str -> {}) -> None

def prod__decl_list(lhs, rhs, init):
    if init:
        lhs.update(scope=[])
        return
    if not rhs:
        return
    lhs['scope'] = rhs['decl']['scope'] + rhs['decl_list']['scope']

def prod__decls(lhs, rhs, init):
    if init:
        lhs.update(scope=[])
        return
    lhs['scope'] = rhs['decl']['scope'] + rhs['decls']['scope']

def prod__decl(lhs, rhs, init):
    if init:
        lhs.update(scope={'funs': [], 'types': []})
        return
    if 'fun_decl' in rhs:
        lhs['scope']['funs'] += rhs['fun_decl']['func']
    elif 'typedef_decl' in rhs:
        lhs['scope']['types'] += rhs['typedef_decl']['type']
    elif 'enum_decl' in rhs:
        lhs['scope']['types'] += rhs['enum_decl']['enum']
    elif 'struct_decl' in rhs:
        lhs['scope']['types'] += rhs['struct_decl']['struct']

def prod__fun_decl(lhs, rhs, init):
    if init:
        lhs.update(func=None)
        return
    lhs['func'] = c_function(rhs['ident?']['val'] or rhs['type']['last_ident'], rhs['param_list']['args'], make_ref(rhs['type']['ref'], rhs['type']['idents'], rhs['ident?']['val']))

def prod__qualifier(lhs, rhs, init):
    if init:
        lhs.update(const=False)
        return
    if rhs:
        lhs['const'] = True

def prod__type(lhs, rhs, init):
    if init:
        lhs.update(idents=[], last_ident=None, const=False, ref=None)
        return
    lhs['idents'] = rhs['ident']['val'] + rhs['type_id_list']['idents']
    lhs['last_ident'] = rhs['type']['idents'][-1:]
    lhs['const'] = rhs['qualifier']['const']
    lhs['ref'] = rhs['type_tail']['ref']

def prod__type_id_list(lhs, rhs, init):
    if init:
        lhs.update(idents=[])
        return
    if not rhs:
        return
    lhs['idents'] += rhs['type_ident']['val'] + rhs['type_id_list']['idents']

def prod__type_tail(lhs, rhs, init):
    if init:
        lhs.update(ref=None)
        rhs['type_tail']['ref'] = c_ref(rhs['qualifier']['const'], rhs['deref']['ref'])
    if not rhs:
        return
    lhs['ref'] = rhs['type_tail']['ref']

def prod__param_list(lhs, rhs, init):
    if init:
        lhs.update(args=[])
        return
    if not rhs:
        return
    lhs['args'] = rhs['param_list_nonempty']['args']

def prod__param_list_nonempty(lhs, rhs, init):
    if init:
        lhs.update(args=[])
        return
    lhs['args'] += [make_ref(rhs['type']['ref'], rhs['type']['idents'], rhs['ident?']['val'])]
    lhs['args'] += rhs['param_tail']['args']

def prod__param_tail(lhs, rhs, init):
    if init:
        lhs.update(args=[])
        return
    if not rhs:
        return
    lhs['args'] += rhs['param_tail_list_or_varargs']['args']

def prod__param_tail_list_or_varargs(lhs, rhs, init):
    if init:
        lhs.update(args=[])
        return
    if not 'param_list_nonempty' in rhs:
        return
    lhs['args'] += rhs['param_list_nonempty']['args']

def prod__enum_decl(lhs, rhs, init):
    if init:
        lhs.update(enum=None)
        return
    lhs['enum'] = c_enum(rhs['ident?']['val'], [rhs['enum_elem']['elem']] + rhs['enum_elem_list']['elems'])

def prod__ident_q(lhs, rhs, init):
    if init:
        lhs.update(val=None)
        return
    if not rhs:
        return
    lhs['val'] = rhs['ident']['val']

def prod__enum_elem(lhs, rhs, init):
    if init:
        lhs.update(elem=None)
        return
    lhs['elem'] = (rhs['ident']['val'], rhs['integer']['val'])

def prod__enum_elem_list(lhs, rhs, init):
    if init:
        lhs.update(elems=[])
        return
    if not rhs:
        return
    lhs['elems'] += [rhs['enum_elem']['elem']] +rhs['enum_elem_list']['elems']

def prod__typedef_decl(lhs, rhs, init):
    if init:
        lhs.update(type=None)
        return
    lhs['type'] = rhs['typedef_type']['type']

def prod__typedef_type(lhs, rhs, init):
    if init:
        lhs.update(type=None)
        return
    if 'enum_decl_anon' in rhs:
        lhs['type'] = c_typedef(rhs['ident']['val'], rhs['enum_decl_anon']['enum'])
    elif 'struct_decl_anon' in rhs:
        lhs['type'] = c_typedef(rhs['ident']['val'], rhs['struct_decl_anon']['struct'])
    elif 'type' in rhs:
        lhs['type'] = c_typedef(rhs['type']['last_ident'], make_ref(rhs['type']['ref'], rhs['type']['idents'], None))

def prod__struct_decl(lhs, rhs, init):
    if init:
        lhs.update(struct=None)
        return
    lhs['struct'] = c_struct(rhs['ident?']['val'], [rhs['struct_elem']['elem']] + rhs['struct_elem_list']['elems'])

def prod__struct_elem(lhs, rhs, init):
    if init:
        lhs.update(elem=None)
        return
    lhs['elem'] = (rhs['ident?']['val'] or rhs['type']['last_ident'], make_ref(rhs['type']['ref'], rhs['type']['idents'], rhs['ident?']['val']))

def prod__struct_elem_list(lhs, rhs, init):
    if init:
        lhs.update(elems=[])
        return
    if not rhs:
        return
    lhs['elems'] += [rhs['struct_elem']['elem']] + rhs['struct_elem_list']['elems']

grammar = [\
            ('decl_list', [['decl', 'decl_list'], []]),\
            ('decls', [['decl', 'decls'], []]),\
            ('decl', [['fun_decl', ';'], ['extern_decl'], ['typedef_decl', ';'], ['enum_decl', ';'], ['struct_decl', ';']]),\
            ('extern_decl', [['extern', 'extern_decl_tail']]),\
            ('extern_decl_tail', [['type', 'ident?', 'array_decl?', ';'], ['"C"', '{', 'decls', '}']]),\
            ('fun_decl', [['type', 'ident?', '(', 'param_list', ')']]),\
            ('qualifier', [['const'], []]),\
            ('type', [['qualifier', 'type_ident', 'type_id_list', 'type_tail']]),\
            ('type_id_list', [['type_ident', 'type_id_list'], []]),\
            ('type_tail', [['*', 'qualifier', 'type_tail'], []]),\
            ('ident?', [['ident'], []]),\
            ('enum_elem', [['ident', '=', 'integer']]),\
            ('enum_elem_list', [[',', 'enum_elem', 'enum_elem_list'], []]),\
            ('enum_decl_named', [['enum', 'ident', '{', 'enum_elem', 'enum_elem_list', '}']]),\
            ('enum_decl_anon', [['enum', '{', 'enum_elem', 'enum_elem_list', '}']]),\
            ('enum_decl', [['enum', 'ident?', '{', 'enum_elem', 'enum_elem_list', '}']]),\
            ('typedef_decl', [['typedef', 'typedef_type']]),\
            ('typedef_type', [['enum_decl_anon', 'ident'], ['struct_decl_anon', 'ident'], ['type']]),\
            ('array_idx', [['integer'], ['ident']]),\
            ('array_decl', [['[', 'array_idx', ']']]),\
            ('array_decl?', [['array_decl'], []]),\
            ('struct_elem', [['type', 'ident?']]),\
            ('struct_elem_list', [['struct_elem', 'struct_elem_list'], []]),\
            ('struct_decl_anon', [['struct', '{', 'struct_elem', 'struct_elem_list', '}']]),\
            ('struct_decl', [['struct', 'ident?', '{', 'struct_elem', 'struct_elem_list', '}']]),\
            ('param_list_nonempty', [['type', 'ident?', 'param_tail']]),\
            ('param_list', [['param_list_nonempty'], []]),\
            ('param_tail', [[',', 'param_tail_list_or_varargs'], []]),\
            ('param_tail_list_or_varargs', [['param_list_nonempty'], ['...']]),\
        ]
combiners = [\
            ('decl_list', [[0, 1], []]),\
            ('decls', [[0, 1], []]),\
            ('decl', [[0], [0], [0], [0], [0]]),\
            ('extern_decl', [[1]]),\
            ('extern_decl_tail', [[0, 1, 2], [2]]),\
            ('fun_decl', [[0, 1, 3]]),\
            ('qualifier', [[0], []]),\
            ('type', [[0, 1, 2, 3]]),\
            ('type_id_list', [[0, 1], []]),\
            ('type_tail', [[0, 1, 2], []]),\
            ('ident?', [[0], []]),\
            ('enum_elem', [[0, 1, 2]]),\
            ('enum_elem_list', [[1, 2], []]),\
            ('enum_decl_named', [[1, 3, 4]]),\
            ('enum_decl_anon', [[2, 3]]),\
            ('enum_decl', [[1, 3, 4]]),\
            ('typedef_decl', [[1]]),\
            ('typedef_type', [[0, 1], [0, 1], [0]]),\
            ('array_idx', [[0], [0]]),\
            ('array_decl', [[1]]),\
            ('array_decl?', [[0], []]),\
            ('struct_elem', [[0, 1]]),\
            ('struct_elem_list', [[0, 1], []]),\
            ('struct_decl_anon', [[2, 3]]),\
            ('struct_decl', [[1, 3, 4]]),\
            ('param_list_nonempty', [[0, 1, 2]]),\
            ('param_list', [[0], []]),\
            ('param_tail', [[1], []]),\
            ('param_tail_list_or_varargs', [[0], [0]]),\
        ]
actions = [\
            ('decl_list', prod__decl_list),\
            ('decls', prod__decls),\
            ('decl', prod__decl),\
            ('fun_decl', prod__fun_decl),\
            ('qualifier', prod__qualifier),\
            ('type', prod__type),\
            ('type_id_list', prod__type_id_list),\
            ('type_tail', prod__type_tail),\
            ('ident?', prod__ident_q),\
            ('enum_elem', prod__enum_elem),\
            ('enum_elem_list', prod__enum_elem_list),\
            ('enum_decl', prod__enum_decl),\
            ('typedef_decl', prod__typedef_decl),\
            ('typedef_type', prod__typedef_type),\
            ('struct_elem', prod__struct_elem),\
            ('struct_elem_list', prod__struct_elem_list),\
            ('struct_decl', prod__struct_decl),\
            ('param_list_nonempty', prod__param_list_nonempty),\
            ('param_list', prod__param_list),\
            ('param_tail', prod__param_tail),\
            ('param_tail_list_or_varargs', prod__param_tail_list_or_varargs),\
        ]


if __name__ == "__main__":
    import sys
    from argparse import *

    aparser = ArgumentParser()
    aparser.add_argument('c_header')
    aparser.add_argument('-v', '--verbose', action='store_true', help='Print all messages')
    aparser.add_argument('-s', '--debug-scanner', action='store_true', help='Debug only scanner')
    aparser.add_argument('-p', '--debug-parser', action='store_true', help='Debug only parser')

    try:
        args = aparser.parse_args()
        f = sys.stdin if args.c_header == '-' else open(args.c_header, 'rt')
        tk = Tokenizer(f, debug=args.verbose or args.debug_scanner)
        tree = Parser(tk, 'decl_list', grammar, combiners, actions, debug=args.verbose or args.debug_parser).parse()
        print(tree)
        f.close()
    except ParseError as e:
        print(f'syntax error: {e}')
    except OSError as e:
        print(e)
