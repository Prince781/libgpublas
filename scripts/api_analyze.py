#!/bin/env python3
# Analyzes all function definitions in a header.

from parser import *

grammar = [\
            ('decl_list', [['decl', 'decl_list'], []]),\
            ('decls', [['decl', 'decls'], []]),\
            ('decl', [['fun_decl', ';'], ['extern_decl'], ['typedef_decl', ';'], ['enum_decl', ';'], ['struct_decl', ';']]),\
            ('extern_decl', [['extern', 'extern_decl_tail']]),\
            ('extern_decl_tail', [['type', 'ident', ';'], ['"C"', '{', 'decls', '}']]),\
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
            ('typedef_type', [['enum_decl_anon', 'ident'], ['type']]),\
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
            ('extern_decl_tail', [[0, 1], [2]]),\
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
            ('typedef_type', [[0, 1], [0]]),\
            ('struct_elem', [[0, 1]]),\
            ('struct_elem_list', [[0, 1], []]),\
            ('struct_decl_anon', [[2, 3]]),\
            ('struct_decl', [[1, 3, 4]]),\
            ('param_list_nonempty', [[0, 1, 2]]),\
            ('param_list', [[0], []]),\
            ('param_tail', [[1], []]),\
            ('param_tail_list_or_varargs', [[0], [0]]),\
        ]


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} c-header")

    try:
        f = sys.stdin if sys.argv[1] == '-' else open(sys.argv[1], 'rt')
        tree = Parser(Tokenizer(f), 'decl_list', grammar, combiners).parse()
        print(tree)
        f.close()
    except ParseError as e:
        print(f'syntax error: {e}')
    except Exception as e:
        print(e)
