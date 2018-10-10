# parser.py
# Princeton Ferro, Oct 2018
# Tokenizer for C-like language and table-driven parser.
import io
import re
import functools
import operator

def foldl(func, acc, xs):
    return functools.reduce(func, xs, acc)

# How to use:
# file = open(...)
# tokenizer = Tokenizer(file)
# 
class Tokenizer:
    def __init__(self, iostream, debug=None):
        assert isinstance(iostream, io.TextIOBase)

        self.iostream = iostream
        self.buf = ''
        self.line = 0
        self.char = 0
        self.line_lengths = {}
        self.debug = (lambda x: print(f'scanner: {x}')) if debug else (lambda x: ())

    def position(self):
        return (self.line + 1, self.char + 1)

    # simply gets the very next character, which may be a space
    def next_char(self):
        c = ''
        if self.buf:
            c = self.buf[0]
            self.buf = self.buf[1:]
        else:
            c = self.iostream.read(1)

        if c == '\r' or c == '\n':
            self.line_lengths[self.line] = self.char
            self.line += 1
            self.char = 0
        elif c:
            self.char += 1

        return c

    def peek_char(self):
        c = self.next_char()
        self.save_char(c)

        return c

    def save_char(self, c, append=False):
        if c == '\n' or c == '\r':
            self.line -= 1
            self.char = self.line_lengths[self.line]
        else:
            self.char -= 1
        if append:          # for save_string()
            self.buf += c
        else:
            self.buf = self.buf + c

    # skips over spaces, comments, and meta-statements
    def skip(self):
        c = None
        notspace = False
        notcomment1 = False
        notcomment2 = False
        notmeta = False
        while not (notspace and notcomment1 and notcomment2 and notmeta):
            # skip over spaces
            while True:
                line, pos = self.position()
                c = self.next_char()
                if not c:   # EOF
                    return ''
                elif not c.isspace():
                    notspace = True
                    break
                if c == '\r' or c == '\n':
                    self.debug(f'skipping line {line}')
                notspace = False

            line, pos = self.position()
            # self.debug(f'{line}:{pos}: c = {c}')

            # skip over meta symbols
            if not c == '#':
                notmeta = True
            else:
                # self.debug(f'{line}:{pos}: skipping over meta')
                while not (c == '\r' or c == '\n'):
                    c = self.next_char()
                line, pos = self.position()
                # self.debug(f'{line}:{pos}: c = {c}')
                notmeta = True
                notspace = False

            # skip over one-line comments
            if not (c == '/' and self.peek_char() == '/'):
                notcomment1 = True
            else:
                # print(f'{line}:{pos}: skipping over one-line comment')
                while not ((c == '\r' or c == '\n') and self.peek_char() != '/'):
                    c = self.next_char()
                notcomment1 = True
                notspace = False        # we're no longer sure
            # skip over multi-line comments
            if not (c == '/' and self.peek_char() == '*'):
                notcomment2 = True
            else:
                # print(f'{line}:{pos}: skipping over multi-line comment')
                while not (c == '*' and self.peek_char() == '/'):
                    c = self.next_char()
                c = self.next_char()    # '/'
                c = self.next_char()    # this could be anything
                notspace = False
                notcomment1 = False
                notcomment2 = False
                notmeta = False

        return c

    def save_string(self, string, atspace=None):
        for c in string:
            self.save_char(c, append=True)
        if atspace == None or atspace:
            self.save_char(' ', append=True)

    # gets the next identifier
    def next_ident(self):
        # skip over spaces
        c = self.skip()

        if not c:
            return c

        if not re.match(r'[A-Za-z_]', c):
            self.save_char(c)
            return None

        s = c
        while True:
            c = self.next_char()
            if not c or c.isspace():    # EOF or space
                break
            if not re.match(r'\w', c):          # push c back
                self.save_char(c)
                break
            s += c

        return s

    def peek_ident(self):
        label = self.next_ident()
        self.save_string(label)
        return label

    # returns either a float or an int, or None if EOF
    def __next_num(self):
        # skip over spaces
        c = self.skip()

        if not c:
            return None

        s = c
        seendec = False
        seenexp = False                 # for floats of the form [\d]E{+-}[\d]
        seenexp2 = False                #
        while True:
            c = self.next_char()
            if not c or c.isspace():    # EOF or space
                break

            if not re.match(r'\d', c):
                if c == '.':
                    if not seendec:
                        seendec = True
                    else:
                        self.save_char(c)
                        break
                elif c.lower() == 'e':
                    if not seenexp:
                        seenexp = True
                    else:
                        self.save_char(c)
                        break
                elif c == '+' or c == '-':
                    if not seenexp:
                        self.save_char(c)
                        break
                    elif not seenexp2:
                        seenexp2 = True
                    else:
                        self.save_char(c)
                        break
                else:
                    self.save_char(c)
                    break

            s += c

        return ((float(s) if (seendec or seenexp) else int(s)) if s else None, s)

    def next_number(self):
        return self.__next_num()[0]

    def peek_number(self):
        number, s = self.__next_num()
        self.save_string(s)
        return number

    # returns the next item separated by spaces
    # or limited by the length [limit]
    def __next_tk(self, limit=None):
        c = self.skip()
        rem_char = ''

        if not c:
            return ('', False, '')

        s = c
        i = 1
        atspace = False
        while not limit or i < limit:
            prev_c = c
            c = self.next_char()
            if not c or c.isspace():            # EOF or space
                atspace = True
                break
            p = re.compile(r'[=|:(){};-><,&^%!~*+/\[\].?]')
            # break around operators
            if (p.match(c) or p.match(prev_c)) \
                    and not ((prev_c == c) and (c == '&' or c == '|' or c == '=' or c == '+' or c == '.')):
                rem_char = c
                break
            s += c
            i += 1

        return (s, atspace, rem_char)

    def next_token(self, limit=None):
        return self.__next_tk(limit)[0]

    def peek_token(self, limit=None):
        s, atspace, rem_char = self.__next_tk(limit)
        self.save_string(s, atspace)
        if rem_char:
            self.save_char(rem_char)
        return s

    def next_string(self):
        c = self.skip()

        if not c:
            return None

        seenquote = False
        s = ''
        # n_prev_slash = 0 (TODO: handle escapes)

        while True:
            if c == '"':
                if seenquote:
                    break               # this is the second quote
                else:
                    seenquote = True    # this is the first quote
            else:
                if not seenquote:       # we've seen something before our quote
                    self.save_char(c)
                    return None
                else:                   # we're in the quote
                    s += c

            c = self.next_char()

        return s

class ParseError(Exception):
    pass

# A table-driver parser
class Parser:
    def __init__(self, tokenizer, init_prod, grammar, combiners, debug=None):
        """
            tokenizer = A Tokenizer object
            init_prod = a string
            grammar = an array of tuples, (prod:string, array(prods))
            combiners = an array of tuples, (prod:string, array(int))
        """
        self.tokenizer = tokenizer
        self.init_prod = init_prod          # first production to expand
        self.grammar = dict(grammar)
        self.combiners = dict(combiners)
        self.keyword_re = r'auto|break|case|char|const|continue|default|do|double|else|enum|extern|float|for|goto|if|int|long|register|return|short|signed|sizeof|static|struct|switch|typedef|union|unsigned|void|volatile|while'
        self.word_re = r'[A-Za-z_]\w*'
        self.nontype_kw_re = r'auto|break|case|const|continue|default|do|else|extern|for|goto|if|return|sizeof|static|switch|typedef|while'
        self.terminals = dict([\
                ('number', re.compile(r'[0-9]*.?[0-9]+|[0-9]+.?[0-9]*')),\
                ('integer', re.compile(r'[0-9]+')),\
                ('ident', re.compile(fr'(?!({self.keyword_re})$){self.word_re}')),\
                ('type_ident', re.compile(fr'(?!({self.nontype_kw_re})$){self.word_re}')),\
                ('string', list('"')),\
                ])
        self.debug = (lambda x: print(f'parser: {x}')) if debug else (lambda x: ())

        # quick check
        for prod in self.grammar:
            if prod in self.combiners:
                if not len(self.combiners[prod]) == len(self.grammar[prod]):
                    raise Exception(f'{prod} has {len(self.grammar[prod])} productions but {len(self.combiners[prod])} combiners')
                for i in range(0, len(self.combiners[prod])):
                    for j in self.combiners[prod][i]:
                        if not j < len(self.grammar[prod][i]):
                            raise Exception(f'{prod}\'s #{i} rule is {len(self.grammar[prod][i])} but that combiner is {len(self.combiners[prod][i])}')
            else:
                raise Exception(f'{prod} doesn\'t have a combiner')

    def eps(self, symbol):
        if isinstance(symbol, list):
            # empty list?
            if not symbol:
                return True
            # list of productions?
            if isinstance(symbol[0], list):
                return foldl(operator.or_, False, [self.eps(prod) for prod in symbol])
            # list of symbols? (a production)
            if isinstance(symbol[0], str):
                for sym in symbol:
                    if not self.eps(sym):
                        return False
                return True

            raise Exception(f'symbol is something else: {symbol}')
        elif isinstance(symbol, str):
            # non-terminal?
            if symbol in self.grammar:
                return self.eps(self.grammar[symbol])
            # terminal ?
            if symbol == '':
                return True
            else:
                return False
        else:
            raise Exception(f'expected list or string for symbol: {symbol}')

    # rule = list of productions [[]] initially
    def first(self, rule):
        self.debug(f'computing first({rule})')
        if isinstance(rule, list):
            # empty list?
            if not rule:
                return []       # epsilon
            # list of productions?
            if isinstance(rule[0], list):
                return [st for prod in rule for st in self.first(prod)]
            # list of symbols? (a production)
            if isinstance(rule[0], str):
                first_set = self.first(rule[0])
                for i in range(1, len(rule)):
                    if self.eps(rule[i-1]):
                        self.debug(f'parser: {rule[i-1]} has epsilon, adding next')
                        first_set += self.first(rule[i])
                    else:
                        break
                return first_set
            raise Exception(f'rule is something else: {rule}')
        elif isinstance(rule, str):
            # non-terminal?
            if rule in self.grammar:
                return self.first(self.grammar[rule])
            # terminal symbol?
            if rule in self.terminals:
                return [self.terminals[rule]]
            else:
                return [rule]
        else:
            raise Exception(f'expected list or string for rule: {rule}')

    def parse(self):
        # used for matching elem to any string/regex in collection
        def contains(collection, elem):
            for item in collection:
                if isinstance(item, re.Pattern):
                    if item.match(elem):
                        return True
                else:
                    if item == elem:
                        return True
            return False

        stack = [self.init_prod]    # [each element is str or [[]] or (PROD, ALTNUM)]
        nodes = []                  # the computed nodes of the parse tree

        while stack:
            prod = stack[0]
            stack = stack[1:]

            if isinstance(prod, str):
                # non-terminal?
                if prod in self.grammar:
                    stack = [(prod, self.grammar[prod])] + stack
                # terminal
                else:
                    if prod == 'number':
                        tk = self.tokenizer.next_number()
                        if tk == None:
                            line, char = self.tokenizer.position()
                            raise ParseError(f'Expected number at line {line}, char {char}')
                        nodes.append(tk)
                    elif prod == 'integer':
                        tk = self.tokenizer.next_number()
                        if tk == None or not isinstance(tk, int):
                            line, char = self.tokenizer.position()
                            raise ParseError(f'Expected an integer at line {line}, char {char}')
                        nodes.append(tk)
                    elif prod == 'ident' or prod == 'type_ident':
                        tk = self.tokenizer.next_ident()
                        if tk == None:
                            line, char = self.tokenizer.position()
                            raise ParseError(f'Expected {prod} at line {line}, char {char}')
                        nodes.append(tk)
                    elif prod == 'string':
                        tk = self.tokenizer.next_string()
                        if tk == None:
                            line, char = self.tokenizer.position()
                            raise ParseError(f'Unexpected token \'{self.tokenizer.peek_token()}\' at line {line}, char {char}. Expected a string.')
                        nodes.append(tk)
                    elif prod != '':
                        tk = self.tokenizer.next_token(len(prod))
                        if not tk or tk != prod:
                            line, char = self.tokenizer.position()
                            raise ParseError(f'Expected token \'{prod}\' at line {line}, char {char}')
                        nodes.append(tk)

            elif isinstance(prod, tuple):
                one, two = prod

                if isinstance(two, int):
                    comb_name = one
                    alt_num = two
                    amt_skip = len(self.grammar[comb_name][alt_num])
                    c = (comb_name, [nodes[len(nodes)-amt_skip + i] for i in self.combiners[comb_name][alt_num]])
                    nodes = nodes[0:len(nodes)-amt_skip] + [c]
                elif isinstance(two, list):
                    # one is str
                    assert two and isinstance(two[0], list) # two is [[]]
                    found = None                        # the selected production []
                    found_i = None
                    peek = self.tokenizer.peek_token()
                    for i in range(0, len(two)):        # alternative production
                        alt = two[i]
                        first_set = self.first(alt)
                        self.debug(f'peek = {peek}, first({alt}) = {first_set}')
                        if contains(first_set, peek) or not alt:
                            found = alt
                            found_i = i
                            self.debug(f'selected {alt}')
                            break

                    if found == None:
                        line, char = self.tokenizer.position()
                        pname = f'<{one}>' if one in self.grammar else one
                        raise ParseError(f'Unexpected token \'{peek}\' at line {line}, char {char} while resolving {pname}')

                    # found could be empty, indicating EPS
                    stack = ([''] if not found else found) + [(one, found_i)] + stack
                else:
                    raise ParseError(f'Unexpected item {prod}')
            else:
                raise ParseError(f'Unexpected item {prod}')

        return nodes

