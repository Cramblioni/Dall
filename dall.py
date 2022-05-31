
# very simple algebraic test language
###############################################################################
# syntax
# each line is either a declaration, definition, or a function call
# So a script may look like
## print "Hello, World"
# or like
## def squared as int -> int
##     squared x = * x x
## print (squared 10)
# which is just a simple program
# any function called in the outermost scope is voided

import sys, os

from enum import Enum, auto
from dataclasses import dataclass
from collections import ChainMap
from functools import reduce

prepare_script = r'''
python
  from functools import reduce
  globals()["reduce"] = reduce
  @TieIn(2)
  def add(*args):
    return reduce(lambda x,y: x + y,args)
  @TieIn(2)
  def sub(x,*args):
    return reduce(lambda x,y: x - y,args)
  @TieIn(2)
  def mul(*args):
    return reduce(lambda x,y: x * y,args)
  @TieIn(2)
  def div(*args):
    return reduce(lambda x,y: x / y,args)
  @TieIn(1)
  def chr(x):return __builtins__.chr(x)
  @TieIn(1)
  def show(x): return repr(x)
  #@TieIn(3)
  #def flip(f,a,b):
  #  return f(b,a)
  @TieIn(1)
  def print(*args):
    __builtins__.print(*args)
  @TieIn(1)
  def error(*args):
    __builtins__.print(*args,file=sys.stderr)
  @TieIn(0)
  def input(*args):
    return input(*args)

dull id
  a = a

dull const
  a b = a

dull flip
  f a b = f b a

error "Succesfully bound standard module"
'''

tests = [r'''
print "Hello, World!"
''',r'''
dull john
  3 = 9
  x = 10
  
dull displayMul
   x y = print (add (add (add (show x) " * ") (add (show y) " = ")) (show (mul x y)))

python
  @TieIn(1)
  def toInt(x): return int(x)

error "Demonstrating Pattern Matching"

(extend print "john 5") (john 5)
(extend print "john 3") (john 3)
(extend print "john 8") (john 8)

error "Demonstrating Function Operation"

displayMul (toInt (extend input "x = ")) (toInt (extend input "y = "))
''']

class _ChunkMode(Enum):
  """
An Enum for The modes the chunker knows.

Normal:
  splits up and annotates each line into either word/literal tokens.
Nothing:
  returns back the line structure for that line and any indent line.
Raw:
  returns the body as a string that would be needed to make that structure
  again in the chunker.

This also has signals to send tell the chunker how it should approach changes
in mode, These are just :

Block:
  Takes one mode then applies that to the current body.
Split:
  For each line in the current body, query for what mode to use.
Ignore:
  Ignores the current body

The chunker handles the toplevel as Split.
  """
  Normal  = auto()
  Nothing = auto()
  Raw     = auto()

  # signals
  Block   = auto()
  Split   = auto()
  Ignore  = auto()

class _TokenType(Enum):
  WORD    = auto()
  INT     = auto()
  FLOAT   = auto()
  STRING  = auto()
  TO      = auto()
  INTO    = auto()
  EQUAL   = auto()
  OPAREN  = auto()
  CPAREN  = auto()
  BIOPER  = auto()
  EOI     = auto()

def __scanNothing(struct):
  """
just returns the line structure given
  """
  yield struct

def __scanRaw(struct):
  """
Takes The line structure and rebuilds it into text
  """
  def lhandle(line,depth):
    return "\t"*depth + line[1] + "\n" +\
           "".join(lhandle(l,depth+1) for l in line[2])
  if isinstance(struct,list):
    yield "\n".join(lhandle(l,0) for l in struct).expandtabs(4)
  elif isinstance(struct,tuple):
    yield lhandle(struct,0).expandtabs(4)

def __scanNormal(struct):
  """
Tokens are structured as (type, lexeme, at)
Token List is structured as (line num, tokens)
  """
  lnum,text,body = struct
  cind : int = 0
  mxlen = len(text)
  out : list = []

  def peek() -> str:
    return text[cind] if cind < mxlen else "\x00"

  def advance() -> str:
    nonlocal cind
    ret = peek()
    cind += 1
    return ret
  
  # we tokenise one line of text
  while cind < mxlen:
    if peek() in " \t":
      advance()
      continue
    elif peek() == ";":
      break
    # the main part of the scan loop
    start = cind
    c = advance()
    ## names
    if c.isalpha() or c == "_":
      if c == "_":
        out.append((_TokenType.WORD,c,start))
        continue
      buff : str = c
      while peek().isalpha():
        buff += advance()
      out.append((_TokenType.WORD,buff,start))
    ## numbers
    elif c.isdigit():
      buff : str = c
      while peek().isdigit() or peek() == "_":
        if peek() == "_":
          advance()
        else:
          buff += advance()
      if peek() != ".":
        out.append((_TokenType.INT,buff,start))
      else:
        buff += advance()
        while peek().isdigit() or peek() == "_":
          if peek() == "_":
            advance()
          else:
            buff += advance()
        out.append((_TokenType.FLOAT,buff,start))
    elif c == '"':
      buff : str = ""
      #print("Started String Scrape",file=sys.stderr)
      while cind < mxlen and peek() != '"':
        buff += advance()
      assert cind < mxlen,\
             f"line {lnum} column {cind}: Unterminated String Literal"
      #print("Stopped String Scrape",file=sys.stderr)
      advance()
      out.append((_TokenType.STRING,buff,start))
    elif text[start:cind+1] == "->":
      out.append((_TokenType.TO,c+advance(),start))
    elif text[start:cind+1] == "<-":
      out.append((_TokenType.INTO,c+advance(),start))
    elif c == "=":
      out.append((_TokenType.EQUAL,c,start))
    elif c == "(":
      out.append((_TokenType.OPAREN,c,start))
    elif c == ")":
      out.append((_TokenType.CPAREN,c,start))
    elif c in "+-*/":
      out.append((_TokenType.BIOPER,c,start))
  yield (lnum,out)
  # now we figure out what we're doing and hand off

##  print("awaiting handling approach",file=sys.stderr)
  handle = yield
##  print("testing",handle.name,file=sys.stderr)
  if handle is _ChunkMode.Block:
##    print("using block approach",file=sys.stderr,end=" ")
    mode = yield
##    print("with",mode.name,file=sys.stderr)
    oper = __ChunkerBindings[mode]
    for line in body:
##      print("block handling ::",line,file=sys.stderr)
      yield from oper(line)
  elif handle is _ChunkMode.Split:
##    print("using split approach",file=sys.stderr)
    for line in body:
##      print("split handling ::",line,file=sys.stderr,end=" ")
      mode = yield
##      print("with",mode.name,file=sys.stderr)
      yield from __ChunkerBindings[mode](line)
  elif handle is _ChunkMode.Ignore:
##    print("Ignoring body",file=sys.stderr)
    pass
  else:
    pass
##    print("Failed check with",handle,file=sys.stderr)
  yield (lnum,_TokenType.EOI)
  return
    

__ChunkerBindings = {
  _ChunkMode.Nothing  : __scanNothing,
  _ChunkMode.Raw      : __scanRaw,
  _ChunkMode.Normal   : __scanNormal
}

def _chunk(text):
  """
Breaks up the text into chunks and builds up a very simple line map
interacts with the scanner/parser to dispatch different chunking modes
per line.

this is an iterator, use `next` on it before sending it a parse mode
  """
  # following the format (linenum,indent,line) for each line
  # here's a nested function to simplify this
  def _processLine(line):
    sl = line.lstrip()
    return len(line)-len(sl),sl
  # now we split up the lines
  lines = [
    (i,*_processLine(l))
    for i,l in enumerate(text.split("\n"))
    if l.strip()
  ]
  # if `lines` is empty the we return immediately.
  # because the code beyond does not work with `lines` as an empty list
  if not lines:
    return
  # now we rebuild the lines into a nested structure of lines
  # that store the indentation of the original text
  # here lines will have the format of (linenum,line,body)
  line_structure : list = []
  indent_stack   : list = [lines[0][1]]
  body_stack     : list = [line_structure]
  for lnum,indent,line in lines:
    if indent > indent_stack[-1]:
      indent_stack.append(indent)
      body_stack.append(body_stack[-1][-1][2])      
    elif indent < indent_stack[-1]:
      while indent_stack[-1] != indent:
        indent_stack.pop()
        body_stack.pop()        
    body_stack[-1].append((lnum,line,[]))
  
  # Now we bind the respective generators to the modes
  
  for l in line_structure:
##    print("Awaiting Main Mode",file=sys.stderr)
    mode = yield
##    print("Using",mode.name,file=sys.stderr)
    yield from __ChunkerBindings[mode](l)

######################################################
# support funcs

def probe(func):
  def wrap(*args,**kwargs):
    print(func.__name__,"called with arguments",file=sys.stderr,end="")
    print("",*args,sep="\n\t",file=sys.stderr)
    print("and Keyword arguments",file=sys.stderr,end="")
    ret = func(*args,**kwargs)
    print(func.__name__,"returned ::",ret,file=sys.stderr)
    return ret
  return wrap

def massub(globs,expr):
  return reduce(lambda x,y: substitute_Expr(*y,x),globs.items(),expr)

def gentiein(globs):
  def tiein(nargs):
    def wrap(func):
      globs[NameVal(func.__name__)] = pyProxy(func,nargs,())
      return func
    return wrap
  return tiein

######################################################

#################### Interpreter #####################
##### Everything Below This Line is Experimental #####
############### and is prone to change ###############

class InterpMode(Enum):
  Expr    = auto()
  Assign  = auto()
  FuncDef = auto()

def Interpret(text,globs,segment=False):
  globs = globs.copy()
  itr = _chunk(text)
  for _ in itr:
    ln,toks = itr.send(_ChunkMode.Normal)
    #print(f"{ln} :",*toks,sep="\n\t> ")
    next(itr)
    # handling either eob and no toks
    if len(toks) == 0:
      itr.send(_ChunkMode.Ignore)
      continue
    elif toks is _TokenType.EOI:
      break
    # now the implementations
    if toks[0][1] == "python":
      # Python mode, Allows the execution of python code
      # and allows you to tie in python code to the dall
      # enviroment
      itr.send(_ChunkMode.Block)
      out : str = ""
      tmp : tuple = itr.send(_ChunkMode.Raw)
      while tmp[1] is not _TokenType.EOI:
        out += tmp
        tmp = next(itr)
      penv = {"__builtins__":__builtins__,
              "TieIn":gentiein(globs),
              "sys":sys}
      exec(out,penv,{})

    elif toks[0][1] == "dull":
      itr.send(_ChunkMode.Block)
      
      targType,target,col = toks[1]
      assert targType is _TokenType.WORD,\
             f"line {ln} column {col}: "\
             f"Expected Word, got {targType.name}"
      assert NameVal(target) not in globs,\
             f"line {ln} column {col}: "\
             f"Cannot reassign constant {target}"
      partials = []
      globs[NameVal(target)] = dallFunct(partials,target)
      globals()["DEBUG_funct"] = globs[NameVal(target)] 
      defs = []
      sln,defin = itr.send(_ChunkMode.Normal)
      while defin is not _TokenType.EOI:
        defs.append((sln,defin))
        next(itr)
        itr.send(_ChunkMode.Ignore)
        sln,defin = next(itr)
      parsed = (*map(lambda x : parseLine(*x,InterpMode.FuncDef),defs),)
      partials.extend(map(lambda x : dallPartial(x[0],massub(globs,x[1])),
                          parsed))
      
    
    elif toks[1][0] is _TokenType.INTO:
      itr.send(_ChunkMode.Ignore)
      targType,target,col = toks[0]
      assert targType is _TokenType.WORD,\
             f"line {ln} column {col}: "\
             f"Expected Word, got {targType.name}"
      expr = parseLine(ln,toks[2:],InterpMode.Expr)
      expr = massub(globs,expr)
      assert NameVal(target) not in globs,\
             f"line {ln} column {col}: "\
             f"Cannot reassign constant {target}"
      globs[NameVal(target)] = solve(expr,True)
      
    else:
      # solves a single expression
      itr.send(_ChunkMode.Ignore)
      expr = parseLine(ln,toks,InterpMode.Expr)
      expr = massub(globs,expr)
      rv = solve(expr)
  return globs

def parseLine(lnum,tokens,mode):
  """
mode == Expr      => Expression
mode == FuncDef   => Pattern, Expression
  """
  cind,mxind = 0,len(tokens)

  def peek():   return tokens[cind]
  def testT(T): return peek()[0] is T
  def testL(L): return peek()[1] == L
  def advance():
    nonlocal cind
    ret,cind = peek(),cind + 1
    return ret

  def _Expression(term=()):
    out = [[]]
    while cind < mxind and peek()[0] not in term:
      tok,lexeme,at = cur = advance()
      if tok is _TokenType.WORD:
        out[-1].append(NameVal(lexeme))
      elif tok is _TokenType.INT:
        out[-1].append(LiterVal(int(lexeme)))
      elif tok is _TokenType.FLOAT:
        out[-1].append(LiterVal(float(lexeme)))
      elif tok is _TokenType.STRING:
        out[-1].append(LiterVal(lexeme))
      elif tok is _TokenType.OPAREN:
        temp = _Expression((_TokenType.CPAREN,))
        assert (err:=advance())[0] is _TokenType.CPAREN,\
               f"line {lnum} column {at}: "\
               "Expected \")\" not \"{err[1]}\""
        
        out[-1].append(temp)
      else:
        assert False,f"line {lnum} column {at}: Unexpected "\
               f"{tok.name} \"{lexeme}\""
    return tuple(out[0])

  def _FuncDef():
    # expects [pattern expression] = [expression]
    pattern_d = []
    while cind < mxind and not testT(_TokenType.EQUAL):
      pattern_d.append(advance())
    assert cind < mxind,\
           f"line {lnum}: "\
           "Expected \"=\" after pattern definition"
    assert peek()[0] is _TokenType.EQUAL,\
           f"line {lnum} at {cind}: "\
           f"expected \"=\" got \"{advance()[1]}\""
    advance()
    target = _Expression()
    pattern = []
    for ptype,plexeme,pat in pattern_d:
      if ptype is _TokenType.WORD:
        if plexeme == "_":
          pattern.append(dpWildcard)
        else:
          pattern.append(dpBinding(NameVal(plexeme)))
      elif ptype is _TokenType.INT:
        pattern.append(dpLiteral(LiterVal(int(plexeme))))
      elif ptype is _TokenType.FLOAT:
        pattern.append(dpLiteral(LiterVal(float(plexeme))))
      elif ptype is _TokenType.STRING:
        pattern.append(dpLiteral(LiterVal(plexeme)))
    return tuple(pattern),target

  if mode is InterpMode.Expr:
    return _Expression()
  elif mode is InterpMode.FuncDef:
    return _FuncDef()

##### Everything Below This Line is Experimental #####
############### and is prone to change ###############

@dataclass()
class NameVal:
  name : str
  def __hash__(self):
    return hash(self.name) ** 2
@dataclass()
class LiterVal:
  value : object
@dataclass()
class pyProxy:
  target : object
  nargs  : int
  args   : tuple

  def __call__(self,*args,py=False):
    pargs = map(lambda x: x.value if isinstance(x,LiterVal) else x,self.args)
    if py:
      return self.target(*pargs,*args)
    else:
      rv = self.target(*pargs,*args)
      if isinstance(rv,pyProxy):
        return rv
      else:
        return LiterVal(rv)

  def __repr__(self):
    return f"<{self.target.__name__}({','.join(map(repr,self.args))})"\
           f"~{self.nargs}>"

### For Functions and Pattern Matching
@dataclass()
class dpBinding:
  name : NameVal
@dataclass()
class dpWildcard: pass
@dataclass()
class dpLiteral:
  value : LiterVal
@dataclass()
class dallPartial:
  pattern : tuple
  target  : tuple
@dataclass()
class dallFunct:
  patterns : tuple
  name     : str = "<Function>"

  def __repr__(self): return self.name
  
###
def substitute_Expr(name,value,expr):
  out = []
  for item in expr:
    if item == name:
      out.append(value)
    elif isinstance(item,tuple | list):
      out.append(substitute_Expr(name,value,item))
    else:
      out.append(item)
  return tuple(out)

def apply_single(func,arg):
  if isinstance(arg,NameVal) or isinstance(func,NameVal):
    assert False, "Name Value In Expression"
  if isinstance(func,pyProxy):
    ret = pyProxy(func.target,func.nargs - 1,
                  func.args +\
                  (arg.value if isinstance(arg,LiterVal) else arg,)
                  )
    if ret.nargs == 0:
      return ret()
    else:
      return ret
  elif isinstance(func,dallFunct):
    partials = []
    for i in func.patterns:
      curAp,*npat = i.pattern
      ntarget = i.target
      if isinstance(curAp,dpBinding):
        ntarget = substitute_Expr(curAp.name,arg,ntarget)
      elif isinstance(curAp,dpWildcard):
        pass
      elif isinstance(curAp,dpLiteral) and curAp.value == arg:
        pass
      else:
        continue
      partials.append(dallPartial(npat,ntarget))

    for part in partials:
      if not part.pattern:
        return part.target
    assert partials,\
      f"Exhausted all patterns in {func.name}"
    return dallFunct(tuple(partials),func.name+"*")
    

def solve_expr(args):
  if len(args) == 1:
    return args[0]
  func,arg,*tail = args
  if isinstance(func,tuple | list):
    return solve(func),arg,*tail
  if isinstance(arg,tuple | list):
    return  func,solve(arg),*tail
  ret = apply_single(func,arg)
  return (ret,*tail)

def solve(args,ffs=False):
  #print("start of solve::",args)
  i = 1
  ret = solve_expr(args)
  #print(f"step {i}::",ret)
  while isinstance(ret,tuple):
    i += 1
    ret = solve_expr(ret)
    #print(f"step {i}::",ret)
  if not ffs and isinstance(ret,pyProxy) and ret.nargs == 0 :
    return ret()
  return ret

def solveMany(exprs,ffs=False):
  for args in exprs:
    out = solve(args,ffs)
  return out

################### Traversal Test ###################

def printTok(Tok):
  if Tok == None:
    print("Nothing Returned")
  elif isinstance(Tok,str):
    print("\n",Tok)
  else:
    l,t = Tok
    if t is not _TokenType.EOI:
      print(f"{l}:",*t,sep="\n> ")
    else:
      print("End Of Body")

def traverse(itr):
  while (inp:=input("? ").strip()).lower()!="quit":
    try:
      try:
        msg = _ChunkMode[inp]
        x = itr.send(msg)
        printTok(x)
      except KeyError:
        ret = next(itr)
        if ret != None:
          printTok(ret)
          
    except StopIteration:
      print("Finished Traversal")
      break
  else:
    print("Quit Traversal")


def main(scrpt):
  global _pglob,_rglob
  def dallExtend(f,x): return pyProxy(f.target,f.nargs,f.args + (x,))
  _tglob = ChainMap({NameVal("extend"):pyProxy(dallExtend,2,())})
  try:
    _pglob = Interpret(prepare_script,_tglob)
    print(scrpt)
    _rglob = Interpret(scrpt,_pglob)
  except AssertionError as e:
    print(e.args[0],file=sys.stderr)
##  #Interpret(tests[2],_pglob)
##  try:
##    while True:
##      expr = input("dall>")
##      if expr.startswith(":q"):
##        raise KeyboardInterrupt
##      temp = Interpret(expr,_pglob,False,True)
##      if isinstance(temp,LiterVal):
##        temp = temp.value
##      if temp != None:
##        print("=> ",repr(temp))
##  except KeyboardInterrupt:
##    print("Exiting Shell")

if __name__ == "__main__":
  main(tests[1])
  
