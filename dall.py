
import sys, os

from enum import Enum, auto
from dataclasses import dataclass
from collections import ChainMap
from functools import reduce

#trying {1b} in todo along with a little of {2b}

# firstly, Enums and dataclasses (:
# I'm programming this in Python3.10, so i'd use 
# dataclass(slots=True,frozen=True) for most of these dataclasses But i don't
# know what people will be running this on, so I'm leaving them blank

_DALL_KEYWORDS : tuple[str] = ("dull",)

@dataclass()
class TokLoc:
  """
for storing the location of tokens in the script in a format thats easily
appliable
  """
  line   : int
  column : int
  lstart : int
  lend   : int
  endcol : int = -1

  def __repr__(self):
    return f"@{self.line}:{self.column}:"

class TokenType(Enum):
  WORD    = auto()
  KEYWORD = auto()
  LITERAL = auto()
  OPAREN  = auto()
  CPAREN  = auto()

@dataclass()
class Token:
  type    : TokenType
  loc     : TokLoc
  lexeme  : str

  def __repr__(self):
    return f"{self.type.name}:{self.loc}:{self.lexeme}"


# custom Error Type to make life easier
@dataclass()
class dallError(BaseException):
  msg : str
  loc : TokLoc
  def __init__(self,msg,loc):
    self.msg,self.loc = msg,loc
# and a nice little function to make raising the error nicer
def throwError(loc : TokLoc,msg : str):
  raise dallError(f"{loc.line}:{loc.column}:: {msg}",loc)


def _scanLine(line : str, lnum : int,start : int, end : int) -> tuple[Token]:
  cind   : int         = 0
  mxind  : int         = len(line)
  obuff   : list[Token] = []
  tstart : int 

  def peek() -> str: return line[cind] if cind < mxind else "\x00"
  def advance() -> str:
    nonlocal cind
    ret,cind = peek(),cind+1
    return ret

  def genTok(_type : TokenType, lexeme : str) -> Token:
    return Token(_type,TokLoc(lnum,tstart,start,end,tstart+len(lexeme)),lexeme)

  while cind < mxind:
    if peek() == " ":
      advance()
      continue
    
    tstart = cind
    cc = advance()
    if cc.isalpha() or cc == "_":
      buff : str = cc
      while peek().isalnum() or peek() == "_":
        buff += advance()
      if buff in _DALL_KEYWORDS:
        obuff.append(genTok(TokenType.KEYWORD,buff))
      else:
        obuff.append(genTok(TokenType.WORD,buff))
    elif cc.isdigit():
      buff : str = cc
      while peek().isdigit():
        buff += advance()
      if peek() == ".":
        buff += advance()
        while peek().isdigit():
          buff += advance()
        obuff.append(genTok(TokenType.LITERAL,float(buff)))
      else:
        obuff.append(genTok(TokenType.LITERAL,int(buff)))
    elif cc == '"':
      buff : str = ""
      while cind < mxind and peek() != '"':
        if peek() == "\\":
          advance()
          escode = advance()
        else:
          buff += advance()
      advance()
      obuff.append(genTok(TokenType.LITERAL,buff))
    elif cc == ";":
      break
    else:
      throwError(TokLoc(lnum,tstart,start,end),
        "Unexpected Token \"%s\""%cc)
  return tuple(obuff)
    

def chunk(text : str) -> tuple:
  # format :: (indent, tokens)
  lines : list[tuple[int,tuple[Token,...]]] = []
  current_start = 0
  for i,line in enumerate(text.split("\n")):
    current_end = current_start + len(line)
    stripped_line = line.lstrip()
    indent = len(line) - len(stripped_line)
    toks = _scanLine(stripped_line,i,current_start,current_end)
    if toks:
      lines.append((indent,toks))
    current_start = current_end + 1
  # now we chunk the text
  #  this takes the lines, and then builds up a nested structure of lines
  # We'll accomplish this with a stack
  if not lines:
    return ()
  indent_stack : list[int] = [lines[0][0]]
  struct : list[tuple[tuple[Token,...],list]] = [[]]
  
  for ind,toks in lines:
    if ind > indent_stack[-1]:
      indent_stack.append(ind)
      struct.append(struct[-1][-1][1])
    elif ind < indent_stack[-1]:
      stackref = indent_stack[:]
      while indent_stack and indent_stack[-1] != ind:
        indent_stack.pop()
        struct.pop()
      if not indent_stack:
        host = toks[0]
        possible = min(stackref,key=lambda x: x - ind)
        throwError(host.loc,"Indent Mismatch, did you mean an indent of "\
                           f"{possible} spaces")
    struct[-1].append((toks,[]))

  return struct[0]

##### Everything Below This Line is for testing  #####
############### and is prone to change ###############

def disScan(struct : tuple[tuple[Token,...],list],toplev=True):
  out = []
  for tok,body in struct:
    out.append(" ".join(map(repr,tok)))
    out.append(1)
    out.extend(disScan(body,False))
    out.append(-1)
  if toplev:
    buff = ""
    ci = 0
    for i in out:
      if isinstance(i,int):
        ci += i
      elif isinstance(i,str):
        buff+= "\t"*ci + i + "\n"
    return buff.expandtabs(4)
  else: return out

def testError(text : str):
  global script_struct
  try:
    script_struct = chunk(text)
    print(script_struct)
    print(disScan(script_struct))
  except dallError as de:
    print("Error:"+de.msg,file=sys.stderr)
    print(text[de.loc.lstart:de.loc.lend],file=sys.stderr)
    print(" "*de.loc.column+"~"*max(1,de.loc.endcol-de.loc.column),file=sys.stderr)
    

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
def main(scrpt):
  global _pglob,_rglob
  def dallExtend(f,x): return pyProxy(f.target,f.nargs,f.args + (x,))
  _tglob = ChainMap({NameVal("extend"):pyProxy(dallExtend,2,())})
  

if __name__ == "__main__":
  main("print 5")
  
