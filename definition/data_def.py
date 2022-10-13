from dataclasses import dataclass, field
from typing import List

#### JSON
@dataclass
class Res_ne:
    id: str = ""
    word: str = ""
    label: str = ""
    begin: int = ""
    end: int = ""

@dataclass
class Res_results:
    id: str = ""
    text: str = ""
    ne_list: List[Res_ne] = field(default_factory=list)

#### Mecab POS
@dataclass
class Tok_Pos:
    tokens: List[str] = field(default=list)
    pos: List[str] = field(default=list)
    ne: str = "O"

@dataclass
class Mecab_Item:
    word: str = ""
    ne: str = "O"
    tok_pos_list: List[Tok_Pos] = field(default=list)