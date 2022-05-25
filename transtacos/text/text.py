import re
from typing import List, Union

import hparam as hp
from .symbols import symbol_to_id, id_to_symbol
from .g2p import to_syl4


_whitespace_re = re.compile(r'\s+')


def text_to_phoneme(text:str) -> str:
  # clean up
  text = text.strip()
  text = text.lower()
  text = re.sub(_whitespace_re, ' ', text)
  
  # g2p
  _converter_mapping = {
   'seq':    lambda _: _,   # => 'str'
   'syl4':   to_syl4,       # => [C, V, T, Vx]
  }
  phs = _converter_mapping[hp.g2p](text)
  return phs


def phoneme_to_sequence(phoneme:Union[str, List]) -> List[int]:
  return [symbol_to_id(ph) for ph in phoneme]


def sequence_to_phoneme(sequence:List[int]) -> str:
  return ''.join([id_to_symbol(id) for id in sequence])
