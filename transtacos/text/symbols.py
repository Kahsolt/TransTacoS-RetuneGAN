# marks
_pad     = '_'    # <PAD>, right padding for fixed-length RNN;
                  # <SIL>/<SP>, short silence / break of speech
_eos     = '~'    # <EOS>, end of sentence
_sep     = '/'    # separtor between syllables
_unk     = '?'    # <UNK>

_markers = [_pad, _eos, _sep, _unk]   # NOTE: `_pad` MUST be at index 0


''' G2P = seq '''
_chars = 'abcdefghijklmnopqrstuvwxyz 12345'


''' G2P = syl4 '''
# phonetic unit under syllable repr refer to `phonodict_cn.txt`
# syl4 := CxVTx = C + V + T + Vx
# _syl4 == [
#   '-', 'b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'x', 'z', 'zh',
#   'Ei', 'R', 'a', 'ai', 'ao', 'e', 'i', 'i0', 'iE', 'iR', 'ia', 'iao', 'io', 'iou', 'o', 'ou', 'u', 'uEi', 'ua', 'uai', 'ue', 'uo', 'v', 'vE',
#   '0', '1', '2', '3', '4', '5',
#   '_N', '_NG', '_R',
# ]
# TODO: 是否合并, 'i'/'iR' - 'i0'
from .phonodict_cn import phonodict
#_syl4_T  = ['0', '1', '2', '3', '4', '5']        # 6, NOTE: exclude from phone table
#_syl4_P  = ['0', '1', '2', '3', '4', '5']        # 6, NOTE: exclude from phone table
_syl4_C  = phonodict.consonants                  # 22
_syl4_V  = phonodict.vowels                      # 24
_syl4_Vx = phonodict.endings                     # 3
_syl4 = _syl4_C + _syl4_V + _syl4_Vx   # 54


# phonetic unit list
_g2p_mapping = {
  'seq':    _chars,
  'syl4':   _syl4,
}

import hparam as hp

assert len(set(_g2p_mapping[hp.g2p])) == len(_g2p_mapping[hp.g2p])        # assure no duplicates
_symbols = _markers + sorted(set(_g2p_mapping[hp.g2p]) - set(_markers))   # keep order
print(f'[Symbols] collect {len(_symbols)} symbols in {hp.g2p} repr')
print(f'    {_symbols}')

_symbol_to_id = {s: i for i, s in enumerate(_symbols)}
_id_to_symbol = {i: s for i, s in enumerate(_symbols)}


def symbol_to_id(sym:str) -> int:
  return _symbol_to_id.get(sym, _symbol_to_id[_unk])


def id_to_symbol(id:int) -> str:
  return _id_to_symbol.get(id, _unk)


def get_vocab_size():
  return len(_symbols)


def get_symbol_id(s:str):
  return {
    'pad': symbol_to_id(_pad),
    'eos': symbol_to_id(_eos),
    'sep': symbol_to_id(_sep),
    'unk': symbol_to_id(_unk),
    'vac': symbol_to_id(phonodict.vacant_symbol),     # := _pad
  }.get(s, symbol_to_id(s))
