from pathlib import Path

# 无声调音素字典，基本等价于X-SAMPA(Vocaloid)方案
# 声母是单纯辅音，介母向韵母方向黏着

LEXDICT_FILE = Path(__file__).absolute().parent / 'phonodict_cn.csv'

# NOT: hard coded accroding to `symbol.py``
_pad = '_'


class Phonodict4:

  def __init__(self, fp=LEXDICT_FILE, vac_sym=_pad):
    self.entry      = { }  # {'hui': 'h uEi _'}, 1 py = 3 syl
    self.initials   = []  # pinyin := initial + final
    self.finals     = []
    self.consonants = []  # syl4 := consonant + vowel + (tone) + ending
    self.vowels     = []
    self.endings    = ['_N', '_NG', '_R']
    self.vacant     = vac_sym       # '_', for zero consonant/ending

    self.dict_fp = fp
    self._load_dict()

  def _load_dict(self):
    I, F, C, V = set(), set(), set(), set()

    with open(self.dict_fp) as fh:
      ilist = list(fh.readline().strip().split(',')[1:])
      ilist[0] = ''   # fix '-' to ''
      for row in fh.readlines():
        ls = row.strip().split(',')
        f, cvlist = ls[0], ls[1:]
        F.add(f)
        for i in ilist:
          I.add(i)
          cv = cvlist[ilist.index(i)]
          if cv:      # found a valid syllble
            if cv == 'R':
              c = self.vacant
              v = 'e'
              e = '_R'
            else:
              if cv.endswith('ng'):
                cv = cv[:-2] ; e = '_NG'
              elif cv.endswith('n'):
                cv = cv[:-1] ; e = '_N'
              else:
                e = self.vacant
              if ' ' in cv:             # 'k ua'
                c, v = cv.split(' ')
              else:
                c = self.vacant ; v = cv
            C.add(c) ; V.add(v)
            self.entry[i + f] = [c, v, e]
      
      self.initials   = sorted(list(I))
      self.finals     = sorted(list(F))
      self.consonants = sorted(list(C))
      self.vowels     = sorted(list(V))
  
  def __getitem__(self, py:str) -> str:
    return self.entry.get(py, None)

  def __len__(self) -> int:
    return len(self.entry)
  
  @property
  def vacant_symbol(self) -> str:
    return self.vacant

  def inspect(self):
    print(f'syllable count: {len(self.entry)}')
    print(f'initials({len(self.initials)}): {self.initials}')
    print(f'finals({len(self.finals)}): {self.finals}')
    print(f'consonants({len(self.consonants)}): {self.consonants}')
    print(f'vowels({len(self.vowels)}): {self.vowels}')
    print(f'endings({len(self.endings)}): {self.endings}')


phonodict = Phonodict4() 


if __name__ == '__main__':
  phonodict.inspect()
