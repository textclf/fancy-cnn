import string
import sys
sys.path.append('../')
from util.misc import memoize

class CharMapper(object):
    '''
    Class the maps characters to integer codes
    NOTE that the number zero is for padding, and thus, is mapped
    to by no character. The space character is NOT recognized
    Example:
    ---------
    >>> cm = CharMapper()
    >>> cm[['My dog.', 'My cat.']]
    [[23, 61, 95, 40, 51, 43, 76], [23, 61, 95, 39, 37, 56, 76]]
    >>> cm[cm[['My dog.', 'My cat.']]]
    [['M', 'y', '<unk>', 'd', 'o', 'g', '.'], 
     ['M', 'y', '<unk>', 'c', 'a', 't', '.']]
    '''
    ALLOWED_CHARS = [ch for ch in (string.digits + string.letters + string.punctuation + ' ')] + ['<word>', '</word>']
    def __init__(self):
        super(CharMapper, self).__init__()
        self.n_chars = len(self.ALLOWED_CHARS)

        self._c2i = {ch : (i + 1) for i, ch in enumerate(self.ALLOWED_CHARS)}
        self._c2i.update({'<unk>' : self.n_chars + 1})
        
        self._i2c = {(i + 1) : ch for i, ch in enumerate(self.ALLOWED_CHARS)}
        self._i2c.update({self.n_chars + 1 : '<unk>'})
        self._i2c.update({0 : '<blank>'})

    def i2c(self, i):
        try:
            return self._i2c[i]
        except KeyError:
            return '<unk>'
    def c2i(self, c):
        try:
            return self._c2i[c]
        except KeyError:
            return self.n_chars + 1

    @memoize
    def __getitem__(self, o):
        if isinstance(o, int):
            return self.i2c(o)
        if isinstance(o, str) or isinstance(o, unicode):
            return [self._c2i['<word>']] + [self.c2i(ch) for ch in o] + [self._c2i['</word>']]
        # if isinstance(o, list):
        if hasattr(o, '__iter__'):
            return [self.__getitem__(so) for so in o]
            
    def get_indices(self, o):
        return self.__getitem__(o)