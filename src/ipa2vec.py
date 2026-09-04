"""
Convert IPA characters to feature vectors using CLTS and SoundVectors.
Please note:
    - The feature vectors are generated using the CLTS broad IPA inventory.
    - Diacritics are stripped if the character is not recognized.
    - Special tokens '[PAD]' and '[MASK]' are handled separately and hardcoded.
"""

import urllib.request
import zipfile
import io
from pathlib import Path
from pyclts import CLTS
from soundvectors import SoundVectors
import unicodedata
import warnings

version = "2.3.0"
target = Path("data/clts")
repo_path = target / f"clts-{version}"

if not repo_path.exists():
    url = f"https://github.com/cldf-clts/clts/archive/refs/tags/v{version}.zip"
    print(f"Downloading CLTS {version}…")

    with urllib.request.urlopen(url) as r:
        with zipfile.ZipFile(io.BytesIO(r.read())) as z:
            z.extractall(target)

clts = CLTS(repo_path)
bipa = clts.bipa # broad IPA
sv_ipa = SoundVectors(ts=bipa)


def strip_diacritics(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn').lower()

def get_feature_vector(char, sv_ipa=sv_ipa, vec_len=39):
    if char == '[PAD]':
        return tuple([3] * vec_len)
    if char == '[MASK]':
        return tuple([4] * vec_len)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        featvec = sv_ipa([char])[0]
        if not featvec:
            # Get closest base character if unknown diacritics are present
            char = strip_diacritics(char)
            featvec = sv_ipa([char[0]])[0]
        if featvec:
            featvec = [x+1 for x in list(featvec)] # Must be positive for classification
            return tuple(featvec)
        else:
            # If still no feature vector, return a "NA" vector
            return tuple([1] * vec_len)

def tokenizer(vocab):
    char2featvec = {char: get_feature_vector(char) for char in vocab}
    return char2featvec