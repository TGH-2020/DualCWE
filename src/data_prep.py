from torch.utils.data import Dataset, Sampler
import random
from collections import defaultdict
import torch
import pandas as pd
from lingpy.sequence.sound_classes import tokens2class


def data_preprocessing(forms, ranked_concepts=None, to_rank=None, token_type="asjp"):
    """
    Preprocess lexical forms into model-ready tuples.

    Args:
        forms: DataFrame with columns Glottocode, Concepticon_Gloss, Family,
               Segments_cleaned (whitespace-separated IPA), ASJP
        ranked_concepts: optional DataFrame with Concepticon_Gloss and Rank columns
        to_rank: keep only the top-N ranked concepts; None = keep all (default). Example: to_rank=105 to use the same subset of concepts as in Jäger (2026).
        token_type: one of "ipa", "sca", "dolgo", "asjp", "featvecs"
            - "ipa": split Segments_cleaned into IPA segments (list of str)
            - "featvecs": same as "ipa"; feature vectors computed later in LexicalDataset
            - "asjp": use ASJP column as a string (each char is a segment)
            - "sca": apply tokens2class(model="sca") to Segments_cleaned segments
            - "dolgo": apply tokens2class(model="dolgo") to Segments_cleaned segments

    Returns:
        data: list of (lang_id, concept, form, family) tuples
        vocab: sorted list of unique segment symbols
        langs: sorted list of unique Glottocodes present in data
        concs: sorted list of unique concepts present in data
        langs2family: dict mapping Glottocode to family name
    """
    use_asjp = token_type == "asjp"
    use_class = token_type in ("sca", "dolgo")

    # Keep only unique Glottocode-Concepticon_Gloss pairs; this collapses dialects sharing a Glottocode to a single languoid -> something to look into
    forms = forms.drop_duplicates(subset=["Glottocode", "Concepticon_Gloss"])

    # Build concept filter
    all_concs = set(forms["Concepticon_Gloss"].dropna().unique())
    if ranked_concepts is not None and to_rank is not None:
        filtered = ranked_concepts[ranked_concepts["Concepticon_Gloss"].isin(all_concs)]
        keep_concepts = set(filtered.sort_values("Rank").head(to_rank)["Concepticon_Gloss"])
    else:
        keep_concepts = None

    langs2family = {}
    vocab = set()
    data = []

    for _, row in forms.iterrows():
        lang_id = row["Glottocode"]
        concept = row["Concepticon_Gloss"]
        family = row["Family"]

        if pd.isna(lang_id) or pd.isna(concept):
            continue
        if keep_concepts is not None and concept not in keep_concepts:
            continue

        if use_asjp:
            form = row["ASJP"]
            if pd.isna(form) or len(str(form)) == 0:
                continue
            form = str(form)
        else:
            raw = row.get("Segments_cleaned", None)
            if raw is None or pd.isna(raw) or len(str(raw).strip()) == 0:
                continue
            segments = str(raw).split()
            # Handle alternate transcriptions separated by "/"; take last option by default
            segments = [s.split("/")[-1] for s in segments if s]
            if not segments:
                continue
            if use_class:
                form = tokens2class(segments, model=token_type)
            else:
                form = segments  # IPA or featvecs (uses IPA): plain list of IPA symbols

        vocab.update(form)
        langs2family[lang_id] = family
        data.append((lang_id, concept, form, family))

    langs = sorted(set(entry[0] for entry in data))
    concs = sorted(set(entry[1] for entry in data))
    return data, sorted(vocab), langs, concs, langs2family


class LexicalDataset(Dataset):
    def __init__(self, data, char2idx, conc2idx, lang2idx, to_fv=False):
        """
        Args:
            data: list of (lang_id, concept, form, family) where form is a
                        str (ASJP) or list of str (IPA/SCA/Dolgo/featvecs)
            char2idx: mapping segment to index
            conc2idx: mapping concept to index
            lang2idx: mapping glottocode to index
            to_fv: if True, also return phonological feature vectors (requires ipa2vec; only valid for token_type in {"ipa", "featvecs"}); disabling speeds up data loading for other token types
        """
        self.data = data
        self.char2idx = char2idx
        self.conc2idx = conc2idx
        self.lang2idx = lang2idx
        self.to_fv = to_fv
        if to_fv:
            from src.ipa2vec import tokenizer
            self.char2featvec = tokenizer(char2idx.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        lang, concept, word, _ = self.data[idx]
        lang_token = torch.tensor(self.lang2idx[lang], dtype=torch.long)
        concept_token = torch.tensor(self.conc2idx[concept], dtype=torch.long)
        chars = torch.tensor([self.char2idx[c] for c in word], dtype=torch.long)

        if self.to_fv:
            char_featvecs = torch.tensor(
                [self.char2featvec[c] for c in word], dtype=torch.float32
            )
        else:
            char_featvecs = torch.tensor([], dtype=torch.float32)

        word_str = "".join(word) if isinstance(word, list) else word

        return {
            "lang_token": lang_token,
            "concept_token": concept_token,
            "word": word_str,
            "chars": chars,
            "char_featvecs": char_featvecs,
        }


# Batch sampler
class LangConcBatchSampler(Sampler):
    """
    Yields batches of N*K indices
    """

    def __init__(self, dataset, N: int, K: int, num_batches: int = None):
        self.N = N
        self.K = K

        lang_concept_index = defaultdict(lambda: defaultdict(list))
        for idx, sample in enumerate(dataset):
            lang    = sample["lang_token"].item()
            concept = sample["concept_token"].item()
            lang_concept_index[lang][concept].append(idx)

        self.lang_concept_index = {
            lang: dict(concepts)
            for lang, concepts in lang_concept_index.items()
        }
        self.langs = list(self.lang_concept_index.keys())

        min_concepts = min(len(c) for c in self.lang_concept_index.values())
        if K > min_concepts:
            print(
                f"[LangConcBatchSampler] Warning: K={K} exceeds the minimum number of concepts ({min_concepts}) for some languages. Concepts will be sampled with replacement for those languages."
            )


        if len(self.langs) < N:
            raise ValueError(
                f"Only {len(self.langs)} languages in dataset but N={N}. "
                f"Reduce N."
            )

        self.num_batches = num_batches or (len(self.langs) // N)

    def __iter__(self):
        for _ in range(self.num_batches):
            sampled_langs = random.sample(self.langs, self.N)
            indices = []
            for lang in sampled_langs:
                concepts = list(self.lang_concept_index[lang].keys())
                # sample with replacement only for languages with < K concepts
                if len(concepts) >= self.K:
                    sampled_concepts = random.sample(concepts, self.K)
                else:
                    sampled_concepts = random.choices(concepts, k=self.K)
                for concept in sampled_concepts:
                    idx = random.choice(self.lang_concept_index[lang][concept])
                    indices.append(idx)
            yield indices

    def __len__(self):
        return self.num_batches

# Collate function for the data loader
def _pad_featvecs(feat_list, batch_size, max_length, pad_value=3):
    feat_dim = feat_list[0].size(-1)
    padded = torch.full((batch_size, max_length, feat_dim), pad_value, dtype=torch.float32)
    for i, fv in enumerate(feat_list):
        padded[i, : fv.size(0)] = fv
    return padded

def collate_fn(batch):
    lang_tokens = torch.stack([item["lang_token"] for item in batch])
    concept_tokens = torch.stack([item["concept_token"] for item in batch])
    words = [item["word"] for item in batch]
    chars = [item["chars"] for item in batch]
    char_featvecs = [item["char_featvecs"] for item in batch]

    padded_chars = torch.nn.utils.rnn.pad_sequence(
        chars, batch_first=True, padding_value=0
    )

    if char_featvecs[0].numel() > 0:
        max_len = max(fv.size(0) for fv in char_featvecs)
        padded_fv = _pad_featvecs(char_featvecs, len(batch), max_len)
    else:
        padded_fv = torch.tensor([], dtype=torch.float32)

    return {
        "lang_tokens": lang_tokens,
        "concept_tokens": concept_tokens,
        "words": words,
        "chars": padded_chars,
        "char_featvecs": padded_fv,
    }
