import torch
import torch.nn as nn
from x_transformers import Encoder

class DualCWE(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 num_concepts,
                 num_languages,
                 feat_dim=39, 
                 embedding_dim=256, 
                 num_layers=1, 
                 nheads=4, 
                 encoder_dropout=0.1, 
                 input_dropout=0.1,
                 swap_p=0.0, 
                 dup_p=0.1, 
                 noise_p=0.5, 
                 del_p=0.0,
                 langs_per_batch=32,
                 concepts_per_lang=20):
        super().__init__()
        self.swap_p = swap_p
        self.dup_p = dup_p
        self.noise_p = noise_p
        self.del_p = del_p
        self.representation_dim = embedding_dim # for compatibility with evaluation scripts
        self.langs_per_batch = langs_per_batch
        self.concepts_per_lang = concepts_per_lang

        self.feat_proj = nn.Linear(feat_dim, embedding_dim)
        self.pad_token_id = 0  # Assuming 0 is the padding index for token IDs
        self.char_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.pad_token_id)
        self.concept_embedding = nn.Embedding(num_concepts, embedding_dim)
        self.language_embedding = nn.Embedding(num_languages, embedding_dim)
        
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(p=input_dropout)

        # Encoder from x-transformers. This is very modular and has lots of different options to try out.
        # See https://github.com/lucidrains/x-transformers
        self.word_encoder = Encoder(
            dim=embedding_dim,
            depth=num_layers,
            heads=nheads,
            rotary_pos_emb=True,
            use_rmsnorm=False,
            ff_glu=False,
            pre_norm=False,
            attn_flash=True,
            attn_dropout=encoder_dropout
        )

        self.lang_encoder = Encoder(
                    dim=embedding_dim,
                    depth=num_layers,
                    heads=nheads,
                    rotary_pos_emb=True, # Counterintuitive to use positional embeddings for language-level encoder, but it seems to help a bit
                    use_rmsnorm=False,
                    ff_glu=False,
                    pre_norm=False,
                    attn_flash=True,
                    attn_dropout=encoder_dropout
                )

        # Projection head for contrastive learning
        # Adapted from https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py 
        self.proj_head_word = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh()
        )
        self.proj_head_lang = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh()
        )

    def _embed_input(self, x):
        """Return (embedded, key_padding_mask) for token IDs or feature vectors."""
        if x.ndim == 3:  # feature vectors (B, S, feat_dim)
            embedded = self.feat_proj(x)
            mask = x[:, :, 0] != 3  # 3 is the feature-vector padding index
        else:  # token IDs (B, S)
            embedded = self.char_embedding(x)
            mask = x != self.pad_token_id
        return embedded, mask.unsqueeze(-1) 

    def swap_positions(self, x, mask):
        """Swap two adjacent positions in the sequence, independently for each batch item."""
        if not self.training or self.swap_p == 0.0:
            return x
        B = x.size(0)
        nvalid_pos = mask.squeeze(-1).sum(dim=1)
        do_swap = (torch.rand(B, device=x.device) < self.swap_p) & (nvalid_pos > 3)
        if not do_swap.any():
            return x
        # Vectorized swap
        max_idx = nvalid_pos - 3 # do not select last position
        rand_idx = (torch.ceil(torch.rand(B, device=x.device) * (max_idx + 1))).long()
        # batch indices for items that will swap
        b_idx = torch.arange(B, device=x.device)[do_swap]
        pos = rand_idx[do_swap]
        # swap x[b, pos] <-> x[b, pos+1]
        tmp = x[b_idx, pos].clone()
        x[b_idx, pos] = x[b_idx, pos + 1]
        x[b_idx, pos + 1] = tmp
        return x
    
    def duplicate_chars(self, x, mask):
        """Duplicate random characters in the sequence, independently for each batch item.
        Remove last padding position if duplicating a character.
        """
        if not self.training or self.dup_p == 0.0:
            return x, mask
        B, L = x.shape[:2]
        nvalid_pos = mask.squeeze(-1).sum(dim=1)
        do_dup = (torch.rand(B, device=x.device) < self.dup_p) & (nvalid_pos < L)
        if not do_dup.any():
            return x, mask
        # Vectorized duplication
        max_idx = nvalid_pos - 1 
        rand_idx = (torch.floor(torch.rand(B, device=x.device) * (max_idx + 1))).long()

        base = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        shifted = base.clone()

        # Shift everything after the duplication position left by 1 (drop last item)
        mask_shift = (base > rand_idx.unsqueeze(1)) & do_dup.unsqueeze(1)
        shifted[mask_shift] -= 1

        # Gather duplicated sequence
        x_dup = x.gather(1, shifted.unsqueeze(-1).expand_as(x))

        # Adjust masking for padding
        new_valid_pos = nvalid_pos + do_dup.long()
        mask = base < new_valid_pos.unsqueeze(1)

        return x_dup, mask.unsqueeze(-1)

    def delete_chars(self, x, mask):
        """Delete random characters in the sequence, independently for each batch item.
        Add padding at the end to maintain sequence length.
        """
        if not self.training or self.del_p == 0.0:
            return x, mask
        B, L = x.shape[:2]
        nvalid_pos = mask.squeeze(-1).sum(dim=1)
        do_del = (torch.rand(B, device=x.device) < self.del_p) & (nvalid_pos > 3)
        if not do_del.any():
            return x
        # Vectorized deletion
        rand_idx = (torch.floor(torch.rand(B, device=x.device) * (nvalid_pos))).long()
        base = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        shifted = base.clone()
        # Shift everything after the deletion position right by 1 (add dummy at the end to replace with padding)
        mask_shift = (base >= rand_idx.unsqueeze(1)) & do_del.unsqueeze(1)
        temp_mask = torch.cat([torch.zeros(B,L-1, dtype=torch.bool, device=x.device), mask_shift[:,-1:]], dim=-1)
        shifted[mask_shift] += 1
        shifted[temp_mask] = 0  # Last position will be masked out later

        # Gather deleted sequence
        x_del = x.gather(1, shifted.unsqueeze(-1).expand_as(x))
        
        # Adjust masking for padding
        mask = mask.squeeze(-1) | temp_mask
        return x_del, mask.unsqueeze(-1)

    def add_noise(self, x):
        """Add Gaussian noise to a portion of the batch and sequence."""
        if not self.training or self.noise_p == 0.0:
            return x
        B, L = x.size(0), x.size(1)
        n_mask = (torch.rand(B, L, 1, device=x.device) < self.noise_p).float()
        noise = torch.randn_like(x) * 0.3
        x = x + noise * n_mask
        return x
    
    def forward(self, x, c=None, l=None):
        w_enc = self.encode_word(x, c=c)           # (N*K, emb_dim)
        l_enc = self.encode_lang(w_enc, l=l)       # (N, emb_dim)
        z_word = self.proj_head_word(w_enc)
        z_lang = self.proj_head_lang(l_enc)
        return z_word, z_lang

    def encode_word(self, x, c=None):
        embedded, mask = self._embed_input(x)
        embedded = self.norm(embedded)
        embedded = self.dropout(embedded)
        
        
        # Data augmentations
        embedded, mask = self.delete_chars(embedded, mask)
        embedded = self.swap_positions(embedded, mask)
        embedded, mask = self.duplicate_chars(embedded, mask)
        embedded = self.add_noise(embedded)

        # Add concept embedding if provided
        if c is not None:
            embedded = embedded + self.concept_embedding(c).unsqueeze(1)

        embedded = embedded * mask.float()
        
        # Transformers encoder
        h = self.word_encoder(embedded, mask=mask.squeeze(-1).bool())

        # Masked mean pooling
        x_masked = h * mask
        h = x_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        return h

    def encode_lang(self, h, l=None):
        # Batches are formatted as (N*K, emb_dim) where N = langs_per_batch, K = concepts_per_lang
        # We pass all words of a language through here and average the resulting representations.
        # For each language, we either average only the word representations or the word representations with language embeddings added position-wise.
        B = h.size(0)
        N = self.langs_per_batch
        K = self.concepts_per_lang
        # Randomize the order of concepts for each language
        if l is not None:
            lang_emb = self.language_embedding(l)  # [B, emb_dim]
            h = h + lang_emb  # [B, emb_dim]
            # Reformat h to (N, K, emb_dim) and pass through language encoder
            h = h.view(N, K, -1)  # [N, K, emb_dim]
            h = self.lang_encoder(h)  # [N, K, emb_dim]
            h = h.mean(dim=1)  # [N, emb_dim]
        else:
            h = h.view(N, K, -1)  # [N, K, emb_dim]
            h = self.lang_encoder(h)  # [N, K, emb_dim]
            h = h.mean(dim=1)  # [N, emb_dim]
        return h
    
    def get_representations(self, x, c=None, l=None): # l kept for compatibility with evaluation scripts
        self.eval()
        with torch.no_grad():
            h = self.encode_word(x, c=c)
        return h

