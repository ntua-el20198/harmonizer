import os
# --- Paths (override with environment variables; defaults = original Kaggle layout) ---
DATA_DIR = os.environ.get("HARMONIZER_DATA", "/kaggle/input/datasets/kostissp/music-ontology-data")    # final_corpus.pkl, symbolic_lookup.pkl
DATA_DIR1 = os.environ.get("HARMONIZER_DATA1", "/kaggle/input/datasets/kostissp/music-ontology-data1")  # extracted_*.json
OUT_DIR = os.environ.get("HARMONIZER_OUT", "/kaggle/working")                                          # checkpoints / results
import pickle
import math
import re
import json
import random
import copy
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Συσκευή: {device}")

# =============================================================================
# compare_ablation.py - ΣΥΓΚΡΙΣΗ ΟΛΩΝ ΤΩΝ VARIANTS ΤΗΣ ΜΕΛΕΤΗΣ ABLATION
# Ακρίβεια (Root/Quality/Joint), οι 9 μετρικές των papers, τήρηση των κανόνων,
# διάφορα γραφήματα και αρχεία CSV. Οι ρυθμίσεις (checkpoints, διαδρομές) βρίσκονται
# στο μπλοκ "ΡΥΘΜΙΣΕΙΣ" προς το τέλος του αρχείου.
# =============================================================================

# --- ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ (ΙΔΙΕΣ για όλα τα bundles - δεν εξαρτώνται από τη
# σύνθεση του corpus) ---
PITCH_LETTER_SEMITONE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
PITCH_NAME_RE = re.compile(r'^Pitch_([A-G])(_sharp_|_flat_)?(-?\d+)$')
PC_TO_SHARP_NAME = ['C', 'C_sharp_', 'D', 'D_sharp_', 'E', 'F', 'F_sharp_', 'G', 'G_sharp_', 'A', 'A_sharp_', 'B']
PC_TO_DISPLAY = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def pitch_name_to_semitone(name):
    if not name:
        return None
    m = PITCH_NAME_RE.match(name)
    if not m:
        return None
    letter, accidental, octave = m.group(1), m.group(2), int(m.group(3))
    base = PITCH_LETTER_SEMITONE[letter]
    if accidental == '_sharp_':
        base += 1
    elif accidental == '_flat_':
        base -= 1
    return base + 12 * octave

def transpose_pitch_name(name, shift):
    m = PITCH_NAME_RE.match(name)
    if not m:
        return name
    letter, accidental, octave = m.group(1), m.group(2), int(m.group(3))
    base = PITCH_LETTER_SEMITONE[letter] + (1 if accidental == '_sharp_' else (-1 if accidental == '_flat_' else 0))
    absolute = base + 12 * octave + shift
    new_octave, new_pc = divmod(absolute, 12)
    return f"Pitch_{PC_TO_SHARP_NAME[new_pc]}{new_octave}"

MODIFIER_RE = re.compile(r'\b(add|alter|omit)\b.*', re.IGNORECASE)
def simplify_chord_name(name):
    simplified = MODIFIER_RE.split(name)[0].strip()
    return simplified if simplified else name.strip()

ROOT_RE = re.compile(r'^([A-G])(_sharp_|_flat_)?')
PITCH_CLASS = {
    'C': 0, 'C_sharp_': 1, 'D_flat_': 1, 'D': 2, 'D_sharp_': 3, 'E_flat_': 3,
    'E': 4, 'F': 5, 'F_sharp_': 6, 'G_flat_': 6, 'G': 7, 'G_sharp_': 8,
    'A_flat_': 8, 'A': 9, 'A_sharp_': 10, 'B_flat_': 10, 'B': 11,
}
SOLFEGE_PREFIXES = [('Sol', 'G'), ('Fa', 'F'), ('Re', 'D'), ('Mi', 'E'),
                     ('La', 'A'), ('Si', 'B'), ('Do', 'C'), ('H', 'B')]
QUALITY_WHITELIST = {
    'maj', 'm', '7', 'm7', 'maj7', 'dim', '6', '9', 'ø7', 'sus', 'm6', 'o7',
    'aug', '7aug', 'm9', 'power', '13', 'sus2', 'M9', 'm11', 'mM7', '11', 'NC',
}
QUALITY_ALIASES = {'N.C.': 'NC', '+': 'aug', '7+': '7aug', 'M': 'maj'}
def canonicalize_quality(q):
    q = QUALITY_ALIASES.get(q, q)
    return q if q in QUALITY_WHITELIST else 'OTHER'

def parse_root_quality_raw(chord_name):
    main = chord_name.split('/')[0].strip()
    for prefix, letter in SOLFEGE_PREFIXES:
        if main.startswith(prefix):
            rest = main[len(prefix):].strip()
            quality = rest if rest else 'maj'
            return PITCH_CLASS[letter], canonicalize_quality(quality)
    m = ROOT_RE.match(main)
    if not m:
        return 'UNKNOWN_ROOT', canonicalize_quality(main if main else 'UNKNOWN_QUALITY')
    root_token = m.group(0)
    quality = main[len(root_token):].strip()
    if quality == '':
        quality = 'maj'
    root_pc = PITCH_CLASS.get(root_token, None)
    if root_pc is None:
        return 'UNKNOWN_ROOT', canonicalize_quality(quality)
    return root_pc, canonicalize_quality(quality)

KS_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
def _pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((x - mb) ** 2 for x in b) ** 0.5
    return 0.0 if da == 0 or db == 0 else num / (da * db)
def detect_key(pitch_classes):
    hist = [0.0] * 12
    for pc in pitch_classes:
        hist[pc] += 1.0
    if sum(hist) == 0:
        return 0, 'major'
    best = (-2.0, 0, 'major')
    for tonic in range(12):
        rotated = [hist[(tonic + k) % 12] for k in range(12)]
        for mode, profile in (('major', KS_MAJOR_PROFILE), ('minor', KS_MINOR_PROFILE)):
            c = _pearson(rotated, profile)
            if c > best[0]:
                best = (c, tonic, mode)
    return best[1], best[2]

# =============================================================================
# load_corpus_and_vocab_v2: υποστηρίζει CMD, HookTheory και POP909 ΑΝΕΞΑΡΤΗΤΑ
# (κάθε ένα προαιρετικό, με το δικό του flag), ώστε να μπορεί να αναπαραστήσει
# ΟΠΟΙΑΔΗΠΟΤΕ από τις 4 συνθέσεις corpus που έχεις πραγματικά εκπαιδεύσει.
# =============================================================================
def load_corpus_and_vocab_v2(include_hooktheory=False, hooktheory_target=7000,
                              include_pop909=False, include_cmd=True, seed=42):
    with open(CORPUS_PATH, 'rb') as f:
        raw_songs = pickle.load(f)
    with open(LOOKUP_PATH, 'rb') as f:
        lookup = pickle.load(f)
    chord_names = copy.deepcopy(lookup['chord_names'])
    note_features = copy.deepcopy(lookup['note_features'])

    base_classical_count = len(raw_songs)  # όριο ΠΡΙΝ μπει το CMD (αν μπει)

    # ΔΙΟΡΘΩΣΗ: ορίζεται ΕΔΩ, εκτός του if include_cmd, γιατί το χρειάζονται
    # ΚΑΙ τα merge blocks του HookTheory/POP909 παρακάτω - πριν έσκαγε με
    # NameError όποτε include_cmd=False αλλά include_pop909=True.
    _PC_SHARP = ['C', 'C_sharp_', 'D', 'D_sharp_', 'E', 'F', 'F_sharp_',
                 'G', 'G_sharp_', 'A', 'A_sharp_', 'B']

    # --- chord-melody-dataset merge (προαιρετικό) ---
    _synthetic_iri_to_is_major = {}
    if include_cmd:
        with open(EXTRACTED_JSON_PATH, 'r', encoding='utf-8') as f:
            _extracted = json.load(f)
        _new_songs = _extracted['songs']
        def _pitch_feature_to_pitch_string(pf):
            pc, octave = pf % 12, pf // 12
            return f"Pitch_{_PC_SHARP[pc]}{octave}"
        def _chord_to_raw_name(root_pc, quality):
            if quality == 'NC' or root_pc == 'UNKNOWN_ROOT':
                return 'N.C.'
            return f"{_PC_SHARP[root_pc]}{quality}"
        for _song_idx, _song in enumerate(_new_songs):
            _title_safe = _song['title'].replace(' ', '_').replace('/', '_')
            _melody_iris, _chord_iris = [], []
            for _note_idx, _note in enumerate(_song['notes']):
                _note_iri = f"LOCAL#{_song_idx}_{_title_safe}_note_{_note_idx}"
                _chord_iri = f"LOCAL#{_song_idx}_{_title_safe}_chord_{_note_idx}"
                note_features[_note_iri] = {
                    'pitch': _pitch_feature_to_pitch_string(_note['pitch_feature']),
                    'duration': _note['duration_label'], 'beat_type': _note['beat_type'],
                    'position': _note['position'], 'interval': _note['interval'],
                    'beat_strength': _note['beat_strength'],
                }
                chord_names[_chord_iri] = _chord_to_raw_name(_note['chord_root_pc'], _note['chord_quality'])
                _melody_iris.append(_note_iri)
                _chord_iris.append(_chord_iri)
            if _melody_iris:
                raw_songs.append((_melody_iris, _chord_iris))
                for _n in _melody_iris:
                    _synthetic_iri_to_is_major[_n] = _song['is_major']

    classical_song_count = len(raw_songs)  # base(+CMD αν include_cmd) - όριο πριν HookTheory/POP909

    # --- HookTheory merge (προαιρετικό) ---
    hk_title_to_iris, hk_title_to_index = {}, {}
    if include_hooktheory:
        with open(HOOKTHEORY_JSON_PATH, 'r', encoding='utf-8') as f:
            _hk_data = json.load(f)
        _hk_songs = _hk_data['songs']
        _hk_songs = [s for s in _hk_songs if 8 <= s['n_notes'] <= 327]
        _hk_rng = random.Random(seed)
        if len(_hk_songs) > hooktheory_target:
            _hk_songs = _hk_rng.sample(_hk_songs, hooktheory_target)
        def _hk_pitch_string(letter, alter, octave):
            suffix = '_sharp_' if alter == 1 else ('_flat_' if alter == -1 else ('' if alter == 0 else None))
            return f"Pitch_{letter}{suffix}{octave}" if suffix is not None else None
        def _hk_chord_to_raw_name(root_pc, quality):
            return 'N.C.' if not isinstance(root_pc, int) else f"{_PC_SHARP[root_pc]}{quality}"
        for _hk_song_idx, _hk_song in enumerate(_hk_songs):
            _hk_title_safe = _hk_song['title'].replace(' ', '_').replace('/', '_')
            _hk_melody_iris, _hk_chord_iris = [], []
            for _hk_note_idx, _hk_note in enumerate(_hk_song['notes']):
                _hk_pitch_str = _hk_pitch_string(_hk_note['letter'], _hk_note['alter'], _hk_note['octave'])
                if _hk_pitch_str is None:
                    continue
                _hk_note_iri = f"HK#{_hk_song_idx}_{_hk_title_safe}_note_{_hk_note_idx}"
                _hk_chord_iri = f"HK#{_hk_song_idx}_{_hk_title_safe}_chord_{_hk_note_idx}"
                note_features[_hk_note_iri] = {
                    'pitch': _hk_pitch_str, 'duration': _hk_note['duration_label'],
                    'beat_type': _hk_note['beat_type'], 'position': _hk_note['position'],
                    'interval': _hk_note['interval'], 'beat_strength': _hk_note['beat_strength'],
                }
                chord_names[_hk_chord_iri] = _hk_chord_to_raw_name(_hk_note['chord_root_pc'], _hk_note['chord_quality'])
                _hk_melody_iris.append(_hk_note_iri)
                _hk_chord_iris.append(_hk_chord_iri)
            if _hk_melody_iris:
                raw_songs.append((_hk_melody_iris, _hk_chord_iris))
                if _hk_song['title'] not in hk_title_to_iris:
                    hk_title_to_iris[_hk_song['title']] = (_hk_melody_iris, _hk_chord_iris)
                    hk_title_to_index[_hk_song['title']] = len(raw_songs) - 1

    pop909_start_index = len(raw_songs)  # <-- ΟΠΟΥ ξεκινάει το POP909 στο raw_songs (αν υπάρχει)
    pop_title_to_iris, pop_title_to_index = {}, {}
    # --- POP909 merge (προαιρετικό) ---
    if include_pop909:
        with open(POP909_JSON_PATH, 'r', encoding='utf-8') as f:
            _pop_data = json.load(f)
        _pop_songs = _pop_data['songs']
        def _pop_pitch_string(letter, alter, octave):
            suffix = '_sharp_' if alter == 1 else ('_flat_' if alter == -1 else ('' if alter == 0 else None))
            return f"Pitch_{letter}{suffix}{octave}" if suffix is not None else None
        def _pop_chord_to_raw_name(root_pc, quality):
            return 'N.C.' if (quality == 'NC' or root_pc == 'UNKNOWN_ROOT') else f"{_PC_SHARP[root_pc]}{quality}"
        for _pop_song_idx, _pop_song in enumerate(_pop_songs):
            _pop_title_safe = _pop_song['title'].replace(' ', '_').replace('/', '_')
            _pop_melody_iris, _pop_chord_iris = [], []
            for _pop_note_idx, _pop_note in enumerate(_pop_song['notes']):
                _pop_pitch_str = _pop_pitch_string(_pop_note['letter'], _pop_note['alter'], _pop_note['octave'])
                if _pop_pitch_str is None:
                    continue
                _pop_note_iri = f"POP909#{_pop_song_idx}_{_pop_title_safe}_note_{_pop_note_idx}"
                _pop_chord_iri = f"POP909#{_pop_song_idx}_{_pop_title_safe}_chord_{_pop_note_idx}"
                note_features[_pop_note_iri] = {
                    'pitch': _pop_pitch_str, 'duration': _pop_note['duration_label'],
                    'beat_type': _pop_note['beat_type'], 'position': _pop_note['position'],
                    'interval': _pop_note['interval'], 'beat_strength': _pop_note['beat_strength'],
                }
                chord_names[_pop_chord_iri] = _pop_chord_to_raw_name(_pop_note['chord_root_pc'], _pop_note['chord_quality'])
                _pop_melody_iris.append(_pop_note_iri)
                _pop_chord_iris.append(_pop_chord_iri)
            if _pop_melody_iris:
                raw_songs.append((_pop_melody_iris, _pop_chord_iris))
                if _pop_song['title'] not in pop_title_to_iris:
                    pop_title_to_iris[_pop_song['title']] = (_pop_melody_iris, _pop_chord_iris)
                    pop_title_to_index[_pop_song['title']] = len(raw_songs) - 1

    # --- key detection & transposition ---
    note_pitch_class_transposed = {}
    transposed_pitch_name = {}
    chord_root_pc_transposed = {}
    chord_quality_raw = {}
    song_is_major = []
    for melody_iris, chord_iris in raw_songs:
        if melody_iris and melody_iris[0] in _synthetic_iri_to_is_major:
            shift = 0
            mode = 'major' if _synthetic_iri_to_is_major[melody_iris[0]] else 'minor'
        else:
            raw_pcs = []
            for n in melody_iris:
                st = pitch_name_to_semitone(note_features.get(n, {}).get('pitch'))
                if st is not None:
                    raw_pcs.append(st % 12)
            tonic, mode = detect_key(raw_pcs)
            shift = ((0 - tonic) % 12) if mode == 'major' else ((9 - tonic) % 12)
        song_is_major.append(mode == 'major')
        for n in melody_iris:
            if n in note_pitch_class_transposed:
                continue
            pname = note_features.get(n, {}).get('pitch')
            st = pitch_name_to_semitone(pname)
            if st is None:
                note_pitch_class_transposed[n] = None
                transposed_pitch_name[n] = pname if pname else '__UNKNOWN__'
            else:
                note_pitch_class_transposed[n] = (st + shift) % 12
                transposed_pitch_name[n] = transpose_pitch_name(pname, shift)
        for c in chord_iris:
            if c in chord_root_pc_transposed:
                continue
            name = simplify_chord_name(chord_names.get(c, ''))
            root_pc, quality = parse_root_quality_raw(name)
            chord_quality_raw[c] = quality
            chord_root_pc_transposed[c] = (root_pc + shift) % 12 if isinstance(root_pc, int) else 'UNKNOWN_ROOT'

    all_notes_iris = sorted(note_pitch_class_transposed.keys())
    def build_feature_vocab(all_iris, feature_key, value_lookup=None):
        vocab, idx_map = {}, {}
        for iri in all_iris:
            val = value_lookup[iri] if value_lookup is not None else note_features.get(iri, {}).get(feature_key)
            key = val if val is not None else "__UNKNOWN__"
            if key not in vocab:
                vocab[key] = len(vocab)
            idx_map[iri] = vocab[key]
        return idx_map, len(vocab)
    pitch_oct_to_idx, NUM_PITCH_OCT_CLASSES = build_feature_vocab(all_notes_iris, None, value_lookup=transposed_pitch_name)
    duration_to_idx, NUM_DURATION_CLASSES = build_feature_vocab(all_notes_iris, 'duration')
    beat_type_to_idx, NUM_BEAT_TYPE_CLASSES = build_feature_vocab(all_notes_iris, 'beat_type')
    position_to_idx, NUM_POSITION_CLASSES = build_feature_vocab(all_notes_iris, 'position')
    interval_to_idx, NUM_INTERVAL_CLASSES = build_feature_vocab(all_notes_iris, 'interval')
    note_beat_strength = {iri: note_features.get(iri, {}).get('beat_strength', 0.0) or 0.0 for iri in all_notes_iris}
    # ΝΕΟ: αντίστροφα λεξικά index -> ΠΡΑΓΜΑΤΙΚΟ label (τα *_to_idx είναι ανά IRI νότας, ΟΧΙ ανά label)
    duration_idx_to_label, beat_type_idx_to_label = {}, {}
    for _iri in all_notes_iris:
        _d = duration_to_idx[_iri]
        if _d not in duration_idx_to_label:
            duration_idx_to_label[_d] = note_features.get(_iri, {}).get('duration') or '__UNKNOWN__'
        _b = beat_type_to_idx[_iri]
        if _b not in beat_type_idx_to_label:
            beat_type_idx_to_label[_b] = note_features.get(_iri, {}).get('beat_type') or '__UNKNOWN__'

    all_chords_iris = sorted(chord_root_pc_transposed.keys())
    root_to_idx, quality_to_idx = {}, {}
    chord_root_idx, chord_quality_idx = {}, {}
    for c in all_chords_iris:
        root_key = chord_root_pc_transposed[c]
        qual_key = chord_quality_raw[c]
        if root_key not in root_to_idx:
            root_to_idx[root_key] = len(root_to_idx)
        if qual_key not in quality_to_idx:
            quality_to_idx[qual_key] = len(quality_to_idx)
        chord_root_idx[c] = root_to_idx[root_key]
        chord_quality_idx[c] = quality_to_idx[qual_key]
    NUM_ROOT_CLASSES, NUM_QUALITY_CLASSES = len(root_to_idx), len(quality_to_idx)
    flat_key_to_idx = {}
    for c in all_chords_iris:
        combo = (chord_root_idx[c], chord_quality_idx[c])
        if combo not in flat_key_to_idx:
            flat_key_to_idx[combo] = len(flat_key_to_idx)
    NUM_CHORD_CLASSES = len(flat_key_to_idx)
    idx_to_root_pc = {idx: pc for pc, idx in root_to_idx.items()}
    idx_to_quality = {idx: q for q, idx in quality_to_idx.items()}

    song_lengths = sorted(len(m) for m, _ in raw_songs)
    p99 = song_lengths[int(len(song_lengths) * 0.99)]
    max_seq_len = max(64, int(p99 * 1.2))

    # --- ΙΔΙΟ replication του πραγματικού train/val split (seed 42, shuffle, 80/20) ---
    filtered_indices = [i for i, (m, _) in enumerate(raw_songs) if min(len(m), max_seq_len) >= 8]
    _rng_state = random.getstate()
    random.seed(42)
    shuffled_indices = filtered_indices[:]
    random.shuffle(shuffled_indices)
    random.setstate(_rng_state)
    _split_idx = int(0.8 * len(shuffled_indices))
    val_only_indices = set(shuffled_indices[_split_idx:])

    print(f"   {len(raw_songs)} τραγούδια συνολικά ({base_classical_count} classics [PDMX+EWLD], "
          f"{classical_song_count - base_classical_count} CMD, "
          f"{pop909_start_index - classical_song_count} HookTheory, "
          f"{len(raw_songs) - pop909_start_index} POP909) | {NUM_PITCH_OCT_CLASSES} pitch, "
          f"{NUM_CHORD_CLASSES} flat chord | max_seq_len={max_seq_len} | "
          f"val-only: {len(val_only_indices)}")

    return {
        'raw_songs': raw_songs, 'classical_song_count': classical_song_count,
        'note_pitch_class_transposed': note_pitch_class_transposed,
        'transposed_pitch_name': transposed_pitch_name, 'song_is_major': song_is_major,
        'duration_idx_to_label': duration_idx_to_label, 'beat_type_idx_to_label': beat_type_idx_to_label,
        'base_classical_count': base_classical_count,
        'pop909_start_index': pop909_start_index,
        'hk_title_to_iris': hk_title_to_iris, 'hk_title_to_index': hk_title_to_index,
        'pop_title_to_iris': pop_title_to_iris, 'pop_title_to_index': pop_title_to_index,
        'val_only_indices': val_only_indices,
        'pitch_oct_to_idx': pitch_oct_to_idx, 'duration_to_idx': duration_to_idx,
        'beat_type_to_idx': beat_type_to_idx, 'position_to_idx': position_to_idx,
        'interval_to_idx': interval_to_idx, 'note_beat_strength': note_beat_strength,
        'chord_root_idx': chord_root_idx, 'chord_quality_idx': chord_quality_idx,
        'idx_to_root_pc': idx_to_root_pc, 'idx_to_quality': idx_to_quality,
        'NUM_PITCH_OCT_CLASSES': NUM_PITCH_OCT_CLASSES, 'NUM_DURATION_CLASSES': NUM_DURATION_CLASSES,
        'NUM_BEAT_TYPE_CLASSES': NUM_BEAT_TYPE_CLASSES, 'NUM_POSITION_CLASSES': NUM_POSITION_CLASSES,
        'NUM_INTERVAL_CLASSES': NUM_INTERVAL_CLASSES, 'NUM_ROOT_CLASSES': NUM_ROOT_CLASSES,
        'NUM_QUALITY_CLASSES': NUM_QUALITY_CLASSES, 'NUM_CHORD_CLASSES': NUM_CHORD_CLASSES,
        'max_seq_len': max_seq_len,
    }

# --- Μοντέλο ---
class HarmonizerConcatEncoder(nn.Module):
    def __init__(self, num_pitch_classes, pad_pitch_id, num_duration_classes, pad_duration_id,
                 num_beat_type_classes, pad_beat_type_id, num_position_classes, pad_position_id,
                 num_interval_classes, pad_interval_id, num_chord_classes, mask_chord_id,
                 embed_dim, num_heads, num_layers, num_root_classes, num_quality_classes,
                 max_seq_len, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pitch_embedding = nn.Embedding(num_pitch_classes + 1, embed_dim, padding_idx=pad_pitch_id)
        self.duration_embedding = nn.Embedding(num_duration_classes + 1, embed_dim, padding_idx=pad_duration_id)
        self.beat_type_embedding = nn.Embedding(num_beat_type_classes + 1, embed_dim, padding_idx=pad_beat_type_id)
        self.position_embedding = nn.Embedding(num_position_classes + 1, embed_dim, padding_idx=pad_position_id)
        self.interval_embedding = nn.Embedding(num_interval_classes + 1, embed_dim, padding_idx=pad_interval_id)
        self.beat_strength_proj = nn.Linear(1, embed_dim)
        self.chord_context_embedding = nn.Embedding(num_chord_classes + 1, embed_dim, padding_idx=mask_chord_id)
        self.mask_chord_id = mask_chord_id
        pe = self._sinusoidal_positional_encoding(max_seq_len, embed_dim)
        self.register_buffer('shared_pos', pe)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_norm = nn.LayerNorm(embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_root = nn.Linear(embed_dim, num_root_classes)
        self.fc_quality = nn.Linear(embed_dim, num_quality_classes)

    @staticmethod
    def _sinusoidal_positional_encoding(seq_len, d_model):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _positions_for(self, L, device):
        if L <= self.max_seq_len:
            return self.shared_pos[:, :L, :]
        return self._sinusoidal_positional_encoding(L, self.embed_dim).to(device)

    def forward(self, pitch_ids, duration_ids, beat_type_ids, position_ids, interval_ids,
                beat_strength, chord_context_ids, pad_mask=None):
        B, L = pitch_ids.shape
        melody_emb = (self.pitch_embedding(pitch_ids) + self.duration_embedding(duration_ids)
                      + self.beat_type_embedding(beat_type_ids) + self.position_embedding(position_ids)
                      + self.interval_embedding(interval_ids)
                      + self.beat_strength_proj(beat_strength.unsqueeze(-1)))
        harmony_emb = self.chord_context_embedding(chord_context_ids)
        full_seq = torch.cat([melody_emb, harmony_emb], dim=1)
        pos = self._positions_for(L, pitch_ids.device)
        full_pos = torch.cat([pos, pos], dim=1)
        full_seq = full_seq + full_pos
        full_seq = self.input_norm(full_seq)
        full_seq = self.dropout(full_seq)
        full_pad_mask = torch.cat([pad_mask, pad_mask], dim=1) if pad_mask is not None else None
        encoded = self.encoder(full_seq, src_key_padding_mask=full_pad_mask)
        encoded = self.output_norm(encoded)
        harmony_encoded = encoded[:, L:, :]
        return self.fc_root(harmony_encoded), self.fc_quality(harmony_encoded)

def build_and_load_model(bundle, checkpoint_path):
    mask_chord_id = bundle['NUM_CHORD_CLASSES']
    model = HarmonizerConcatEncoder(
        num_pitch_classes=bundle['NUM_PITCH_OCT_CLASSES'], pad_pitch_id=bundle['NUM_PITCH_OCT_CLASSES'],
        num_duration_classes=bundle['NUM_DURATION_CLASSES'], pad_duration_id=bundle['NUM_DURATION_CLASSES'],
        num_beat_type_classes=bundle['NUM_BEAT_TYPE_CLASSES'], pad_beat_type_id=bundle['NUM_BEAT_TYPE_CLASSES'],
        num_position_classes=bundle['NUM_POSITION_CLASSES'], pad_position_id=bundle['NUM_POSITION_CLASSES'],
        num_interval_classes=bundle['NUM_INTERVAL_CLASSES'], pad_interval_id=bundle['NUM_INTERVAL_CLASSES'],
        num_chord_classes=bundle['NUM_CHORD_CLASSES'], mask_chord_id=mask_chord_id,
        embed_dim=512, num_heads=8, num_layers=8,
        num_root_classes=bundle['NUM_ROOT_CLASSES'], num_quality_classes=bundle['NUM_QUALITY_CLASSES'],
        max_seq_len=bundle['max_seq_len'], dropout=0.3
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict, used_key = None, None
    for _key in CHECKPOINT_KEY_PREFERENCE:
        if checkpoint.get(_key) is not None:
            state_dict, used_key = checkpoint[_key], _key
            break
    if state_dict is None:
        raise KeyError(f"Το checkpoint δεν περιέχει κανένα από τα κλειδιά {CHECKPOINT_KEY_PREFERENCE}: {list(checkpoint.keys())}")
    model.load_state_dict(state_dict)
    model.eval()
    _epoch = checkpoint.get('best_epoch') if used_key == 'best_model_state_dict' else checkpoint.get('epoch')
    print(f"   ✅ Φορτώθηκε {checkpoint_path} | βάρη: '{used_key}' | εποχή: {_epoch} | κλειδιά στο αρχείο: {list(checkpoint.keys())}")
    return model, mask_chord_id

def evaluate_song(model, mask_chord_id, bundle, melody_iris, chord_iris):
    n = len(melody_iris)
    pitch_ids = [bundle['pitch_oct_to_idx'].get(x, bundle['NUM_PITCH_OCT_CLASSES']) for x in melody_iris]
    duration_ids = [bundle['duration_to_idx'].get(x, bundle['NUM_DURATION_CLASSES']) for x in melody_iris]
    beat_type_ids = [bundle['beat_type_to_idx'].get(x, bundle['NUM_BEAT_TYPE_CLASSES']) for x in melody_iris]
    position_ids = [bundle['position_to_idx'].get(x, bundle['NUM_POSITION_CLASSES']) for x in melody_iris]
    interval_ids = [bundle['interval_to_idx'].get(x, bundle['NUM_INTERVAL_CLASSES']) for x in melody_iris]
    beat_strength_vals = [bundle['note_beat_strength'].get(x, 0.0) for x in melody_iris]
    t_pitch = torch.tensor([pitch_ids], dtype=torch.long, device=device)
    t_duration = torch.tensor([duration_ids], dtype=torch.long, device=device)
    t_beat_type = torch.tensor([beat_type_ids], dtype=torch.long, device=device)
    t_position = torch.tensor([position_ids], dtype=torch.long, device=device)
    t_interval = torch.tensor([interval_ids], dtype=torch.long, device=device)
    t_beat_strength = torch.tensor([beat_strength_vals], dtype=torch.float32, device=device)
    t_pad_mask = torch.zeros((1, n), dtype=torch.bool, device=device)
    chord_context_ids = torch.full((1, n), mask_chord_id, dtype=torch.long, device=device)
    with torch.no_grad():
        root_logits, quality_logits = model(t_pitch, t_duration, t_beat_type, t_position,
                                             t_interval, t_beat_strength, chord_context_ids, t_pad_mask)
    pred_root = root_logits.argmax(dim=-1)[0].tolist()
    pred_qual = quality_logits.argmax(dim=-1)[0].tolist()
    true_root = [bundle['chord_root_idx'][c] for c in chord_iris]
    true_qual = [bundle['chord_quality_idx'][c] for c in chord_iris]
    root_hits = sum(1 for i in range(n) if pred_root[i] == true_root[i])
    qual_hits = sum(1 for i in range(n) if pred_qual[i] == true_qual[i])
    joint_hits = sum(1 for i in range(n) if pred_root[i] == true_root[i] and pred_qual[i] == true_qual[i])
    return root_hits, qual_hits, joint_hits, n

def evaluate_batch(model, mask_chord_id, bundle, songs, label):
    total_root, total_qual, total_joint, total_notes = 0, 0, 0, 0
    per_song_joint = []
    for melody_iris, chord_iris in songs:
        r, q, j, n = evaluate_song(model, mask_chord_id, bundle, melody_iris, chord_iris)
        total_root += r; total_qual += q; total_joint += j; total_notes += n
        per_song_joint.append(j / n * 100 if n > 0 else 0.0)
    pooled_root = total_root / total_notes * 100 if total_notes > 0 else 0.0
    pooled_qual = total_qual / total_notes * 100 if total_notes > 0 else 0.0
    pooled_joint = total_joint / total_notes * 100 if total_notes > 0 else 0.0
    mean_joint = sum(per_song_joint) / len(per_song_joint) if per_song_joint else 0.0
    print(f"   [{label}] {len(songs)} τραγούδια, {total_notes} νότες -> "
          f"Root={pooled_root:.1f}% Qual={pooled_qual:.1f}% Joint={pooled_joint:.1f}% "
          f"(μέσος όρος ανά τραγούδι: {mean_joint:.1f}%)")
    return {'root': pooled_root, 'qual': pooled_qual, 'joint': pooled_joint, 'mean_per_song': mean_joint}


# =============================================================================
# ΜΕΤΡΙΚΕΣ ΤΩΝ ΔΥΟ PAPERS (Yeh et al. 2021 για CHE/CC/CTD/CTnCTR/PCS/MCTD,
# Wu et al. 2024 για HRHE/HRC/CBS) - ΟΛΟΙ οι ορισμοί επιβεβαιώθηκαν απευθείας
# από τα πρωτότυπα papers (μέσω αναζήτησης), ΟΧΙ από γενική ανάμνηση.
#
# ΣΗΜΑΝΤΙΚΟ: όλες οι μετρικές εδώ υπολογίζονται ΔΥΟ φορές ανά τραγούδι - μία
# πάνω στην ΠΡΟΒΛΕΨΗ του μοντέλου, μία πάνω στην ΠΡΑΓΜΑΤΙΚΗ χορδική πρόοδο -
# και το τελικό αποτέλεσμα είναι το |διαφορά| ανάμεσα στις δύο τιμές,
# ΜΕΣΟ ΟΡΟ σε όλο το held-out test set (ΑΚΡΙΒΩΣ όπως στους Πίνακες 2/3 του
# δεύτερου paper) - ΟΧΙ ακρίβεια θέση-προς-θέση όπως το Root/Quality/Joint.
# =============================================================================

# --- Μοτίβα διαστημάτων ανά ποιότητα (ΙΔΙΑ με το decode_chord_quality.py -
# χρειάζονται εδώ για να ξέρουμε ΠΟΙΟΙ φθόγγοι απαρτίζουν κάθε χορδή) ---
QUALITY_INTERVALS = {
    'maj': (0, 4, 7), 'm': (0, 3, 7), 'dim': (0, 3, 6), 'aug': (0, 4, 8),
    'sus': (0, 5, 7), 'sus2': (0, 2, 7), 'power': (0, 7),
    '7': (0, 4, 7, 10), 'm7': (0, 3, 7, 10), 'maj7': (0, 4, 7, 11),
    'o7': (0, 3, 6, 9), 'ø7': (0, 3, 6, 10), 'mM7': (0, 3, 7, 11),
    '7aug': (0, 4, 8, 11), '6': (0, 4, 7, 9), 'm6': (0, 3, 7, 9),
    '9': (0, 4, 7, 10, 2), 'm9': (0, 3, 7, 10, 2), 'M9': (0, 4, 7, 11, 2),
    '11': (0, 4, 7, 10, 2, 5), 'm11': (0, 3, 7, 10, 2, 5),
    '13': (0, 4, 7, 10, 2, 5, 9),
}
def chord_pitch_classes(root_pc, quality):
    """Επιστρέφει το ΣΥΝΟΛΟ φθόγγων (pitch classes) μιας χορδής, ή None αν
    δεν έχει νόημα (N.C./OTHER/UNKNOWN)."""
    if not isinstance(root_pc, int) or quality not in QUALITY_INTERVALS:
        return None
    return {(root_pc + iv) % 12 for iv in QUALITY_INTERVALS[quality]}

# --- Harte tonal centroid (Harte & Sandler 2006, "Detecting Harmonic Change
# in Musical Audio") - ΤΥΠΙΚΟΣ, καθιερωμένος μετασχηματισμός, ΙΔΙΟΣ με π.χ.
# το librosa.feature.tonnetz. 6 διαστάσεις = 3 κύκλοι (πέμπτες, μικρές
# τρίτες, μεγάλες τρίτες), ακτίνες r1=1, r2=1, r3=0.5. ---
def _harte_centroid(pitch_classes):
    """pitch_classes: σύνολο ή λίστα φθόγγων (0-11). Επιστρέφει 6-D centroid
    (tuple από floats), κανονικοποιημένο ως προς το πλήθος ενεργών φθόγγων."""
    if not pitch_classes:
        return (0.0,) * 6
    n = len(pitch_classes)
    r1, r2, r3 = 1.0, 1.0, 0.5
    dims = [0.0] * 6
    for l in pitch_classes:
        dims[0] += r1 * math.sin(l * 7 * math.pi / 6)
        dims[1] += r1 * math.cos(l * 7 * math.pi / 6)
        dims[2] += r2 * math.sin(l * 3 * math.pi / 2)
        dims[3] += r2 * math.cos(l * 3 * math.pi / 2)
        dims[4] += r3 * math.sin(l * 2 * math.pi / 3)
        dims[5] += r3 * math.cos(l * 2 * math.pi / 3)
    return tuple(d / n for d in dims)

def _euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def tonal_distance(pcs_a, pcs_b):
    return _euclidean(_harte_centroid(pcs_a), _harte_centroid(pcs_b))

# --- PCS - πίνακας συνωδίας (ΙΔΙΟΣ με τον Yeh et al. 2021 / μεταγενέστερα
# papers που τον επαναλαμβάνουν ρητά): 1 για ενωδά διαστήματα, 0 για καθαρή
# τετάρτη, -1 για όλα τα υπόλοιπα. Η μελωδική νότα θεωρείται ΠΑΝΤΑ η ΠΑΝΩ
# φωνή (ταιριάζει με το πώς είναι δομημένο το project: μελωδία πάνω, χορδή
# "από κάτω" εννοιολογικά). ---
def _consonance_score(interval_semitones):
    ic = interval_semitones % 12
    if ic in (0, 3, 4, 7, 8, 9):   # ενωδό/τρίτη/πέμπτη/έκτη
        return 1.0
    elif ic == 5:                  # καθαρή τετάρτη
        return 0.0
    else:                          # 2ες, τρίτονο, 7ες
        return -1.0

# =============================================================================
# generate_full_harmonization: ΠΡΟΒΛΕΨΗ (όχι απλή αξιολόγηση) - επιστρέφει τη
# ΠΛΗΡΗ προβλεπόμενη ακολουθία root/quality (ως pitch_class/quality strings,
# ΟΧΙ index) για ολόκληρο το τραγούδι, single-pass (μασκαρισμένο chord context,
# όπως παντού στο project μέχρι τώρα).
# =============================================================================
def generate_full_harmonization(model, mask_chord_id, bundle, melody_iris):
    n = len(melody_iris)
    pitch_ids = [bundle['pitch_oct_to_idx'].get(x, bundle['NUM_PITCH_OCT_CLASSES']) for x in melody_iris]
    duration_ids = [bundle['duration_to_idx'].get(x, bundle['NUM_DURATION_CLASSES']) for x in melody_iris]
    beat_type_ids = [bundle['beat_type_to_idx'].get(x, bundle['NUM_BEAT_TYPE_CLASSES']) for x in melody_iris]
    position_ids = [bundle['position_to_idx'].get(x, bundle['NUM_POSITION_CLASSES']) for x in melody_iris]
    interval_ids = [bundle['interval_to_idx'].get(x, bundle['NUM_INTERVAL_CLASSES']) for x in melody_iris]
    beat_strength_vals = [bundle['note_beat_strength'].get(x, 0.0) for x in melody_iris]
    t_pitch = torch.tensor([pitch_ids], dtype=torch.long, device=device)
    t_duration = torch.tensor([duration_ids], dtype=torch.long, device=device)
    t_beat_type = torch.tensor([beat_type_ids], dtype=torch.long, device=device)
    t_position = torch.tensor([position_ids], dtype=torch.long, device=device)
    t_interval = torch.tensor([interval_ids], dtype=torch.long, device=device)
    t_beat_strength = torch.tensor([beat_strength_vals], dtype=torch.float32, device=device)
    t_pad_mask = torch.zeros((1, n), dtype=torch.bool, device=device)
    chord_context_ids = torch.full((1, n), mask_chord_id, dtype=torch.long, device=device)
    with torch.no_grad():
        root_logits, quality_logits = model(t_pitch, t_duration, t_beat_type, t_position,
                                             t_interval, t_beat_strength, chord_context_ids, t_pad_mask)
    pred_root_idx = root_logits.argmax(dim=-1)[0].tolist()
    pred_qual_idx = quality_logits.argmax(dim=-1)[0].tolist()
    pred_root_pc = [bundle['idx_to_root_pc'].get(r, 'UNKNOWN_ROOT') for r in pred_root_idx]
    pred_quality = [bundle['idx_to_quality'].get(q, 'OTHER') for q in pred_qual_idx]
    return pred_root_pc, pred_quality

def ground_truth_sequence(bundle, chord_iris):
    root_pc = [bundle['idx_to_root_pc'].get(bundle['chord_root_idx'][c], 'UNKNOWN_ROOT') for c in chord_iris]
    quality = [bundle['idx_to_quality'].get(bundle['chord_quality_idx'][c], 'OTHER') for c in chord_iris]
    return root_pc, quality

def melody_pitch_classes(bundle, melody_iris):
    """Επιστρέφει τη λίστα ΗΔΗ-μεταθεμένων φθόγγων (0-11) της μελωδίας,
    ΑΠΕΥΘΕΙΑΣ από το bundle (υπολογίστηκε ήδη στο load_corpus_and_vocab_v2)."""
    return [bundle['note_pitch_class_transposed'].get(iri) for iri in melody_iris]

DURATION_TO_QUARTERS = {
    'Duration_whole': 4.0, 'Duration_dotted-half': 3.0, 'Duration_half': 2.0,
    'Duration_dotted-quarter': 1.5, 'Duration_quarter': 1.0, 'Duration_dotted-eighth': 0.75,
    'Duration_eighth': 0.5, 'Duration_dotted-16th': 0.375, 'Duration_16th': 0.25,
    'Duration_32nd': 0.125, 'Duration_64th': 0.0625, 'Duration_128th': 0.03125,
    'Duration_breve': 8.0, 'Duration_longa': 16.0, 'Duration_OTHER': 0.5,
}
def melody_durations_in_quarters(bundle, melody_iris):
    """Διάρκεια κάθε νότας σε τεταρτημόρια (για το ΣΤΑΘΜΙΣΜΕΝΟ MCTD), μέσω του
    ΠΡΑΓΜΑΤΙΚΟΥ label (bundle['duration_idx_to_label'])."""
    labels = bundle['duration_idx_to_label']
    out = []
    for iri in melody_iris:
        idx = bundle['duration_to_idx'].get(iri)
        out.append(DURATION_TO_QUARTERS.get(labels.get(idx), 0.5))
    return out

# =============================================================================
# ΟΙ 9 ΜΕΤΡΙΚΕΣ - καθεμία δέχεται (root_seq, qual_seq, melody_pcs, durations,
# beat_types) όπου χρειάζεται. root_seq/qual_seq είναι ΛΙΣΤΕΣ strings (pitch
# class 0-11 ή 'UNKNOWN_ROOT'/'NC'/'OTHER' αντίστοιχα), ΙΔΙΟ μήκος με τη
# μελωδία (μία χορδή ανά νότα - ΙΔΙΑ σύμβαση με όλο το project μέχρι τώρα).
# =============================================================================

def metric_CHE(root_seq, qual_seq):
    """Chord Histogram Entropy - εντροπία Shannon του ιστογράμματος χορδικών
    labels (root,quality) - ΚΑΘΕ νότα μετράει (frame-level, ΟΧΙ μόνο αλλαγές -
    ίδια σύμβαση με το πλέγμα ανά-τεταρτημόριο των δύο papers)."""
    hist = {}
    for r, q in zip(root_seq, qual_seq):
        hist[(r, q)] = hist.get((r, q), 0) + 1
    total = sum(hist.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in hist.values())

def metric_CC(root_seq, qual_seq):
    """Chord Coverage - πλήθος ΔΙΑΚΡΙΤΩΝ χορδικών labels."""
    return len({(r, q) for r, q in zip(root_seq, qual_seq)})

def metric_CTD(root_seq, qual_seq):
    """Chord Tonal Distance - μέση τονική απόσταση (Harte) ανάμεσα σε
    ΔΙΑΔΟΧΙΚΕΣ νότες (0 όταν η χορδή δεν αλλάζει - ίδια σύμβαση με το πλέγμα
    ανά-τεταρτημόριο, όπου μια χορδή που κρατάει επαναλαμβάνεται)."""
    if len(root_seq) < 2:
        return 0.0
    dists = []
    for i in range(len(root_seq) - 1):
        pcs_a = chord_pitch_classes(root_seq[i], qual_seq[i])
        pcs_b = chord_pitch_classes(root_seq[i + 1], qual_seq[i + 1])
        if pcs_a is None or pcs_b is None:
            continue
        dists.append(tonal_distance(pcs_a, pcs_b))
    return sum(dists) / len(dists) if dists else 0.0

def metric_CTnCTR(root_seq, qual_seq, melody_pcs):
    """CTnCTR = (n_c + n_p) / (n_c + n_n) - Yeh et al. 2021, Eq. οριστικός
    τύπος. n_p = "σωστός" μη-αρμονικός φθόγγος: η ΕΠΟΜΕΝΗ νότα είναι φθόγγος
    της χορδής ΚΑΙ το διάστημα προς αυτήν είναι ≤2 ημιτόνια."""
    n_c, n_n, n_p = 0, 0, 0
    for i, (r, q, mpc) in enumerate(zip(root_seq, qual_seq, melody_pcs)):
        if mpc is None:
            continue
        pcs = chord_pitch_classes(r, q)
        if pcs is None:
            continue
        is_chord_tone = mpc in pcs
        if is_chord_tone:
            n_c += 1
        else:
            n_n += 1
            if i + 1 < len(melody_pcs) and melody_pcs[i + 1] is not None:
                next_pcs = chord_pitch_classes(root_seq[i + 1], qual_seq[i + 1]) if i + 1 < len(root_seq) else None
                if next_pcs is not None and melody_pcs[i + 1] in next_pcs:
                    diff = min((melody_pcs[i + 1] - mpc) % 12, (mpc - melody_pcs[i + 1]) % 12)
                    if diff <= 2:
                        n_p += 1
    denom = n_c + n_n
    return (n_c + n_p) / denom if denom > 0 else 1.0

def metric_PCS(root_seq, qual_seq, melody_pcs):
    """Pitch Consonance Score - μέσος όρος συνωδίας ανάμεσα σε κάθε νότα
    μελωδίας και ΚΑΘΕ φθόγγο της τρέχουσας χορδής (Yeh et al. 2021)."""
    scores = []
    for r, q, mpc in zip(root_seq, qual_seq, melody_pcs):
        if mpc is None:
            continue
        pcs = chord_pitch_classes(r, q)
        if not pcs:
            continue
        for chord_pc in pcs:
            interval = (mpc - chord_pc) % 12
            scores.append(_consonance_score(interval))
    return sum(scores) / len(scores) if scores else 0.0

def metric_MCTD(root_seq, qual_seq, melody_pcs, durations):
    """Melody-Chord Tonal Distance - τονική απόσταση (Harte) ανάμεσα σε κάθε
    νότα (ως μονο-φθογγικό centroid) και την τρέχουσα χορδή, ΣΤΑΘΜΙΣΜΕΝΗ ανά
    διάρκεια νότας (Yeh et al. 2021)."""
    total_w, total = 0.0, 0.0
    for r, q, mpc, dur in zip(root_seq, qual_seq, melody_pcs, durations):
        if mpc is None:
            continue
        pcs = chord_pitch_classes(r, q)
        if pcs is None:
            continue
        d = tonal_distance({mpc}, pcs)
        w = dur if dur and dur > 0 else 1.0
        total += d * w
        total_w += w
    return total / total_w if total_w > 0 else 0.0

def metric_HRHE_HRC(root_seq, qual_seq, beat_types):
    """Harmonic Rhythm Histogram Entropy + Coverage - ΙΔΙΟ ιστόγραμμα: θέση
    (beat_type) κάθε ΑΛΛΑΓΗΣ χορδής. HRHE=εντροπία, HRC=πλήθος διακριτών
    θέσεων που χρησιμοποιήθηκαν (Wu et al. 2024, EURASIP)."""
    hist = {}
    prev = None
    for (r, q), bt in zip(zip(root_seq, qual_seq), beat_types):
        if (r, q) != prev:
            hist[bt] = hist.get(bt, 0) + 1
        prev = (r, q)
    total = sum(hist.values())
    if total == 0:
        return 0.0, 0
    entropy = -sum((c / total) * math.log(c / total) for c in hist.values())
    return entropy, len(hist)

def _cbs_score(label):
    """Strong=0, Weak=1, Offbeat=2 (το FF paper ορίζει: lower = ισχυρότερη
    ευθυγράμμιση, higher = συγκοπή). Ταίριασμα βάσει λέξης στο label, ώστε να μην
    εξαρτάται από την ακριβή ορθογραφία του. Άγνωστο label -> None (παραλείπεται)."""
    l = str(label).lower()
    if 'strong' in l:
        return 0.0
    if 'weak' in l:
        return 1.0
    if 'off' in l:
        return 2.0
    return None

def metric_CBS(root_seq, qual_seq, beat_type_labels):
    """Chord Beat Strength - μέσος όρος "συγκοπής" στις ΑΛΛΑΓΕΣ χορδής."""
    scores = []
    prev = None
    for (r, q), bt in zip(zip(root_seq, qual_seq), beat_type_labels):
        if (r, q) != prev:
            sc = _cbs_score(bt)
            if sc is not None:
                scores.append(sc)
        prev = (r, q)
    return sum(scores) / len(scores) if scores else 0.0

HR_USE_POSITION = True   # True: τα HRHE/HRC χρησιμοποιούν ζεύγος (beat_type, position) ως bin
                         # (έως 9 bins αντί για 3 - πιο κοντά στη λεπτή μετρική θέση των papers)
METRIC_NAMES = ['CHE', 'CC', 'CTD', 'CTnCTR', 'PCS', 'MCTD', 'HRHE', 'HRC', 'CBS']

def evaluate_song_all_metrics(model, mask_chord_id, bundle, melody_iris, chord_iris):
    """Επιστρέφει (abs_diffs, signed_diffs) ανά μετρική για ΕΝΑ τραγούδι.
    signed = πρόβλεψη - πραγματικό (>0: το μοντέλο ΥΠΕΡΒΑΙΝΕΙ την τιμή)."""
    pred_root, pred_qual = generate_full_harmonization(model, mask_chord_id, bundle, melody_iris)
    true_root, true_qual = ground_truth_sequence(bundle, chord_iris)
    mpcs = melody_pitch_classes(bundle, melody_iris)
    durations = melody_durations_in_quarters(bundle, melody_iris)
    bt_labels_map = bundle['beat_type_idx_to_label']
    bt_idx = [bundle['beat_type_to_idx'].get(iri) for iri in melody_iris]
    beat_labels = [bt_labels_map.get(i, '__UNKNOWN__') for i in bt_idx]
    if HR_USE_POSITION:
        hr_bins = [(b, bundle['position_to_idx'].get(iri)) for b, iri in zip(bt_idx, melody_iris)]
    else:
        hr_bins = bt_idx

    pred_vals, true_vals = {}, {}
    for name, (rs, qs, store) in {'pred': (pred_root, pred_qual, pred_vals), 'true': (true_root, true_qual, true_vals)}.items():
        store['CHE'] = metric_CHE(rs, qs)
        store['CC'] = metric_CC(rs, qs)
        store['CTD'] = metric_CTD(rs, qs)
        store['CTnCTR'] = metric_CTnCTR(rs, qs, mpcs)
        store['PCS'] = metric_PCS(rs, qs, mpcs)
        store['MCTD'] = metric_MCTD(rs, qs, mpcs, durations)
        store['HRHE'], store['HRC'] = metric_HRHE_HRC(rs, qs, hr_bins)
        store['CBS'] = metric_CBS(rs, qs, beat_labels)
    abs_d = {k: abs(pred_vals[k] - true_vals[k]) for k in METRIC_NAMES}
    sgn_d = {k: pred_vals[k] - true_vals[k] for k in METRIC_NAMES}
    n = len(true_root)
    root_hit = [pr == tr for pr, tr in zip(pred_root, true_root)]
    qual_hit = [pq == tq for pq, tq in zip(pred_qual, true_qual)]
    acc = {'Root': sum(root_hit) / n * 100, 'Qual': sum(qual_hit) / n * 100,
           'Joint': sum(1 for a, b in zip(root_hit, qual_hit) if a and b) / n * 100}
    return abs_d, sgn_d, acc

def evaluate_batch_all_metrics(model, mask_chord_id, bundle, songs, label):
    abs_all = {k: [] for k in METRIC_NAMES}
    sgn_all = {k: [] for k in METRIC_NAMES}
    acc_all = {'Root': [], 'Qual': [], 'Joint': []}
    for melody_iris, chord_iris in songs:
        a, g, acc = evaluate_song_all_metrics(model, mask_chord_id, bundle, melody_iris, chord_iris)
        for k in METRIC_NAMES:
            abs_all[k].append(a[k]); sgn_all[k].append(g[k])
        for k in acc_all:
            acc_all[k].append(acc[k])
    mean = lambda v: sum(v) / len(v) if v else 0.0
    mae = {k: mean(v) for k, v in abs_all.items()}
    bias = {k: mean(v) for k, v in sgn_all.items()}
    acc_mean = {k: mean(v) for k, v in acc_all.items()}
    print(f"\n[{label}] {len(songs)} τραγούδια")
    print(f"   {'':8}" + ''.join(f"{k:>9}" for k in METRIC_NAMES))
    print(f"   {'MAE':8}" + ''.join(f"{mae[k]:>9.4f}" for k in METRIC_NAMES))
    print(f"   {'bias':8}" + ''.join(f"{bias[k]:>+9.4f}" for k in METRIC_NAMES) + "   (bias = πρόβλεψη - πραγματικό)")
    print(f"   Ακρίβεια (μέσος όρος ανά τραγούδι): Root={acc_mean['Root']:.1f}% Qual={acc_mean['Qual']:.1f}% Joint={acc_mean['Joint']:.1f}%")
    return mae, bias, acc_mean


# =============================================================================
# ΑΝΑΛΥΣΗ ABLATION
# Για κάθε variant (checkpoint[s]) υπολογίζει, στο ΠΛΗΡΕΣ σύνολο επικύρωσης:
#   1. ακρίβεια Root / Quality / Joint (ανά νότα - συγκρίσιμη με το "Val Acc" των logs)
#   2. τις 9 μετρικές των papers (MAE και bias, πρόβλεψη - πραγματικό)
#   3. πόσο ΤΗΡΟΥΝΤΑΙ οι κανόνες 1,2,3,5,6,7,8, δηλαδή πόσο μοιάζει η συμπεριφορά
#      των προβλέψεων με τη συμπεριφορά των ΠΡΑΓΜΑΤΙΚΩΝ δεδομένων ως προς κάθε κανόνα
# και φτιάχνει γραφήματα + αρχεία CSV. Ο Κανόνας 4 δεν περιλαμβάνεται (κλειστός).
# =============================================================================
import os
import csv
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

# ============================== ΡΥΘΜΙΣΕΙΣ ====================================
# (ΑΛΛΑΞΕ ΜΟΝΟ ΑΥΤΟ ΤΟ ΜΠΛΟΚ)

# Δεδομένα: ΙΔΙΑ με αυτά που χρησιμοποιήθηκαν στην εκπαίδευση (παλιά εξαγωγή)
CORPUS_PATH = os.path.join(DATA_DIR, 'final_corpus.pkl')
LOOKUP_PATH = os.path.join(DATA_DIR, 'symbolic_lookup.pkl')
EXTRACTED_JSON_PATH = os.path.join(DATA_DIR1, 'extracted_songs.json')
POP909_JSON_PATH = os.path.join(DATA_DIR1, 'extracted_pop909.json')

OUTPUT_DIR = os.environ.get('HARMONIZER_RESULTS', os.path.join(OUT_DIR, 'ablation_results'))

# Φάκελος με τους 8 υποφακέλους των μοντέλων (final_data). None = βρίσκεται αυτόματα:
# ψάχνει έναν φάκελο με όνομα 'final_data' κάτω από το INPUT_SEARCH_ROOT.
# Αν δεν τον βρει, βάλε εδώ την πλήρη διαδρομή, π.χ. '/kaggle/input/final_data'.
CKPT_ROOT = os.environ.get('HARMONIZER_CKPTS') or None
INPUT_SEARCH_ROOT = os.environ.get('HARMONIZER_SEARCH_ROOT', '/kaggle/input')

# Τα ονόματα των αρχείων μέσα σε ΚΑΘΕ υποφάκελο:
FINAL_FILE = 'best_transformer_model.pth'            # μοντέλο της ΤΕΛΕΥΤΑΙΑΣ εποχής (200)
BEST_FILE = 'best_val_acc_transformer_model.pth'     # μοντέλο της εποχής με την καλύτερη val acc

# Ποιο αρχείο αξιολογείται σε αυτή την εκτέλεση. Τρέξε το script ΔΥΟ φορές:
#   'final' = τελευταία εποχή (ΚΥΡΙΟ πρωτόκολλο) | 'best' = καλύτερη val acc (έλεγχος ευρωστίας)
# Τα αποτελέσματα κάθε εκτέλεσης πάνε σε ξεχωριστό φάκελο.
PROTOCOL = 'final'

# True = ΔΕΝ φορτώνει κανένα μοντέλο και καμία πρόβλεψη. Φορτώνει μόνο τα δεδομένα και υπολογίζει τη μέση τιμή
# των 9 μετρικών στις ΠΡΑΓΜΑΤΙΚΕΣ αρμονίες (για σύγκριση με τον Πίνακα Ι του paper αναφοράς). Γράφει το
# truth_paper_means.csv και τυπώνει τον πίνακα. Τρέχει σε λίγα λεπτά και δεν χρειάζεται τον φάκελο final_data.
ONLY_TRUTH_MEANS = False

# Κάθε variant: ετικέτα, ποιοι κανόνες ΛΕΙΠΟΥΝ (removed), και ο υποφάκελος (folders).
# Αν έχεις πολλά seeds, βάλε περισσότερους φακέλους στη λίστα (βγαίνουν μέσος όρος και std).
ALL_RULES = [1, 2, 3, 5, 6, 7, 8]

# Τα μοντέλα "only X" εκπαιδεύτηκαν ΜΟΝΟ με τους κανόνες της ομάδας X, άρα ΛΕΙΠΟΥΝ οι υπόλοιποι.
def _rest(enabled):
    return [r for r in ALL_RULES if r not in enabled]

VARIANTS = [
    {'label': 'baseline',       'removed': ALL_RULES,         'folders': ['no_rules']},
    {'label': 'full',           'removed': [],                'folders': ['all_the_rules']},
    {'label': 'full − {1,2,5}', 'removed': [1, 2, 5],         'folders': ['except_125_final']},
    {'label': 'full − {3,7}',   'removed': [3, 7],            'folders': ['except_37_final']},
    {'label': 'full − {6,8}',   'removed': [6, 8],            'folders': ['except_68_final']},
    {'label': 'only {1,2,5}',   'removed': _rest([1, 2, 5]),  'folders': ['only_125_final']},
    {'label': 'only {3,7}',     'removed': _rest([3, 7]),     'folders': ['only_37_final']},
    {'label': 'only {6,8}',     'removed': _rest([6, 8]),     'folders': ['only_68_final']},
]
FULL_LABEL = 'full'
BASELINE_LABEL = 'baseline'

# Προαιρετικά: logs εκπαίδευσης (αντιγραμμένο κείμενο) για καμπύλες Val Acc.
# {'full': '/path/log_full.txt', ...} - μία διαδρομή ή λίστα διαδρομών (seeds).
LOG_PATHS = {}

MAX_SONGS_PER_SOURCE = None   # None = ΟΛΑ τα τραγούδια επικύρωσης. Βάλε π.χ. 300 για γρήγορη δοκιμή
USE_PRED_CACHE = True         # αποθηκεύει τις προβλέψεις, ώστε να μην ξανατρέχει το μοντέλο
BOOTSTRAP_B = 1000            # επαναδειγματοληψίες για διαστήματα εμπιστοσύνης
SHOW_PLOTS = True             # εμφάνιση στο notebook (τα αρχεία αποθηκεύονται πάντα)
# ============================================================================

# Ο πίνακας συμβατότητας της εκπαίδευσης χρησιμοποιεί ΑΥΤΕΣ τις τιμές για 7aug και 13
# (οι μετρικές πιο πάνω είχαν ελαφρώς διαφορετικές): ευθυγραμμίζουμε με την εκπαίδευση.
QUALITY_INTERVALS['7aug'] = (0, 4, 8, 10)
QUALITY_INTERVALS['13'] = (0, 4, 7, 10, 2, 9)

# Τα αρχεία checkpoint της εκπαίδευσης έχουν ΕΝΑ σετ βαρών με κλειδί 'model_state_dict'.
# (Το 'best_model_state_dict' υπάρχει μόνο σε παλιότερα αρχεία και είναι εφεδρικό.)
CHECKPOINT_KEY_PREFERENCE = ('model_state_dict', 'best_model_state_dict')

# Cache στο κεντροειδές Harte (επιταχύνει πολύ τα CTD/MCTD, ίδιο αποτέλεσμα).
_harte_centroid_uncached = _harte_centroid
_harte_cache = {}
def _harte_centroid(pitch_classes):
    key = frozenset(pitch_classes)
    if key not in _harte_cache:
        _harte_cache[key] = _harte_centroid_uncached(pitch_classes)
    return _harte_cache[key]

SOURCES = ['classics', 'CMD', 'POP909']
BEAT_KEYS = ('strong', 'weak', 'off')
FUNCS = ['T', 'S', 'D', 'X']
RULE_COLS = [1, 2, 3, 5, 6, 7, 8]

# Λειτουργίες T/S/D: ΑΥΤΟΛΕΞΕΙ ίδιοι πίνακες με την εκπαίδευση (Κανόνας 6).
MAJOR_FUNCTION_BY_OFFSET = {0: 'T', 2: 'S', 4: 'T', 5: 'S', 7: 'D', 9: 'T', 11: 'D'}
MINOR_FUNCTION_BY_OFFSET = {0: 'T', 2: 'S', 3: 'T', 5: 'S', 7: 'D', 8: 'T', 10: 'D', 11: 'D'}

def chord_function_label(root_pc, is_major):
    if not isinstance(root_pc, int):
        return 'X'
    tonic_pc = 0 if is_major else 9
    offset = (root_pc - tonic_pc) % 12
    table = MAJOR_FUNCTION_BY_OFFSET if is_major else MINOR_FUNCTION_BY_OFFSET
    return table.get(offset, 'X')

def beat_key(label):
    l = str(label).lower()
    if 'strong' in l:
        return 'strong'
    if 'weak' in l:
        return 'weak'
    if 'off' in l:
        return 'off'
    return None

def pct(a, b):
    return 100.0 * a / b if b > 0 else float('nan')

# =============================================================================
# 1. ΤΜΗΜΑΤΑ ΣΥΝΟΛΟΥ ΕΠΙΚΥΡΩΣΗΣ ΚΑΙ CACHE ΑΝΑ ΤΡΑΓΟΥΔΙ
# =============================================================================
def split_val_by_source(bundle, max_per_source=None, seed=0):
    val = sorted(bundle['val_only_indices'])
    b = bundle['base_classical_count']
    c = bundle['classical_song_count']
    p = bundle['pop909_start_index']
    groups = {'classics': [i for i in val if i < b],
              'CMD': [i for i in val if b <= i < c],
              'POP909': [i for i in val if i >= p]}
    if max_per_source:
        rng = random.Random(seed)
        for k, ids in list(groups.items()):
            if len(ids) > max_per_source:
                groups[k] = sorted(rng.sample(ids, max_per_source))
    groups = {k: v for k, v in groups.items() if v}
    all_ids = sorted(i for v in groups.values() for i in v)
    return groups, all_ids

def build_song_cache(bundle, song_idx):
    """Ό,τι χρειάζεται η ανάλυση για ένα τραγούδι, ΜΙΑ φορά (όχι ανά variant)."""
    full_melody, full_chords = bundle['raw_songs'][song_idx]
    L = min(len(full_melody), bundle['max_seq_len'])      # ίδια περικοπή με την εκπαίδευση
    melody_iris, chord_iris = full_melody[:L], full_chords[:L]
    is_major = bool(bundle['song_is_major'][song_idx])
    mpcs = melody_pitch_classes(bundle, melody_iris)
    semis = [pitch_name_to_semitone(bundle['transposed_pitch_name'].get(i)) for i in melody_iris]

    # Κανόνας 2: ΑΥΤΟΛΕΞΕΙ ο ορισμός της εκπαίδευσης (βηματική κίνηση = ≤2 ημιτόνια
    # από την προηγούμενη Ή προς την επόμενη νότα· άγνωστες τιμές μετράνε ως βηματικές)
    step_resolved = [1] * L
    for t in range(L):
        if semis[t] is None:
            continue
        approached_ok = (t == 0) or (semis[t - 1] is None) or (abs(semis[t] - semis[t - 1]) <= 2)
        left_ok = (t == L - 1) or (semis[t + 1] is None) or (abs(semis[t + 1] - semis[t]) <= 2)
        step_resolved[t] = 1 if (approached_ok or left_ok) else 0

    # Κανόνας 5: τονική προσαγωγή που επιλύεται (11 → 0 σε μείζονα, 8 → 9 σε ελάσσονα)
    lt_pc, tonic_pc = (11, 0) if is_major else (8, 9)
    resolving_lt = [0] * L
    for t in range(L - 1):
        if mpcs[t] == lt_pc and mpcs[t + 1] == tonic_pc:
            resolving_lt[t] = 1

    bt_idx = [bundle['beat_type_to_idx'].get(i) for i in melody_iris]
    beat_labels = [bundle['beat_type_idx_to_label'].get(b, '__UNKNOWN__') for b in bt_idx]
    true_root, true_qual = ground_truth_sequence(bundle, chord_iris)
    cs = {
        'L': L, 'truncated': len(full_melody) > bundle['max_seq_len'], 'is_major': is_major,
        'mpcs': mpcs, 'step_resolved': step_resolved, 'resolving_lt': resolving_lt,
        'beat_labels': beat_labels, 'beat_keys': [beat_key(x) for x in beat_labels],
        'hr_bins': [(b, bundle['position_to_idx'].get(i)) for b, i in zip(bt_idx, melody_iris)]
                   if HR_USE_POSITION else bt_idx,
        'durations': melody_durations_in_quarters(bundle, melody_iris),
        'true_root': true_root, 'true_qual': true_qual,
    }
    cs['truth_paper'] = compute_paper_values(true_root, true_qual, cs)
    return cs

def compute_paper_values(root, qual, cs):
    v = {}
    v['CHE'] = metric_CHE(root, qual)
    v['CC'] = metric_CC(root, qual)
    v['CTD'] = metric_CTD(root, qual)
    v['CTnCTR'] = metric_CTnCTR(root, qual, cs['mpcs'])
    v['PCS'] = metric_PCS(root, qual, cs['mpcs'])
    v['MCTD'] = metric_MCTD(root, qual, cs['mpcs'], cs['durations'])
    v['HRHE'], v['HRC'] = metric_HRHE_HRC(root, qual, cs['hr_bins'])
    v['CBS'] = metric_CBS(root, qual, cs['beat_labels'])
    return v

# =============================================================================
# 2. ΜΕΤΡΗΣΕΙΣ ΤΗΡΗΣΗΣ ΤΩΝ ΚΑΝΟΝΩΝ
# Για κάθε κανόνα μετράμε τη συμπεριφορά μιας ακολουθίας συγχορδιών, ΙΔΙΑ για τις
# προβλέψεις και για τα πραγματικά δεδομένα. "Τήρηση" = πόσο κοντά είναι η
# συμπεριφορά των προβλέψεων σε αυτή των πραγματικών (απόκλιση "gap": μικρότερη = καλύτερη).
# =============================================================================
def is_compatible(note_pc, root_pc, quality):
    """ΙΔΙΑ λογική με τον πίνακα συμβατότητας της εκπαίδευσης: άγνωστο περιεχόμενο
    συγχορδίας (NC, OTHER, άγνωστη ρίζα) ή άγνωστη νότα => πάντα συμβατό."""
    pcs = chord_pitch_classes(root_pc, quality)
    if pcs is None or note_pc is None:
        return True
    return note_pc in pcs

def rule_counts_for_song(cs, root_seq, qual_seq):
    c = Counter()
    L = cs['L']
    is_major = cs['is_major']
    for t in range(L):
        r, q = root_seq[t], qual_seq[t]
        bk = cs['beat_keys'][t]
        viol = 0 if is_compatible(cs['mpcs'][t], r, q) else 1
        c[('notes',)] += 1
        if chord_pitch_classes(r, q) is None:
            c[('unknown',)] += 1
        if bk:                                   # Κανόνας 1
            c[('tot_beat', bk)] += 1
            c[('viol_beat', bk)] += viol
        sk = 'step' if cs['step_resolved'][t] else 'nonstep'     # Κανόνας 2
        c[('tot_step', sk)] += 1
        c[('viol_step', sk)] += viol
        if cs['resolving_lt'][t]:                # Κανόνας 5
            c[('tot_lt',)] += 1
            c[('viol_lt',)] += viol
        f = chord_function_label(r, is_major)
        c[('qf', f, q)] += 1                     # Κανόνας 8: ποιότητα ανά λειτουργία
        if t >= 1:
            same = (r, q) == (root_seq[t - 1], qual_seq[t - 1])   # Κανόνας 3 (σωστή σύγκριση συγχορδιών)
            c[('pairs',)] += 1
            c[('changes',)] += 0 if same else 1
            if bk:
                c[('tot_same', bk)] += 1
                c[('same', bk)] += 1 if same else 0
            pf = chord_function_label(root_seq[t - 1], is_major)  # Κανόνας 6
            if pf != f:
                c[('trans', pf, f)] += 1
    if not cs['truncated'] and L > 0:            # Κανόνας 7: ΜΟΝΟ τραγούδια που δεν περικόπηκαν
        c[('end_total',)] += 1
        if chord_function_label(root_seq[-1], is_major) == 'T':
            c[('end_T',)] += 1
    return c

def rule_levels(cnt):
    out = {}
    for bk in BEAT_KEYS:
        out[f'R1_viol_{bk}'] = pct(cnt[('viol_beat', bk)], cnt[('tot_beat', bk)])
        out[f'R3_same_{bk}'] = pct(cnt[('same', bk)], cnt[('tot_same', bk)])
    for sk in ('step', 'nonstep'):
        out[f'R2_viol_{sk}'] = pct(cnt[('viol_step', sk)], cnt[('tot_step', sk)])
    out['R3_change_rate'] = pct(cnt[('changes',)], cnt[('pairs',)])
    out['R5_viol_lt'] = pct(cnt[('viol_lt',)], cnt[('tot_lt',)])
    d_total = sum(cnt[('trans', 'D', g)] for g in FUNCS if g != 'D')
    out['R6_DT'] = pct(cnt[('trans', 'D', 'T')], d_total)
    out['R7_end_tonic'] = pct(cnt[('end_T',)], cnt[('end_total',)])
    out['unknown_chord_frac'] = pct(cnt[('unknown',)], cnt[('notes',)])
    return out

def rule_gaps(pc, tc):
    """Απόκλιση (σε ποσοστιαίες μονάδες) της συμπεριφοράς των προβλέψεων (pc) από
    τα πραγματικά δεδομένα (tc), ανά κανόνα. 0 = ακριβώς όπως τα δεδομένα."""
    pl, tl = rule_levels(pc), rule_levels(tc)
    def mean_abs(names):
        d = [abs(pl[n] - tl[n]) for n in names if not (np.isnan(pl[n]) or np.isnan(tl[n]))]
        return float(np.mean(d)) if d else float('nan')
    g = {
        'R1_gap': mean_abs([f'R1_viol_{b}' for b in BEAT_KEYS]),
        'R2_gap': mean_abs(['R2_viol_step', 'R2_viol_nonstep']),
        'R3_gap': mean_abs([f'R3_same_{b}' for b in BEAT_KEYS]),
        'R5_gap': mean_abs(['R5_viol_lt']),
        'R7_gap': mean_abs(['R7_end_tonic']),
    }
    # Κανόνας 6: σταθμισμένη απόσταση (total variation) των κατανομών "προς ποια
    # λειτουργία πηγαίνει η αρμονία όταν αλλάζει λειτουργία", ανά λειτουργία-αφετηρία
    num = den = 0.0
    for f in FUNCS:
        others = [x for x in FUNCS if x != f]
        tn = sum(tc[('trans', f, x)] for x in others)
        if tn <= 0:
            continue
        pn = sum(pc[('trans', f, x)] for x in others)
        tv = 1.0 if pn <= 0 else 0.5 * sum(abs(tc[('trans', f, x)] / tn - pc[('trans', f, x)] / pn) for x in others)
        num += tn * tv
        den += tn
    g['R6_gap'] = 100.0 * num / den if den > 0 else float('nan')
    # Κανόνας 8: ίδια λογική για την κατανομή P(ποιότητα | λειτουργία)
    quals = sorted({k[2] for k in list(pc.keys()) + list(tc.keys()) if k[0] == 'qf'})
    num = den = 0.0
    for f in FUNCS:
        tn = sum(tc[('qf', f, q)] for q in quals)
        if tn <= 0:
            continue
        pn = sum(pc[('qf', f, q)] for q in quals)
        tv = 1.0 if pn <= 0 else 0.5 * sum(abs(tc[('qf', f, q)] / tn - pc[('qf', f, q)] / pn) for q in quals)
        num += tn * tv
        den += tn
    g['R8_gap'] = 100.0 * num / den if den > 0 else float('nan')
    return g

# =============================================================================
# 3. ΠΡΟΒΛΕΨΕΙΣ ΜΟΝΤΕΛΟΥ (το ΜΟΝΟ κομμάτι που χρειάζεται torch/GPU)
# =============================================================================
def safe_name(s):
    return ''.join(ch if ch.isalnum() else '_' for ch in s)

def predict_checkpoint(bundle, ckpt_path, song_ids, cache_file=None):
    if USE_PRED_CACHE and cache_file and os.path.exists(cache_file):
        print(f"   ↩️ Φορτώνονται αποθηκευμένες προβλέψεις: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    model, mask_chord_id = build_and_load_model(bundle, ckpt_path)
    preds = {}
    for k, i in enumerate(song_ids):
        melody_iris = bundle['raw_songs'][i][0][:bundle['max_seq_len']]
        preds[i] = generate_full_harmonization(model, mask_chord_id, bundle, melody_iris)
        if (k + 1) % 500 == 0:
            print(f"      ... {k + 1}/{len(song_ids)} τραγούδια")
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    if USE_PRED_CACHE and cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(preds, f)
    return preds

# =============================================================================
# 4. ΑΝΑΛΥΣΗ ΕΝΟΣ CHECKPOINT
# =============================================================================
def bootstrap_ci_acc(hits, notes, B, seed=0):
    hits, notes = np.asarray(hits, dtype=float), np.asarray(notes, dtype=float)
    S = len(hits)
    if S == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, S, size=(B, S))
    accs = 100.0 * hits[idx].sum(axis=1) / notes[idx].sum(axis=1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))

def analyze_checkpoint(cache, groups, all_ids, preds, truth_counts):
    per = {}
    for i in all_ids:
        cs = cache[i]
        pr, pq = preds[i]
        tr, tq = cs['true_root'], cs['true_qual']
        rh = sum(1 for a, b in zip(pr, tr) if a == b)
        qh = sum(1 for a, b in zip(pq, tq) if a == b)
        jh = sum(1 for a, b, c_, d in zip(pr, tr, pq, tq) if a == b and c_ == d)
        pv, tv = compute_paper_values(pr, pq, cs), cs['truth_paper']
        per[i] = {'n': cs['L'], 'rh': rh, 'qh': qh, 'jh': jh,
                  'diff': {m: pv[m] - tv[m] for m in METRIC_NAMES},
                  'cnt': rule_counts_for_song(cs, pr, pq)}
    srcs = {'ALL': all_ids}
    srcs.update(groups)
    flat = {}
    for src, ids in srcs.items():
        n = sum(per[i]['n'] for i in ids)
        jh_arr = [per[i]['jh'] for i in ids]
        n_arr = [per[i]['n'] for i in ids]
        flat[f'{src}|acc|root_notes'] = pct(sum(per[i]['rh'] for i in ids), n)
        flat[f'{src}|acc|qual_notes'] = pct(sum(per[i]['qh'] for i in ids), n)
        flat[f'{src}|acc|joint_notes'] = pct(sum(jh_arr), n)
        flat[f'{src}|acc|joint_song_mean'] = float(np.mean([pct(per[i]['jh'], per[i]['n']) for i in ids]))
        lo, hi = bootstrap_ci_acc(jh_arr, n_arr, BOOTSTRAP_B)
        flat[f'{src}|acc|joint_ci_lo'], flat[f'{src}|acc|joint_ci_hi'] = lo, hi
        for m in METRIC_NAMES:
            d = [per[i]['diff'][m] for i in ids]
            flat[f'{src}|paper_mae|{m}'] = float(np.mean(np.abs(d)))
            flat[f'{src}|paper_bias|{m}'] = float(np.mean(d))
        cnt = Counter()
        for i in ids:
            cnt.update(per[i]['cnt'])
        for k, v in {**rule_levels(cnt), **rule_gaps(cnt, truth_counts[src])}.items():
            flat[f'{src}|rule|{k}'] = v
    per_song_joint = (np.array([per[i]['jh'] for i in all_ids], dtype=float),
                      np.array([per[i]['n'] for i in all_ids], dtype=float))
    return flat, per_song_joint

# =============================================================================
# 5. ΣΥΓΚΕΝΤΡΩΣΗ (seeds), ΔΙΑΦΟΡΑ ΑΠΟ ΤΟ FULL, ΑΡΧΕΙΑ CSV
# =============================================================================
def aggregate(results):
    agg = {}
    for label, lst in results.items():
        keys = sorted(set().union(*[set(d.keys()) for d in lst]))
        agg[label] = {}
        for k in keys:
            vals = np.array([d[k] for d in lst if k in d], dtype=float)
            agg[label][k] = (float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0, len(vals))   # ddof=1: δειγματική τυπική απόκλιση (n-1)
    return agg

def fval(agg, label, key):
    return agg.get(label, {}).get(key, (float('nan'), 0.0, 0))[0]

def ferr(agg, label, key):
    """Μπάρα σφάλματος: std ανά seeds αν υπάρχουν πολλά, αλλιώς μισό 95% CI για την Joint."""
    mean, std, n = agg.get(label, {}).get(key, (float('nan'), 0.0, 0))
    if n > 1:
        return std
    if key.endswith('|acc|joint_notes'):
        lo, hi = fval(agg, label, key.replace('joint_notes', 'joint_ci_lo')), fval(agg, label, key.replace('joint_notes', 'joint_ci_hi'))
        if not (np.isnan(lo) or np.isnan(hi)):
            return (hi - lo) / 2.0
    return 0.0

def compute_delta(per_song_by_variant, ref_label, B, seed=1):
    """Ζευγαρωτό bootstrap: Δ Joint (π.μ.) κάθε variant έναντι του ref_label, στα ΙΔΙΑ τραγούδια."""
    full_label = ref_label
    if full_label not in per_song_by_variant:
        return {}
    def avg(lst):
        return np.mean([h for h, _ in lst], axis=0), lst[0][1]
    hf, nf = avg(per_song_by_variant[full_label])
    S = len(hf)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, S, size=(B, S))
    den = nf[idx].sum(axis=1)
    acc_f = hf[idx].sum(axis=1) / den
    out = {}
    for label, lst in per_song_by_variant.items():
        if label == full_label:
            continue
        hv, nv = avg(lst)
        d = 100.0 * (hv[idx].sum(axis=1) / den - acc_f)
        point = 100.0 * (hv.sum() - hf.sum()) / nf.sum()
        out[label] = (float(point), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    return out

def write_csvs(results, agg, delta_full, delta_base, outdir):
    with open(os.path.join(outdir, 'results_per_checkpoint.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'checkpoint_index', 'key', 'value'])
        for label, lst in results.items():
            for k, d in enumerate(lst):
                for key, val in sorted(d.items()):
                    w.writerow([label, k, key, val])
    with open(os.path.join(outdir, 'results_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'key', 'mean', 'std', 'n_checkpoints'])
        for label, d in agg.items():
            for key, (m, s, n) in sorted(d.items()):
                w.writerow([label, key, m, s, n])
    for fname, delta in (('delta_joint_vs_full.csv', delta_full), ('delta_joint_vs_baseline.csv', delta_base)):
        with open(os.path.join(outdir, fname), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['variant', 'delta_joint_pp', 'ci95_lo', 'ci95_hi'])
            for label, (p, lo, hi) in delta.items():
                w.writerow([label, p, lo, hi])

# =============================================================================
# 6. LOGS ΕΚΠΑΙΔΕΥΣΗΣ (προαιρετικά): καμπύλες Val Acc
# =============================================================================
LOG_RE = re.compile(r'Epoch\s+(\d+)\s*\|\s*Train Loss:\s*([\d.]+)\s*\|\s*Val Loss:\s*([\d.]+)\s*\|\s*'
                    r'Val Acc:\s*([\d.]+)%\s*\(Root:\s*([\d.]+)%,\s*Qual:\s*([\d.]+)%\)')

def parse_training_log(text):
    rows = []
    for m in LOG_RE.finditer(text):
        e, tl, vl, va, vr, vq = m.groups()
        rows.append({'epoch': int(e), 'val_acc': float(va), 'val_root': float(vr), 'val_qual': float(vq)})
    return sorted(rows, key=lambda r: r['epoch'])

def load_logs(log_paths):
    out = {}
    for label, paths in log_paths.items():
        paths = [paths] if isinstance(paths, str) else paths
        runs = []
        for p in paths:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    rows = parse_training_log(f.read())
                if rows:
                    runs.append(rows)
        if runs:
            out[label] = runs
    return out

# =============================================================================
# 7. ΓΡΑΦΗΜΑΤΑ
# =============================================================================
def variant_colors(variants):
    cmap = plt.get_cmap('tab10')
    return {v['label']: cmap(i % 10) for i, v in enumerate(variants)}

def _finish(fig, path, show):
    bottom = 0.075 if fig.legends else 0.0          # χώρος για το υπόμνημα, ώστε να μην πέφτει πάνω στις ετικέτες
    fig.tight_layout(rect=[0, bottom, 1, 1])
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"   🖼️ {path}")
    if show:
        plt.show()
    plt.close(fig)

def _legend(fig, variants, colors, with_hatch):
    handles = [Patch(facecolor=colors[v['label']], label=v['label']) for v in variants]
    if with_hatch:
        handles.append(Patch(facecolor='white', edgecolor='black', hatch='//', label='ο κανόνας ΔΕΝ ήταν ενεργός'))
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 6), frameon=False,
               bbox_to_anchor=(0.5, 0.0))

def draw_grouped_bars(ax, categories, variants, colors, value_fn, err_fn=None, rule_k=None,
                      truth_fn=None):
    series = []
    if truth_fn is not None:
        series.append(('Πραγματικά δεδομένα', '#555555', None))
    for v in variants:
        hatched = rule_k is not None and rule_k in v['removed']
        series.append((v['label'], colors[v['label']], '//' if hatched else None, v))
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(categories))
    for j, s in enumerate(series):
        if s[0] == 'Πραγματικά δεδομένα':
            vals = [truth_fn(c) for c in categories]
            errs = None
        else:
            vals = [value_fn(s[3]['label'], c) for c in categories]
            errs = [err_fn(s[3]['label'], c) for c in categories] if err_fn else None
        ax.bar(x + (j - (n - 1) / 2) * width, vals, width, yerr=errs, color=s[1],
               hatch=s[2], edgecolor='black' if s[2] else 'none', linewidth=0.8, capsize=2)
    ax.set_xticks(x)

def plot_accuracy(agg, variants, srcs, outdir, show):
    colors = variant_colors(variants)
    labels = [v['label'] for v in variants]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    x = np.arange(len(labels))
    for name, marker, off, title in (('root_notes', 's', -0.15, 'Root'), ('qual_notes', '^', 0.0, 'Quality'),
                                      ('joint_notes', 'o', 0.15, 'Joint')):
        ys = [fval(agg, l, f'ALL|acc|{name}') for l in labels]
        es = [ferr(agg, l, f'ALL|acc|{name}') for l in labels]
        ax.errorbar(x + off, ys, yerr=es, fmt=marker, capsize=4, markersize=7, label=title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Ακρίβεια ανά νότα (%)')
    ax.set_title('Ακρίβεια στο πλήρες σύνολο επικύρωσης (όλες οι πηγές μαζί)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    ax2 = axes[1]
    n = len(variants)
    width = 0.8 / n
    xs = np.arange(len(srcs))
    for j, v in enumerate(variants):
        ys = [fval(agg, v['label'], f'{s}|acc|joint_notes') for s in srcs]
        es = [ferr(agg, v['label'], f'{s}|acc|joint_notes') for s in srcs]
        ax2.errorbar(xs + (j - (n - 1) / 2) * width, ys, yerr=es, fmt='o', capsize=3,
                     color=colors[v['label']], label=v['label'])
    ax2.set_xticks(xs)
    ax2.set_xticklabels(srcs)
    ax2.set_ylabel('Joint (%)')
    ax2.set_title('Joint ανά πηγή (μπάρες: 95% διάστημα εμπιστοσύνης ή std ανά seeds)')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(fontsize=8)
    _finish(fig, os.path.join(outdir, 'fig_accuracy.png'), show)

def plot_deltas(delta_base, delta_full, variants, outdir, show):
    colors = variant_colors(variants)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    panels = ((axes[0], delta_base, 'Έναντι baseline: τι προσφέρει κάθε variant (θετικό = βελτίωση)'),
              (axes[1], delta_full, 'Έναντι full: τι χάνεται (αρνητικό = χειρότερο από το full)'))
    for ax, delta, title in panels:
        items = [v for v in variants if v['label'] in delta]
        if not items:
            ax.axis('off')
            continue
        for k, v in enumerate(items):
            p, lo, hi = delta[v['label']]
            ax.errorbar(p, k, xerr=[[p - lo], [hi - p]], fmt='o', color=colors[v['label']],
                        capsize=4, markersize=8)
        ax.axvline(0, color='black', linewidth=0.9)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([v['label'] for v in items])
        ax.invert_yaxis()
        ax.set_xlabel('Δ Joint (ποσοστιαίες μονάδες), 95% διάστημα εμπιστοσύνης')
        ax.set_title(title, fontsize=11)
        ax.grid(axis='x', alpha=0.3)
    _finish(fig, os.path.join(outdir, 'fig_delta_joint.png'), show)

def plot_val_curves(logs, variants, outdir, show):
    if not logs:
        return
    colors = variant_colors(variants)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for v in variants:
        runs = logs.get(v['label'])
        if not runs:
            continue
        for k, rows in enumerate(runs):
            ax.plot([r['epoch'] for r in rows], [r['val_acc'] for r in rows],
                    color=colors[v['label']], alpha=0.9 if k == 0 else 0.4,
                    label=v['label'] if k == 0 else None, linewidth=1.5)
    ax.set_xlabel('Εποχή')
    ax.set_ylabel('Val Acc (%)')
    ax.set_title('Καμπύλες Val Acc από τα logs εκπαίδευσης')
    ax.grid(alpha=0.3)
    ax.legend()
    _finish(fig, os.path.join(outdir, 'fig_val_acc_curves.png'), show)

def plot_paper_metrics(agg, variants, srcs, kind, outdir, show):
    colors = variant_colors(variants)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    cats = ['ALL'] + list(srcs)
    for ax, m in zip(axes.ravel(), METRIC_NAMES):
        draw_grouped_bars(ax, cats, variants, colors,
                          value_fn=lambda l, c, m=m: fval(agg, l, f'{c}|{kind}|{m}'),
                          err_fn=lambda l, c, m=m: agg.get(l, {}).get(f'{c}|{kind}|{m}', (0, 0, 0))[1])
        ax.set_xticklabels(cats)
        ax.set_title(m)
        ax.grid(axis='y', alpha=0.3)
        if kind == 'paper_bias':
            ax.axhline(0, color='black', linewidth=0.8)
    title = ('Μέσο απόλυτο σφάλμα (MAE) των 9 μετρικών: πρόβλεψη έναντι πραγματικών (μικρότερο = καλύτερο)'
             if kind == 'paper_mae' else
             'Bias των 9 μετρικών: πρόβλεψη − πραγματικό (>0: το μοντέλο υπερβαίνει τα δεδομένα)')
    fig.suptitle(title, fontsize=13)
    _legend(fig, variants, colors, with_hatch=False)
    _finish(fig, os.path.join(outdir, 'fig_paper_mae.png' if kind == 'paper_mae' else 'fig_paper_bias.png'), show)

GAP_PANELS = [
    (1, 'R1_gap', 'Κανόνας 1: μετρική διαβάθμιση συμβατότητας'),
    (2, 'R2_gap', 'Κανόνας 2: βηματική κίνηση'),
    (3, 'R3_gap', 'Κανόνας 3: σταθερότητα αρμονικού ρυθμού'),
    (5, 'R5_gap', 'Κανόνας 5: προσαγωγέας που επιλύεται'),
    (6, 'R6_gap', 'Κανόνας 6: μεταβάσεις T/S/D'),
    (7, 'R7_gap', 'Κανόνας 7: τονική κατάληξη'),
    (8, 'R8_gap', 'Κανόνας 8: ποιότητα ανά λειτουργία'),
]

def plot_rule_gaps(agg, variants, srcs, outdir, show):
    colors = variant_colors(variants)
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    cats = ['ALL'] + list(srcs)
    flat_axes = axes.ravel()
    for ax, (k, key, title) in zip(flat_axes, GAP_PANELS):
        draw_grouped_bars(ax, cats, variants, colors,
                          value_fn=lambda l, c, key=key: fval(agg, l, f'{c}|rule|{key}'),
                          err_fn=lambda l, c, key=key: agg.get(l, {}).get(f'{c}|rule|{key}', (0, 0, 0))[1],
                          rule_k=k)
        ax.set_xticklabels(cats)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('απόκλιση από τα δεδομένα (π.μ.)')
        ax.grid(axis='y', alpha=0.3)
    ax = flat_axes[7]
    draw_grouped_bars(ax, cats, variants, colors,
                      value_fn=lambda l, c: fval(agg, l, f'{c}|rule|unknown_chord_frac'),
                      err_fn=lambda l, c: agg.get(l, {}).get(f'{c}|rule|unknown_chord_frac', (0, 0, 0))[1])
    ax.set_xticklabels(cats)
    ax.set_title('Διαγνωστικό: % προβλέψεων NC/OTHER\n(μετρούν πάντα ως "συμβατές")', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Τήρηση των κανόνων: απόκλιση της συμπεριφοράς των προβλέψεων από τα πραγματικά '
                 'δεδομένα (μικρότερη = καλύτερα). Γραμμοσκιασμένες μπάρες: ο κανόνας έλειπε από το variant.',
                 fontsize=12)
    _legend(fig, variants, colors, with_hatch=True)
    _finish(fig, os.path.join(outdir, 'fig_rule_gaps.png'), show)

def plot_rule_heatmaps(agg, variants, srcs, outdir, show):
    cats = ['ALL'] + list(srcs)
    n_cols = 2
    n_rows = int(np.ceil(len(cats) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 3.2 * n_rows + 1.5), squeeze=False)
    labels = [v['label'] for v in variants]
    for ax in axes.ravel()[len(cats):]:
        ax.axis('off')
    for ax, src in zip(axes.ravel(), cats):
        M = np.array([[fval(agg, l, f'{src}|rule|R{k}_gap') for k in RULE_COLS] for l in labels])
        col_max = np.nanmax(M, axis=0)
        col_max[(col_max == 0) | np.isnan(col_max)] = 1.0
        ax.imshow(M / col_max, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                if not np.isnan(M[r, c]):
                    dark = (M[r, c] / col_max[c]) > 0.55
                    ax.text(c, r, f'{M[r, c]:.1f}', ha='center', va='center', fontsize=9,
                            color='white' if dark else 'black')
                if RULE_COLS[c] in variants[r]['removed']:
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='blue', linewidth=2.2))
        ax.set_xticks(range(len(RULE_COLS)))
        ax.set_xticklabels([f'Κ{k}' for k in RULE_COLS])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f'{src}', fontsize=11)
    fig.suptitle('Απόκλιση τήρησης κανόνων (π.μ., μικρότερη = καλύτερα, χρώμα ανά στήλη). '
                 'Μπλε πλαίσιο: ο κανόνας έλειπε από το variant.', fontsize=12)
    _finish(fig, os.path.join(outdir, 'fig_rule_heatmap.png'), show)

LEVEL_PANELS = [
    (1, 'Κανόνας 1: % παραβίασης συμβατότητας ανά τύπο χρόνου',
     [('Ισχυρός', 'R1_viol_strong'), ('Ασθενής', 'R1_viol_weak'), ('Off-beat', 'R1_viol_off')]),
    (2, 'Κανόνας 2: % παραβίασης (βηματική / όχι)',
     [('Βηματική', 'R2_viol_step'), ('Όχι βηματική', 'R2_viol_nonstep')]),
    (3, 'Κανόνας 3: % "ίδια συγχορδία με την προηγούμενη νότα"',
     [('Ισχυρός', 'R3_same_strong'), ('Ασθενής', 'R3_same_weak'), ('Off-beat', 'R3_same_off')]),
    (5, 'Κανόνας 5: % παραβίασης σε προσαγωγέα που επιλύεται', [('', 'R5_viol_lt')]),
    (6, 'Κανόνας 6: P(D → T | αλλαγή από D) (%)', [('', 'R6_DT')]),
    (7, 'Κανόνας 7: % τραγουδιών με τέλος σε τονική', [('', 'R7_end_tonic')]),
]

def plot_rule_levels(agg, truth_flat, variants, src, outdir, show):
    colors = variant_colors(variants)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (k, title, items) in zip(axes.ravel(), LEVEL_PANELS):
        cats = [it[0] for it in items]
        keys = {it[0]: it[1] for it in items}
        draw_grouped_bars(ax, cats, variants, colors,
                          value_fn=lambda l, c, keys=keys: fval(agg, l, f'{src}|rule|{keys[c]}'),
                          err_fn=lambda l, c, keys=keys: agg.get(l, {}).get(f'{src}|rule|{keys[c]}', (0, 0, 0))[1],
                          rule_k=k,
                          truth_fn=lambda c, keys=keys: truth_flat.get(f'{src}|rule|{keys[c]}', float('nan')))
        ax.set_xticklabels(cats)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle(f'Τιμές των κανόνων ({src}): πρόβλεψη έναντι πραγματικών δεδομένων (γκρι μπάρα). '
                 'Γραμμοσκιασμένες μπάρες: ο κανόνας έλειπε.', fontsize=12)
    handles = [Patch(facecolor='#555555', label='Πραγματικά δεδομένα')]
    handles += [Patch(facecolor=colors[v['label']], label=v['label']) for v in variants]
    handles.append(Patch(facecolor='white', edgecolor='black', hatch='//', label='ο κανόνας ΔΕΝ ήταν ενεργός'))
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 7), frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    _finish(fig, os.path.join(outdir, f'fig_rule_levels_{src}.png'), show)

# =============================================================================
# 8. ΕΚΤΥΠΩΣΗ ΠΕΡΙΛΗΨΗΣ
# =============================================================================
def print_summary(agg, variants, srcs, delta_full, delta_base, logs):
    labels = [v['label'] for v in variants]
    w = max(len(l) for l in labels) + 2
    print("\n" + "=" * 100)
    print("ΑΚΡΙΒΕΙΑ ΑΝΑ ΝΟΤΑ στο πλήρες σύνολο επικύρωσης (%)")
    print("=" * 100)
    print(f"{'variant':{w}} {'Root':>7} {'Qual':>7} {'Joint':>7}  " + ''.join(f"{s:>9}" for s in srcs))
    for l in labels:
        row = (f"{l:{w}} {fval(agg, l, 'ALL|acc|root_notes'):7.2f} {fval(agg, l, 'ALL|acc|qual_notes'):7.2f} "
               f"{fval(agg, l, 'ALL|acc|joint_notes'):7.2f}  ")
        row += ''.join(f"{fval(agg, l, f'{s}|acc|joint_notes'):9.2f}" for s in srcs)
        print(row)
    def fmt(d, l):
        if l not in d:
            return '—'
        p, lo, hi = d[l]
        return f"{p:+.2f} [{lo:+.2f}, {hi:+.2f}]" + (' ~' if lo <= 0 <= hi else '  ')
    print("\nΔ Joint (ALL), σε π.μ. με 95% CI. '~' = εντός θορύβου (το διάστημα περιέχει το 0)")
    print(f"{'variant':{w}} {'έναντι baseline (επάρκεια)':>34} {'έναντι full (αναγκαιότητα)':>34}")
    for l in labels:
        print(f"{l:{w}} {fmt(delta_base, l):>34} {fmt(delta_full, l):>34}")
    if logs:
        print("\nΜέσος Val Acc των ΤΕΛΕΥΤΑΙΩΝ 10 εποχών από τα logs:")
        for l in labels:
            if l in logs:
                vals = [np.mean([r['val_acc'] for r in run[-10:]]) for run in logs[l]]
                print(f"   {l:{w}} {np.mean(vals):6.2f}%  (εποχές: {[run[-1]['epoch'] for run in logs[l]]})")
    print("\n" + "=" * 100)
    print("9 ΜΕΤΡΙΚΕΣ ΤΩΝ PAPERS: MAE (ALL)")
    print("=" * 100)
    print(f"{'variant':{w}}" + ''.join(f"{m:>9}" for m in METRIC_NAMES))
    for l in labels:
        print(f"{l:{w}}" + ''.join(f"{fval(agg, l, f'ALL|paper_mae|{m}'):9.3f}" for m in METRIC_NAMES))
    print("\n" + "=" * 100)
    print("ΤΗΡΗΣΗ ΚΑΝΟΝΩΝ: απόκλιση από τα πραγματικά δεδομένα (π.μ., ALL). * = ο κανόνας έλειπε")
    print("=" * 100)
    print(f"{'variant':{w}}" + ''.join(f"{'Κ' + str(k):>9}" for k in RULE_COLS))
    for v in variants:
        row = f"{v['label']:{w}}"
        for k in RULE_COLS:
            val = fval(agg, v['label'], f'ALL|rule|R{k}_gap')
            row += f"{val:8.2f}{'*' if k in v['removed'] else ' '}"
        print(row)

# =============================================================================
# ΚΥΡΙΟ ΠΡΟΓΡΑΜΜΑ
# =============================================================================
def write_truth_means(truth_paper_means, outdir):
    """Μέση τιμή κάθε μετρικής στις ΠΡΑΓΜΑΤΙΚΕΣ αρμονίες (για σύγκριση με τον Πίνακα Ι του paper)."""
    with open(os.path.join(outdir, 'truth_paper_means.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['source', 'metric', 'mean_over_real_harmonizations'])
        for src, d in truth_paper_means.items():
            for mn, v in d.items():
                w.writerow([src, mn, v])

def write_truth_rule_levels(truth_counts, outdir):
    """Τιμές των κανόνων στις ΠΡΑΓΜΑΤΙΚΕΣ αρμονίες, ανά πηγή (χρειάζονται για το σχήμα fig_rule_levels)."""
    with open(os.path.join(outdir, 'truth_rule_levels.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['source', 'name', 'value'])
        for src, cnt in truth_counts.items():
            for k, v in rule_levels(cnt).items():
                w.writerow([src, k, v])

def print_relative_errors(agg, variants, truth_paper_means, src='ALL'):
    """Σχετικό σφάλμα = MAE / μέση πραγματική τιμή (%). Αφαιρεί τη διαφορά κλίμακας ανάμεσα σε διαφορετικά
    σύνολα δεδομένων και πλέγματα χρόνου, και είναι πιο κοντά σε σύγκριση με το paper αναφοράς."""
    labels = [v['label'] for v in variants]
    w = max(len(l) for l in labels) + 2
    tm = truth_paper_means[src]
    print("\n" + "=" * 100)
    print(f"ΣΧΕΤΙΚΟ ΣΦΑΛΜΑ ΤΩΝ 9 ΜΕΤΡΙΚΩΝ ({src}): MAE / μέση τιμή των πραγματικών αρμονιών, σε %")
    print("=" * 100)
    print(f"{'μετρική':{w}}" + ''.join(f"{mn:>9}" for mn in METRIC_NAMES))
    print(f"{'μέση πραγματική':{w}}" + ''.join(f"{tm[mn]:9.3f}" for mn in METRIC_NAMES))
    for l in labels:
        row = f"{l:{w}}"
        for mn in METRIC_NAMES:
            mae = fval(agg, l, f'{src}|paper_mae|{mn}')
            row += f"{100.0 * mae / tm[mn]:9.1f}" if tm[mn] else f"{'—':>9}"
        print(row)

def find_ckpt_root():
    """Επιστρέφει τη διαδρομή του φακέλου 'final_data' (ή None)."""
    if CKPT_ROOT:
        return CKPT_ROOT if os.path.isdir(CKPT_ROOT) else None
    base_depth = INPUT_SEARCH_ROOT.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, _ in os.walk(INPUT_SEARCH_ROOT):
        if dirpath.count(os.sep) - base_depth >= 5:      # όριο βάθους, για να μη σαρώνει τα πάντα
            dirnames[:] = []
            continue
        if os.path.basename(dirpath) == 'final_data':
            return dirpath
    return None

def main():
    # Ξεχωριστός φάκελος (και cache προβλέψεων) ανά πρωτόκολλο φόρτωσης βαρών, ώστε οι εκτελέσεις
    # 'final' και 'best' να μην αναμειγνύονται ούτε να ξαναχρησιμοποιούν τις προβλέψεις η μία της άλλης.
    assert PROTOCOL in ('final', 'best'), "Το PROTOCOL πρέπει να είναι 'final' ή 'best'"
    tag = PROTOCOL
    fname = FINAL_FILE if PROTOCOL == 'final' else BEST_FILE
    out_dir = f"{OUTPUT_DIR}_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 Πρωτόκολλο: '{PROTOCOL}' (αρχείο: {fname}) -> αποτελέσματα στο {out_dir}")
    root = None
    if not ONLY_TRUTH_MEANS:
        root = find_ckpt_root()
        if root is None:
            print("❌ Δεν βρέθηκε ο φάκελος 'final_data'. Βάλε τη διαδρομή του στο CKPT_ROOT (μπλοκ ρυθμίσεων).")
            return
        print(f"📂 Φάκελος checkpoints: {root}")
    print("⏳ Φόρτωση corpus + λεξικού (PDMX + OpenEWLD + CMD + POP909)...")
    bundle = load_corpus_and_vocab_v2(include_hooktheory=False, include_cmd=True, include_pop909=True)
    groups, all_ids = split_val_by_source(bundle, MAX_SONGS_PER_SOURCE)
    print("   Τραγούδια επικύρωσης ανά πηγή: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items())
          + f" (σύνολο {len(all_ids)})")

    print("⏳ Προετοιμασία δεδομένων ανά τραγούδι + πραγματικές τιμές μετρικών/κανόνων...")
    cache = {}
    for k, i in enumerate(all_ids):
        cache[i] = build_song_cache(bundle, i)
        if (k + 1) % 1000 == 0:
            print(f"   ... {k + 1}/{len(all_ids)}")
    srcs_all = {'ALL': all_ids}
    srcs_all.update(groups)
    if ONLY_TRUTH_MEANS:
        tpm = {src: {mn: float(np.mean([cache[i]['truth_paper'][mn] for i in ids])) for mn in METRIC_NAMES}
               for src, ids in srcs_all.items()}
        write_truth_means(tpm, out_dir)
        print("\n" + "=" * 100)
        print("ΜΕΣΗ ΤΙΜΗ ΤΩΝ 9 ΜΕΤΡΙΚΩΝ ΣΤΙΣ ΠΡΑΓΜΑΤΙΚΕΣ ΑΡΜΟΝΙΕΣ (σύνολο επικύρωσης)")
        print("=" * 100)
        print(f"{'πηγή':10}" + ''.join(f"{mn:>9}" for mn in METRIC_NAMES))
        for src, d in tpm.items():
            print(f"{src:10}" + ''.join(f"{d[mn]:9.4f}" for mn in METRIC_NAMES))
        tc = {}
        for src, ids in srcs_all.items():
            cnt = Counter()
            for i in ids:
                cnt.update(rule_counts_for_song(cache[i], cache[i]['true_root'], cache[i]['true_qual']))
            tc[src] = cnt
        with open(os.path.join(out_dir, 'truth_rule_levels.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['source', 'name', 'value'])
            for src, cnt in tc.items():
                for k, v in rule_levels(cnt).items():
                    w.writerow([src, k, v])
        print(f"\n✅ Αποθηκεύτηκαν: {os.path.join(out_dir, 'truth_paper_means.csv')} και truth_rule_levels.csv")
        return
    truth_counts = {}
    for src, ids in srcs_all.items():
        cnt = Counter()
        for i in ids:
            cnt.update(rule_counts_for_song(cache[i], cache[i]['true_root'], cache[i]['true_qual']))
        truth_counts[src] = cnt
    truth_flat = {f'{src}|rule|{k}': v for src, cnt in truth_counts.items() for k, v in rule_levels(cnt).items()}
    truth_paper_means = {src: {mn: float(np.mean([cache[i]['truth_paper'][mn] for i in ids])) for mn in METRIC_NAMES}
                         for src, ids in srcs_all.items()}
    write_truth_means(truth_paper_means, out_dir)
    write_truth_rule_levels(truth_counts, out_dir)

    results, per_song_by_variant = {}, {}
    for v in VARIANTS:
        paths = [os.path.join(root, folder, fname) for folder in v['folders']]
        for k, path in enumerate(paths):
            if not os.path.exists(path):
                print(f"⚠️ Δεν βρέθηκε το checkpoint, παραλείπεται: {path}")
                continue
            print(f"\n▶ {v['label']} (checkpoint {k + 1}/{len(paths)}): {path}")
            cache_file = os.path.join(out_dir, 'preds', f"{safe_name(v['label'])}_{k}.pkl")
            preds = predict_checkpoint(bundle, path, all_ids, cache_file)
            flat, ps = analyze_checkpoint(cache, groups, all_ids, preds, truth_counts)
            results.setdefault(v['label'], []).append(flat)
            per_song_by_variant.setdefault(v['label'], []).append(ps)
            print(f"   Joint (ALL) = {flat['ALL|acc|joint_notes']:.2f}%")
    if not results:
        print("❌ Δεν αναλύθηκε κανένα checkpoint, έλεγξε τις διαδρομές στο μπλοκ ρυθμίσεων.")
        return

    variants = [v for v in VARIANTS if v['label'] in results]
    srcs = list(groups.keys())
    agg = aggregate(results)
    delta_full = compute_delta(per_song_by_variant, FULL_LABEL, BOOTSTRAP_B, seed=1)
    delta_base = compute_delta(per_song_by_variant, BASELINE_LABEL, BOOTSTRAP_B, seed=2)
    logs = load_logs(LOG_PATHS)

    write_csvs(results, agg, delta_full, delta_base, out_dir)
    print_summary(agg, variants, srcs, delta_full, delta_base, logs)
    print_relative_errors(agg, variants, truth_paper_means)

    print("\n🖼️ Δημιουργία γραφημάτων...")
    plot_val_curves(logs, variants, out_dir, SHOW_PLOTS)
    plot_accuracy(agg, variants, srcs, out_dir, SHOW_PLOTS)
    plot_deltas(delta_base, delta_full, variants, out_dir, SHOW_PLOTS)
    plot_paper_metrics(agg, variants, srcs, 'paper_mae', out_dir, SHOW_PLOTS)
    plot_paper_metrics(agg, variants, srcs, 'paper_bias', out_dir, SHOW_PLOTS)
    plot_rule_gaps(agg, variants, srcs, out_dir, SHOW_PLOTS)
    plot_rule_heatmaps(agg, variants, srcs, out_dir, SHOW_PLOTS)
    for src in ['ALL'] + srcs:
        plot_rule_levels(agg, truth_flat, variants, src, out_dir, SHOW_PLOTS)
    print(f"\n✅ Τα αποτελέσματα (CSV και γραφήματα) αποθηκεύτηκαν στο {out_dir}")

if __name__ == '__main__':
    main()
