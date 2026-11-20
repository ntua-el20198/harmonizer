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
# ΣΚΟΠΟΣ: αξιολόγηση ΚΑΙ ΤΩΝ 4 checkpoints (classics+cmd+pop909, classics+cmd,
# classics+pop909, classics-only) σε τρία πιθανά held-out test batches -
# "classics" (PDMX+EWLD, κοινό και δίκαιο ανάμεσα και στα 4), "CMD" (μόνο για
# τα checkpoints που το περιλαμβάνουν) και "POP909" (μόνο για τα checkpoints
# που το περιλαμβάνουν) - ώστε να συγκριθούν όλα σε μία ενιαία εικόνα, πριν
# αποφασιστεί ποιο θα χρησιμοποιηθεί ως βάση για τη μελέτη αφαίρεσης (ablation).
# =============================================================================
# ΔΕΝ χρειάζονται οι 8 κανόνες/compatibility matrix εδώ - μόνο τα βασικά feature
# vocabularies (1Δ) χρειάζονται για το forward pass και τη σύγκριση με τα
# πραγματικά root/quality.
# =============================================================================

CORPUS_PATH = os.path.join(DATA_DIR, 'final_corpus.pkl')
LOOKUP_PATH = os.path.join(DATA_DIR, 'symbolic_lookup.pkl')
EXTRACTED_JSON_PATH = os.path.join(DATA_DIR1, 'extracted_songs.json')
HOOKTHEORY_JSON_PATH = os.path.join(DATA_DIR1, 'extracted_hooktheory.json')
POP909_JSON_PATH = os.path.join(DATA_DIR1, 'extracted_pop909.json')

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
    for melody_iris, chord_iris in raw_songs:
        if melody_iris and melody_iris[0] in _synthetic_iri_to_is_major:
            shift = 0
        else:
            raw_pcs = []
            for n in melody_iris:
                st = pitch_name_to_semitone(note_features.get(n, {}).get('pitch'))
                if st is not None:
                    raw_pcs.append(st % 12)
            tonic, mode = detect_key(raw_pcs)
            shift = ((0 - tonic) % 12) if mode == 'major' else ((9 - tonic) % 12)
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
    state_dict = checkpoint.get('best_model_state_dict') or checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    print(f"   ✅ Φορτώθηκε {checkpoint_path} (εποχή {checkpoint.get('best_epoch', checkpoint.get('epoch'))})")
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
# ΚΥΡΙΟ ΠΡΟΓΡΑΜΜΑ - ΕΝΑ checkpoint, αξιολόγηση με τις 9 μετρικές των δύο
# papers πάνω σε held-out test set (ίδιο πνεύμα με τους Πίνακες 2/3 τους).
# =============================================================================
N_SONGS = 50
INCLUDE_CMD = True       # <-- ΑΛΛΑΞΕ ανάλογα με το checkpoint σου
INCLUDE_POP909 = True    # <-- ΑΛΛΑΞΕ ανάλογα με το checkpoint σου
CHECKPOINT_PATH = '/kaggle/input/datasets/kostisinthisb/music-ontology-data/best_val_acc_clcmdpop.pth'  # παλιά εξαγωγή, εποχή 127

print("⏳ Φόρτωση corpus + λεξικού...")
bundle = load_corpus_and_vocab_v2(include_hooktheory=False, include_cmd=INCLUDE_CMD, include_pop909=INCLUDE_POP909)

print("\n⏳ Φόρτωση μοντέλου...")
model, mask_chord_id = build_and_load_model(bundle, CHECKPOINT_PATH)
print(f"   Beat-type labels στο λεξιλόγιο: {sorted(set(bundle['beat_type_idx_to_label'].values()))}")
print(f"   Duration labels στο λεξιλόγιο: {sorted(set(bundle['duration_idx_to_label'].values()))}")

random.seed(123)
classical_pool = [i for i in range(bundle['base_classical_count'])
                   if i in bundle['val_only_indices'] and 20 <= len(bundle['raw_songs'][i][0]) <= 300]
classical_sample = random.sample(classical_pool, min(N_SONGS, len(classical_pool)))
classical_songs = [bundle['raw_songs'][i] for i in classical_sample]
print(f"\n📦 Batch classics: {len(classical_songs)} τραγούδια (held-out, pool={len(classical_pool)}).")

res_classical = evaluate_batch_all_metrics(model, mask_chord_id, bundle, classical_songs, "classics")

if INCLUDE_CMD:
    random.seed(456)
    cmd_pool = [i for i in range(bundle['base_classical_count'], bundle['classical_song_count'])
                if i in bundle['val_only_indices'] and 20 <= len(bundle['raw_songs'][i][0]) <= 300]
    cmd_sample = random.sample(cmd_pool, min(N_SONGS, len(cmd_pool)))
    cmd_songs = [bundle['raw_songs'][i] for i in cmd_sample]
    print(f"\n📦 Batch CMD: {len(cmd_songs)} τραγούδια (held-out, pool={len(cmd_pool)}).")
    res_cmd = evaluate_batch_all_metrics(model, mask_chord_id, bundle, cmd_songs, "CMD")

if INCLUDE_POP909:
    random.seed(789)
    pop_pool = [i for i in range(bundle['pop909_start_index'], len(bundle['raw_songs']))
                if i in bundle['val_only_indices'] and 100 <= len(bundle['raw_songs'][i][0]) <= 500]
    pop_sample = random.sample(pop_pool, min(N_SONGS, len(pop_pool)))
    pop_songs = [bundle['raw_songs'][i] for i in pop_sample]
    print(f"\n📦 Batch POP909: {len(pop_songs)} τραγούδια (held-out, pool={len(pop_pool)}).")
    res_pop = evaluate_batch_all_metrics(model, mask_chord_id, bundle, pop_songs, "POP909")

# =============================================================================
# ΣΥΝΔΥΑΣΜΕΝΗ ΑΞΙΟΛΟΓΗΣΗ: ΟΛΑ τα datasets μαζί, ΣΤΑΘΜΙΣΜΕΝΗ με την ΠΡΑΓΜΑΤΙΚΗ
# κατανομή του validation set (πόσα held-out τραγούδια έχει κάθε πηγή), ΟΧΙ
# 50+50+50. Κάθε μέτρηση είναι μέσος όρος ανά τραγούδι, οπότε ο σωστός
# συνδυασμός είναι ο μέσος όρος των batches με βάρη = μερίδιο της κάθε πηγής
# στα held-out τραγούδια. Έτσι δεν χρειάζεται να δειγματολειπτούμε
# αναλογικά (που θα άφηνε ~3 τραγούδια CMD).
# =============================================================================
_val = bundle['val_only_indices']
_counts = {"classics": sum(1 for i in _val if i < bundle['base_classical_count'])}
if INCLUDE_CMD:
    _counts["CMD"] = sum(1 for i in _val if bundle['base_classical_count'] <= i < bundle['classical_song_count'])
if INCLUDE_POP909:
    _counts["POP909"] = sum(1 for i in _val if i >= bundle['pop909_start_index'])
_batch_results = {"classics": res_classical}
if INCLUDE_CMD:
    _batch_results["CMD"] = res_cmd
if INCLUDE_POP909:
    _batch_results["POP909"] = res_pop

if len(_batch_results) > 1:
    _tot = sum(_counts.values())
    _w = {name: _counts[name] / _tot for name in _batch_results}
    print("\n" + "=" * 78)
    print("ΣΥΝΔΥΑΣΜΕΝΟ - σταθμισμένο με την ΠΡΑΓΜΑΤΙΚΗ κατανομή του validation set")
    print("   βάρη: " + ", ".join(f"{n}={_counts[n]} τραγούδια ({_w[n]*100:.1f}%)" for n in _batch_results))
    print("=" * 78)
    c_mae = {k: sum(_w[n] * _batch_results[n][0][k] for n in _batch_results) for k in METRIC_NAMES}
    c_bias = {k: sum(_w[n] * _batch_results[n][1][k] for n in _batch_results) for k in METRIC_NAMES}
    c_acc = {k: sum(_w[n] * _batch_results[n][2][k] for n in _batch_results) for k in ('Root', 'Qual', 'Joint')}
    print(f"   {'':8}" + ''.join(f"{k:>9}" for k in METRIC_NAMES))
    print(f"   {'MAE':8}" + ''.join(f"{c_mae[k]:>9.4f}" for k in METRIC_NAMES))
    print(f"   {'bias':8}" + ''.join(f"{c_bias[k]:>+9.4f}" for k in METRIC_NAMES))
    print(f"   Ακρίβεια (μέσος όρος ανά τραγούδι): Root={c_acc['Root']:.1f}% Qual={c_acc['Qual']:.1f}% Joint={c_acc['Joint']:.1f}%")
    print("   ΣΗΜΕΙΩΣΗ: επειδή τα classics είναι ~94% των τραγουδιών, το συνδυασμένο")
    print("   αποτέλεσμα θα είναι κοντά στα classics - οι γραμμές ανά dataset παραπάνω")
    print("   είναι αυτές που δείχνουν τη διαφορά ανάμεσα στα είδη.")
