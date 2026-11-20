import os
# --- Paths (override with environment variables; defaults = original Kaggle layout) ---
DATA_DIR = os.environ.get("HARMONIZER_DATA", "/kaggle/input/datasets/kostissp/music-ontology-data")    # final_corpus.pkl, symbolic_lookup.pkl
DATA_DIR1 = os.environ.get("HARMONIZER_DATA1", "/kaggle/input/datasets/kostissp/music-ontology-data1")  # extracted_*.json
OUT_DIR = os.environ.get("HARMONIZER_OUT", "/kaggle/working")                                          # checkpoints / results
# pip install x-transformers  (see requirements.txt)

import shutil, os
import pickle
import math
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# ΚΑΝΟΝΑΣ ΕΝΟΣ ΝΕΟΥ ΠΕΙΡΑΜΑΤΟΣ: αντικαθιστούμε το nn.TransformerEncoder (απόλυτη,
# sinusoidal θέση) με το Encoder του x-transformers (σχετική θέση, T5-style
# rel_pos_bias) - το ίδιο μοτίβο που το Music Transformer (Huang et al., 2018)
# έδειξε ότι δουλεύει καλύτερα ειδικά για συμβολική μουσική, ακριβώς επειδή
# μουσική δομή (επανάληψη, φράσεις) εξαρτάται από ΣΧΕΤΙΚΗ απόσταση, όχι απόλυτη
# θέση - συνδέεται άμεσα με τις ανησυχίες μας για το positional encoding στο
# whole-song training. ΑΠΑΙΤΕΙ: pip install x-transformers (--break-system-packages
# αν χρειαστεί) σε ΞΕΧΩΡΙΣΤΟ κελί πριν τρέξει αυτό το script.
from x_transformers import Encoder as XTEncoder
import numpy as np
from collections import Counter
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Εκπαίδευση στη συσκευή: {device}")
EMBEDDING_DIM = 256
# Πόσο βαραίνει το tonal-distance loss έναντι του κύριου quality/root loss.
# ΔΕΝ βαθμονομείται από τα δεδομένα (δεν είναι "πόσο σπάνια παραβιάζεται κάτι").
TONAL_DISTANCE_WEIGHT = 0.3

# =============================================================================
# TONAL-DISTANCE LOSS (v2 - ΑΝΤΙΚΑΘΙΣΤΑ το προηγούμενο "απόλυτες νότες" head,
# που ΔΟΚΙΜΑΣΤΗΚΕ και ΔΕΝ έδειξε καμία βελτίωση - βλ. diagnose_wrong_
# prediction_distance.py: 3.681 vs 3.674 μέση απόσταση σε λάθος προβλέψεις,
# αμελητέα διαφορά). Η ΝΕΑ προσέγγιση είναι ΑΜΕΣΗ, όχι ξεχωριστό head:
# υπολογίζει το ΑΝΑΜΕΝΟΜΕΝΟ σημείο του ΠΡΑΓΜΑΤΙΚΟΥ root+quality πρόβλεψης του
# μοντέλου στον 6-διάστατο τονικό χώρο του Harte, Sandler & Gasser (2006)
# "Detecting Harmonic Change in Musical Audio", και το συγκρίνει ΑΠΕΥΘΕΙΑΣ με
# το σημείο της αληθινής χορδής. Βλ. harte_centroid_test.py και
# expected_centroid_test.py για την επικύρωση πριν ενσωματωθεί εδώ -
# επιβεβαιώθηκε ότι: σωστό<κοντινό λάθος<μακρινό λάθος, με τη ΣΩΣΤΗ σειρά.
# ΜΟΝΙΜΟ global (όχι μόνο στο checkpoint-loading cell όπως πριν) - χρειάζεται
# από το chord_display_name ΑΝΕΞΑΡΤΗΤΑ από το αν φορτώνουμε checkpoint ή
# εκπαιδεύουμε από την αρχή στην ΙΔΙΑ session (βλ. bug που βρέθηκε στο Cell 5).
PC_TO_DISPLAY = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

QUALITY_TONES = {
    'maj': {0,4,7}, 'm': {0,3,7}, '7': {0,4,7,10}, 'm7': {0,3,7,10},
    'maj7': {0,4,7,11}, 'dim': {0,3,6}, '6': {0,4,7,9}, '9': {0,4,7,10,2},
    'ø7': {0,3,6,10}, 'sus': {0,5,7}, 'm6': {0,3,7,9}, 'o7': {0,3,6,9},
    'aug': {0,4,8}, '7aug': {0,4,8,10}, 'm9': {0,3,7,10,2}, 'power': {0,7},
    '13': {0,4,7,10,9}, 'sus2': {0,2,7}, 'M9': {0,4,7,11,2},
    'm11': {0,3,7,10,2,5}, 'mM7': {0,3,7,11}, '11': {0,4,7,10,2,5},
    'NC': set(),  # κανένα σημείο - ΈΓΚΥΡΟ (κεντροειδές = μηδέν, βλ. παρακάτω)
}
HARTE_R1, HARTE_R2, HARTE_R3 = 1.0, 1.0, 0.5  # ακτίνες: πέμπτες, ελάσσονες τρίτες, μείζονες τρίτες

def _harte_centroid_6d(pitch_classes):
    """Μετατρέπει ένα σύνολο pitch classes (0-11) στο 6D κεντροειδές του Harte
    et al. (2006). Άδειο σύνολο (NC) -> μηδενικό διάνυσμα."""
    if not pitch_classes:
        return [0.0]*6
    x = [0.0]*6
    for l in pitch_classes:
        x[0] += HARTE_R1 * math.sin(l*7*math.pi/6); x[1] += HARTE_R1 * math.cos(l*7*math.pi/6)
        x[2] += HARTE_R2 * math.sin(l*3*math.pi/2); x[3] += HARTE_R2 * math.cos(l*3*math.pi/2)
        x[4] += HARTE_R3 * math.sin(l*2*math.pi/3); x[5] += HARTE_R3 * math.cos(l*2*math.pi/3)
    return [v/len(pitch_classes) for v in x]

def build_centroid_table(idx_to_root_pc, idx_to_quality, num_root_classes, num_quality_classes):
    """Πίνακας [num_root, num_quality, 6] με το Harte κεντροειδές ΚΑΘΕ (root,
    quality) συνδυασμού, ΜΙΑ φορά πριν την εκπαίδευση. Για 'OTHER' (χωρίς
    σταθερό pitch-content) ή UNKNOWN_ROOT+πραγματική ποιότητα, βάζουμε ένα
    ΜΑΚΡΙΝΟ, σταθερό placeholder κεντροειδές ώστε να ΜΗΝ μπερδεύεται κατά λάθος
    με ένα πραγματικό, κοντινό σημείο (η μάσκα παρακάτω τα αγνοεί όπως και πριν,
    αυτό είναι απλά άμυνα σε βάθος)."""
    table = torch.zeros(num_root_classes, num_quality_classes, 6)
    mask = torch.zeros(num_root_classes, num_quality_classes, dtype=torch.bool)
    for r in range(num_root_classes):
        root_pc = idx_to_root_pc.get(r, 'UNKNOWN_ROOT')
        for q in range(num_quality_classes):
            quality_name = idx_to_quality.get(q, 'OTHER')
            tones = QUALITY_TONES.get(quality_name)
            if tones is None:
                continue  # 'OTHER' - mask παραμένει False
            if quality_name == 'NC':
                mask[r, q] = True  # κεντροειδές ήδη μηδέν, σωστό
                continue
            if not isinstance(root_pc, int):
                continue  # UNKNOWN_ROOT + πραγματική ποιότητα - χωρίς νόημα
            absolute = {(root_pc + t) % 12 for t in tones}
            table[r, q] = torch.tensor(_harte_centroid_6d(absolute))
            mask[r, q] = True
    return table, mask

def build_tone_vector_table(idx_to_root_pc, idx_to_quality, num_root_classes, num_quality_classes):
    """Πίνακας [num_root, num_quality, 12] με 12-bit ΔΥΑΔΙΚΟ διάνυσμα (ποιες
    απόλυτες νότες υπάρχουν) ΚΑΘΕ (root, quality) συνδυασμού. ΔΕΝ χρησιμοποιείται
    στο loss - είναι ΚΑΘΑΡΑ για ΜΕΤΡΗΣΗ 'κοντινής' ακρίβειας ΖΩΝΤΑΝΑ κατά τη
    διάρκεια της επικύρωσης (βλ. diagnose_near_match_accuracy.py για την
    αρχική, εκτός-εκπαίδευσης έκδοση αυτής της ιδέας)."""
    table = torch.zeros(num_root_classes, num_quality_classes, 12)
    mask = torch.zeros(num_root_classes, num_quality_classes, dtype=torch.bool)
    for r in range(num_root_classes):
        root_pc = idx_to_root_pc.get(r, 'UNKNOWN_ROOT')
        for q in range(num_quality_classes):
            quality_name = idx_to_quality.get(q, 'OTHER')
            tones = QUALITY_TONES.get(quality_name)
            if tones is None:
                continue  # 'OTHER' - mask παραμένει False
            if quality_name == 'NC':
                mask[r, q] = True  # διάνυσμα ήδη μηδέν, σωστό ("καμία νότα")
                continue
            if not isinstance(root_pc, int):
                continue
            absolute = {(root_pc + t) % 12 for t in tones}
            for pc in absolute:
                table[r, q, pc] = 1.0
            mask[r, q] = True
    return table, mask

# =============================================================================
# ΓΙΑ ΤΟ HarmonizerTransformerIterative (#1 + #6 από την έρευνα βιβλιογραφίας):
# curriculum μερικής αποκάλυψης πραγματικών χορδών κατά την εκπαίδευση. Βλ.
# curriculum_test.py για την επικύρωση της λογικής πριν ενσωματωθεί εδώ.
# =============================================================================
def build_root_qual_to_flat_table(flat_key_to_idx, num_root_classes, num_quality_classes, mask_chord_id):
    """Πίνακας [num_root, num_quality] -> flat chord index (η ΙΔΙΑ flat vocabulary
    που ήδη υπάρχει από το corpus setup, βλ. flat_key_to_idx). Συνδυασμοί που ΔΕΝ
    εμφανίζονται ΠΟΤΕ στο corpus παίρνουν mask_chord_id ως ασφαλές fallback - ΔΕΝ
    θα έπρεπε ποτέ να χρειαστεί στην πράξη, αφού κάθε ΑΛΗΘΙΝΗ (root,quality) τιμή
    στα δεδομένα έχει εξ ορισμού flat index (έτσι χτίστηκε το flat_key_to_idx)."""
    table = torch.full((num_root_classes, num_quality_classes), mask_chord_id, dtype=torch.long)
    valid = torch.zeros(num_root_classes, num_quality_classes, dtype=torch.bool)
    for (r_idx, q_idx), flat_idx in flat_key_to_idx.items():
        table[r_idx, q_idx] = flat_idx
        valid[r_idx, q_idx] = True
    return table, valid

def curriculum_reveal_fraction(current_step, total_steps):
    """v = (k/k_total)^5 - ΙΔΙΟΣ τύπος με Kaliakatsos-Papakostas et al. Ο εκθέτης
    5 σημαίνει ότι ΠΕΡΙΠΟΥ ο μισός χρόνος εκπαίδευσης μένει σε σχεδόν-πλήρη
    απόκρυψη (βλ. curriculum_test.py: στο 50% της προόδου, v≈3%, όχι 50%)."""
    if total_steps <= 0:
        return 1.0
    v = (current_step / total_steps) ** 5
    return max(0.0, min(1.0, v))

def build_chord_context_ids(root_y, quality_y, pad_mask, reveal_fraction,
                             root_qual_to_flat_table, mask_chord_id):
    """ΝΕΑ ΕΚΔΟΣΗ - αντιγράφει ΑΚΡΙΒΩΣ τη λογική του πραγματικού κώδικα του paper
    (full_to_partial_masking στο train_utils.py του repo τους): ΤΥΧΑΙΑ ΜΕΤΑΘΕΣΗ
    (permutation) των θέσεων, ΑΚΡΙΒΗΣ αριθμός αποκαλυμμένων = min(floor(v*L), L-1).
    ΔΙΑΦΟΡΑ από την ΠΡΟΗΓΟΥΜΕΝΗ μας υλοποίηση (per-position Bernoulli): αυτή η
    προσέγγιση εγγυάται το L-1 cap ΑΠΟ ΤΗΝ ΚΑΤΑΣΚΕΥΗ της, χωρίς να χρειάζεται
    ξεχωριστό "διόρθωσε αν όλα αποκαλύφθηκαν" πέρασμα.
    ΣΗΜΕΙΩΣΗ: το ΔΙΚΟ ΤΟΥΣ code μοιράζεται ΜΙΑ μετάθεση σε ΟΛΟ το batch (έχουν
    ΣΤΑΘΕΡΟ μήκος grid) - εμείς έχουμε ΔΙΑΦΟΡΕΤΙΚΟ έγκυρο μήκος ανά τραγούδι
    (λόγω padding), άρα κάνουμε την ΙΔΙΑ λογική ΑΝΑ ΓΡΑΜΜΗ (ανά τραγούδι) αντί
    για ολόκληρο το batch μαζί - ΙΔΙΑ ουσία, προσαρμοσμένη στα δικά μας δεδομένα.
    Loss ΜΟΝΟ στις masked θέσεις (βλ. build_loss_targets παρακάτω)."""
    device = root_y.device
    B, T = root_y.shape
    safe_root_y = root_y.clamp(min=0, max=root_qual_to_flat_table.shape[0] - 1)
    safe_qual_y = quality_y.clamp(min=0, max=root_qual_to_flat_table.shape[1] - 1)
    true_flat_ids = root_qual_to_flat_table[safe_root_y, safe_qual_y]  # [B,T]

    should_reveal = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b in range(B):
        valid_positions = torch.nonzero(~pad_mask[b], as_tuple=True)[0]
        L_b = valid_positions.numel()
        if L_b == 0:
            continue
        perm = valid_positions[torch.randperm(L_b, device=device)]
        n_visible = min(int(L_b * reveal_fraction), L_b - 1)  # ΑΚΡΙΒΩΣ όπως Εξ.1 του paper
        if n_visible > 0:
            should_reveal[b, perm[:n_visible]] = True

    mask_tensor = torch.full_like(true_flat_ids, mask_chord_id)
    chord_context_ids = torch.where(should_reveal, true_flat_ids, mask_tensor)
    return chord_context_ids, should_reveal




# ABLATION: Κανόνας 6, ΞΕΧΩΡΙΣΤΟΣ στόχος function-transition ανά mode (major vs
# minor) αντί για ΕΝΑΝ κοινό/μικτό στόχο. ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ εξ ορισμού μετά από
# retrain όπου η ακρίβεια ήταν χαμηλότερη σε σχέση με το προηγούμενο, γνωστό-καλό
# checkpoint (68.13% joint vs 66.71% val acc σε αυτό το run) - ΣΗΜΕΙΩΣΗ: αυτό το
# retrain είχε ΚΑΙ τη διόρθωση βαθμονόμησης του Κανόνα 3 μαζί, άρα η αιτία δεν
# είναι πλήρως απομονωμένη· ο έλεγχος κώδικα δεν βρήκε bug στο mode-split
# μηχανισμό (ίδιο, ήδη επιβεβαιωμένο indexing pattern με το root_to_func_by_mode).
# Άλλαξε σε True για να ξανα-δοκιμάσεις με ΟΛΑ τα άλλα σταθερά.
ENABLE_RULE6_MODE_SPLIT = False

# --- 1Α. ΦΟΡΤΩΣΗ FINAL CORPUS & LOOKUP ---
CORPUS_PATH = os.path.join(DATA_DIR, 'final_corpus.pkl')
print("⏳ Φόρτωση του Final Corpus...")
with open(CORPUS_PATH, 'rb') as f:
    raw_songs = pickle.load(f)
print(f"✅ Φορτώθηκαν {len(raw_songs)} τραγούδια.")

LOOKUP_PATH = os.path.join(DATA_DIR, 'symbolic_lookup.pkl')
with open(LOOKUP_PATH, 'rb') as f:
    lookup = pickle.load(f)
chord_names = lookup['chord_names']
note_features = lookup['note_features']

# =============================================================================
# ΕΝΣΩΜΑΤΩΣΗ chord-melody-dataset (shiehn) - PART A
# =============================================================================
# Ανακατασκευάζουμε raw strings ('Pitch_C5', 'Am7' κ.λπ.) από τα ήδη
# υπολογισμένα δεδομένα του extraction pipeline, ώστε να περάσουν από ΤΟΝ
# ΙΔΙΟ κώδικα ανίχνευσης key/μετάθεσης που ήδη τρέχει για τα υπόλοιπα 20.122
# τραγούδια - καμία ξεχωριστή διαδρομή κώδικα. Οι ανακατασκευασμένες
# συμβολοσειρές έχουν δοκιμαστεί σε round-trip (μέσω των ΑΚΡΙΒΩΝ
# parse_root_quality_raw/pitch_name_to_semitone) - ταιριάζουν 100%.
import json

INCLUDE_CHORD_MELODY_DATASET = True   # <-- toggle εδώ
EXTRACTED_JSON_PATH = os.path.join(DATA_DIR1, 'extracted_songs.json')  # <-- άλλαξε στο δικό σου path

if INCLUDE_CHORD_MELODY_DATASET:
    print("⏳ Ενσωμάτωση chord-melody-dataset...")

    with open(EXTRACTED_JSON_PATH, 'r', encoding='utf-8') as f:
        _extracted = json.load(f)
    _new_songs = _extracted['songs']

    _PC_TO_SHARP_NAME_MERGE = ['C', 'C_sharp_', 'D', 'D_sharp_', 'E', 'F', 'F_sharp_',
                                'G', 'G_sharp_', 'A', 'A_sharp_', 'B']

    def _pitch_feature_to_pitch_string(pitch_feature):
        pc, octave = pitch_feature % 12, pitch_feature // 12
        return f"Pitch_{_PC_TO_SHARP_NAME_MERGE[pc]}{octave}"

    def _chord_to_raw_name(chord_root_pc, chord_quality):
        if chord_quality == 'NC' or chord_root_pc == 'UNKNOWN_ROOT':
            return 'N.C.'
        return f"{_PC_TO_SHARP_NAME_MERGE[chord_root_pc]}{chord_quality}"

    _synthetic_note_iris = []  # για το verification βήμα (PART B) παρακάτω
    _synthetic_iri_to_is_major = {}  # ΝΕΟ: γνωστό, ήδη επικυρωμένο mode ανά νότα -
                                       # επιτρέπει στο βήμα ανίχνευσης key να ΠΑΡΑΚΑΜΨΕΙ
                                       # την επανα-ανίχνευση για αυτά τα τραγούδια
                                       # (βλ. σχόλιο στο βήμα ανίχνευσης key παρακάτω)
    _added_songs, _added_notes = 0, 0

    for _song_idx, _song in enumerate(_new_songs):
        _title_safe = _song['title'].replace(' ', '_').replace('/', '_')
        _melody_iris, _chord_iris = [], []

        for _note_idx, _note in enumerate(_song['notes']):
            _note_iri = f"LOCAL#{_song_idx}_{_title_safe}_note_{_note_idx}"
            _chord_iri = f"LOCAL#{_song_idx}_{_title_safe}_chord_{_note_idx}"

            note_features[_note_iri] = {
                'pitch': _pitch_feature_to_pitch_string(_note['pitch_feature']),
                'duration': _note['duration_label'],
                'beat_type': _note['beat_type'],
                'position': _note['position'],
                'interval': _note['interval'],
                'beat_strength': _note['beat_strength'],
            }
            chord_names[_chord_iri] = _chord_to_raw_name(_note['chord_root_pc'], _note['chord_quality'])

            _melody_iris.append(_note_iri)
            _chord_iris.append(_chord_iri)
            _synthetic_note_iris.append(_note_iri)
            _synthetic_iri_to_is_major[_note_iri] = _song['is_major']
            _added_notes += 1

        if _melody_iris:
            raw_songs.append((_melody_iris, _chord_iris))
            _added_songs += 1

    print(f"✅ Προστέθηκαν {_added_songs} τραγούδια ({_added_notes} νότες) από το chord-melody-dataset.")
    print(f"   Σύνολο raw_songs μετά την ενσωμάτωση: {len(raw_songs)}")
else:
    _synthetic_note_iris = []
    _synthetic_iri_to_is_major = {}
    print("ℹ️ Το chord-melody-dataset ΔΕΝ ενσωματώνεται (INCLUDE_CHORD_MELODY_DATASET=False).")

# =============================================================================
# ΕΝΣΩΜΑΤΩΣΗ HookTheory (chrisdonahue/sheetsage export) - PART A
# =============================================================================
# ΤΟ ΙΔΙΟ μοτίβο με το chord-melody-dataset merge παραπάνω: ανακατασκευάζουμε
# raw strings ('Pitch_B0', 'Gmaj' κ.λπ.) από τα ήδη υπολογισμένα δεδομένα του
# extract_hooktheory.py pipeline, ώστε να περάσουν από ΤΟΝ ΙΔΙΟ κώδικα
# ανίχνευσης key/μετάθεση που τρέχει για όλο το υπόλοιπο corpus. Η ΜΟΝΗ
# διαφορά: το duration/beat_type/position/interval είναι ΗΔΗ στη σωστή μορφή
# (reused conventions), και τα chords είναι ΗΔΗ ευθυγραμμισμένα ανά νότα (όχι
# ξεχωριστή λίστα geitonikών timestamps) - το extract_hooktheory.py το έκανε
# ήδη αυτό.
INCLUDE_HOOKTHEORY_DATASET = False   # <-- toggle εδώ
HOOKTHEORY_JSON_PATH = os.path.join(DATA_DIR1, 'extracted_hooktheory.json')  # <-- άλλαξε στο δικό σου path

if INCLUDE_HOOKTHEORY_DATASET:
    print("⏳ Ενσωμάτωση HookTheory...")
    with open(HOOKTHEORY_JSON_PATH, 'r', encoding='utf-8') as f:
        _hk_data = json.load(f)
    _hk_songs = _hk_data['songs']

    # --- Φιλτράρισμα μήκους + τυχαία υποδειγματοληψία ---
    # Στόχος: (1) αφαίρεση ΠΡΑΓΜΑΤΙΚΩΝ ακραίων τιμών μήκους - πολύ κοντά
    # τραγούδια προσθέτουν ελάχιστο πλαίσιο, πολύ μακριά καταστρέφονται από το
    # max_seq_len truncation (π.χ. 762 νότες -> 392, ή 16623 -> 392, σχεδόν
    # τα πάντα χάνονται). (2) μείωση του ΣΥΝΟΛΙΚΟΥ μεριδίου του HookTheory στο
    # corpus (ήταν ~50%, πιθανή αιτία της κατάρρευσης ακρίβειας) - ΤΥΧΑΙΑ
    # υποδειγματοληψία (όχι βάσει άλλου κριτηρίου), ώστε να ελέγξουμε
    # ΕΙΔΙΚΑ την υπόθεση όγκου, χωρίς να μπερδεύεται με άλλες μεταβλητές.
    HOOKTHEORY_MIN_NOTES = 8      # αποκλείει εξαιρετικά μικρά αποσπάσματα
    HOOKTHEORY_MAX_NOTES = 327    # το ΔΙΚΟ ΤΟΥ HookTheory 99ό εκατοστημόριο -
                                   # όχι σύγκριση με την υπόλοιπη corpus
    HOOKTHEORY_DOWNSAMPLE_TARGET = 7000  # από 20992 -> ~25% της νέας corpus
                                          # (πριν ήταν ~50%)
    HOOKTHEORY_SAMPLE_SEED = 42

    _hk_before_filter = len(_hk_songs)
    _hk_songs = [s for s in _hk_songs if HOOKTHEORY_MIN_NOTES <= s['n_notes'] <= HOOKTHEORY_MAX_NOTES]
    _hk_after_length_filter = len(_hk_songs)

    _hk_rng = random.Random(HOOKTHEORY_SAMPLE_SEED)
    if len(_hk_songs) > HOOKTHEORY_DOWNSAMPLE_TARGET:
        _hk_songs = _hk_rng.sample(_hk_songs, HOOKTHEORY_DOWNSAMPLE_TARGET)

    print(f"   HookTheory: {_hk_before_filter} -> {_hk_after_length_filter} μετά το φίλτρο μήκους "
          f"({HOOKTHEORY_MIN_NOTES}-{HOOKTHEORY_MAX_NOTES} νότες) -> {len(_hk_songs)} μετά την "
          f"τυχαία υποδειγματοληψία (στόχος={HOOKTHEORY_DOWNSAMPLE_TARGET}, seed={HOOKTHEORY_SAMPLE_SEED})")

    def _hk_pitch_string(letter, alter, octave):
        if alter == 0:
            suffix = ''
        elif alter == 1:
            suffix = '_sharp_'
        elif alter == -1:
            suffix = '_flat_'
        else:
            return None  # έξω από το υποστηριζόμενο εύρος - παραλείπεται με ασφάλεια
        return f"Pitch_{letter}{suffix}{octave}"

    def _hk_chord_to_raw_name(chord_root_pc, chord_quality):
        if not isinstance(chord_root_pc, int):
            return 'N.C.'
        return f"{_PC_TO_SHARP_NAME_MERGE[chord_root_pc]}{chord_quality}"

    _PC_TO_SHARP_NAME_MERGE = ['C', 'C_sharp_', 'D', 'D_sharp_', 'E', 'F', 'F_sharp_',
                                'G', 'G_sharp_', 'A', 'A_sharp_', 'B']

    _hk_synthetic_note_iris = []
    _hk_added_songs, _hk_added_notes, _hk_skipped_unspellable = 0, 0, 0

    for _hk_song_idx, _hk_song in enumerate(_hk_songs):
        _hk_title_safe = _hk_song['title'].replace(' ', '_').replace('/', '_')
        _hk_melody_iris, _hk_chord_iris = [], []

        for _hk_note_idx, _hk_note in enumerate(_hk_song['notes']):
            _hk_pitch_str = _hk_pitch_string(_hk_note['letter'], _hk_note['alter'], _hk_note['octave'])
            if _hk_pitch_str is None:
                _hk_skipped_unspellable += 1
                continue

            _hk_note_iri = f"HK#{_hk_song_idx}_{_hk_title_safe}_note_{_hk_note_idx}"
            _hk_chord_iri = f"HK#{_hk_song_idx}_{_hk_title_safe}_chord_{_hk_note_idx}"

            note_features[_hk_note_iri] = {
                'pitch': _hk_pitch_str,
                'duration': _hk_note['duration_label'],
                'beat_type': _hk_note['beat_type'],
                'position': _hk_note['position'],
                'interval': _hk_note['interval'],
                'beat_strength': _hk_note['beat_strength'],
            }
            chord_names[_hk_chord_iri] = _hk_chord_to_raw_name(_hk_note['chord_root_pc'], _hk_note['chord_quality'])

            _hk_melody_iris.append(_hk_note_iri)
            _hk_chord_iris.append(_hk_chord_iri)
            _hk_synthetic_note_iris.append(_hk_note_iri)
            _hk_added_notes += 1

        if _hk_melody_iris:
            raw_songs.append((_hk_melody_iris, _hk_chord_iris))
            _hk_added_songs += 1

    print(f"✅ Προστέθηκαν {_hk_added_songs} τραγούδια ({_hk_added_notes} νότες) από το HookTheory "
          f"({_hk_skipped_unspellable} νότες παραλείφθηκαν - μη αναπαραστάσιμη γραφή).")
    print(f"   Σύνολο raw_songs μετά την ενσωμάτωση: {len(raw_songs)}")
else:
    _hk_synthetic_note_iris = []
    print("ℹ️ Το HookTheory ΔΕΝ ενσωματώνεται (INCLUDE_HOOKTHEORY_DATASET=False).")
    
INCLUDE_POP909_DATASET = True
POP909_JSON_PATH = os.path.join(DATA_DIR1, 'extracted_pop909.json')
if INCLUDE_POP909_DATASET:
    print("⏳ Ενσωμάτωση POP909...")
    with open(POP909_JSON_PATH, 'r', encoding='utf-8') as f:
        _pop_data = json.load(f)
    _pop_songs = _pop_data['songs']

    # <<< ΝΕΟ: ΔΙΚΕΣ ΤΟΥ, ΑΝΕΞΑΡΤΗΤΕΣ βοηθητικές συναρτήσεις - ΔΕΝ εξαρτώνται
    # πλέον από το να έχει τρέξει το HookTheory block πρώτα (που τις όριζε
    # μέσα στο ΔΙΚΟ ΤΟΥ if-block - αν INCLUDE_HOOKTHEORY_DATASET=False, ΠΟΤΕ
    # δεν οριζόντουσαν, εξ ου και το NameError).
    def _pop_pitch_string(letter, alter, octave):
        if alter == 0:
            suffix = ''
        elif alter == 1:
            suffix = '_sharp_'
        elif alter == -1:
            suffix = '_flat_'
        else:
            return None
        return f"Pitch_{letter}{suffix}{octave}"

    _POP_PC_TO_SHARP_NAME = ['C', 'C_sharp_', 'D', 'D_sharp_', 'E', 'F', 'F_sharp_',
                              'G', 'G_sharp_', 'A', 'A_sharp_', 'B']
    def _pop_chord_to_raw_name(chord_root_pc, chord_quality):
        if chord_quality == 'NC' or chord_root_pc == 'UNKNOWN_ROOT':
            return 'N.C.'
        return f"{_POP_PC_TO_SHARP_NAME[chord_root_pc]}{chord_quality}"
    # >>> ΤΕΛΟΣ ΝΕΟΥ

    _pop_added_songs, _pop_added_notes = 0, 0
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
            _pop_added_notes += 1
        if _pop_melody_iris:
            raw_songs.append((_pop_melody_iris, _pop_chord_iris))
            _pop_added_songs += 1
    print(f"✅ Προστέθηκαν {_pop_added_songs} τραγούδια ({_pop_added_notes} νότες) από το POP909.")
    print(f"   Σύνολο raw_songs μετά την ενσωμάτωση: {len(raw_songs)}")
    
else: print("NO 909")
# =============================================================================
# ΠΕΙΡΑΜΑ: εκπαίδευση ΑΠΟΚΛΕΙΣΤΙΚΑ στα "Qm..." τραγούδια (μεγάλη, απλή-ποιότητας
# πηγή, ~19.685 τραγούδια) - αποκλείει ΚΑΙ τα 437 real-name jazz standards ΚΑΙ
# οποιαδήποτε τραγούδια από το merge (LOCAL#), ώστε η σύγκριση KK vs TKP
# key-profile να είναι καθαρή, χωρίς την πολυπλοκότητα του merge να μπερδεύει
# το αποτέλεσμα.
TRAIN_ON_QM_ONLY_SUBSET = False  # <-- toggle εδώ

if TRAIN_ON_QM_ONLY_SUBSET:
    import re as _re_filter
    _CID_PATTERN_FILTER = _re_filter.compile(r'^Qm[1-9A-HJ-NP-Za-km-z]{44}')

    def _extract_identifier_filter(iri):
        frag = iri.split('#')[-1]
        parts = frag.rsplit('_', 2)
        return parts[0] if len(parts) == 3 else frag

    _before_count = len(raw_songs)
    raw_songs = [
        (melody_iris, chord_iris) for melody_iris, chord_iris in raw_songs
        if melody_iris and _CID_PATTERN_FILTER.match(_extract_identifier_filter(melody_iris[0]))
    ]
    print(f"⏳ Φιλτράρισμα σε ΜΟΝΟ 'Qm...' τραγούδια: {_before_count} -> {len(raw_songs)} "
          f"(αφαιρέθηκαν {_before_count - len(raw_songs)} - jazz standards + merge, αν υπήρχαν)")

# ΣΗΜΕΙΩΣΗ: τα note_core_pitches/chord_pitches (βασισμένα σε CorePitch ontology
# names που έχουμε επιβεβαιώσει μόνο ένα φορά) ΔΕΝ χρησιμοποιούνται πλέον· η
# συμβατότητα υπολογίζεται τώρα από μουσική θεωρία (root+quality -> intervals),
# βασισμένη στο ήδη επικυρωμένο pitch name parser (μόνο 142/2.2M αποτυχίες).

# --- 1Β. ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ: pitch names, root/quality parsing, μετάθεση, εύρεση key ---
PITCH_LETTER_SEMITONE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
PITCH_NAME_RE = re.compile(r'^Pitch_([A-G])(_sharp_|_flat_)?(-?\d+)$')
PC_TO_SHARP_NAME = ['C', 'C_sharp_', 'D', 'D_sharp_', 'E', 'F', 'F_sharp_', 'G', 'G_sharp_', 'A', 'A_sharp_', 'B']

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
    """Επιστρέφει (root_pc, quality) ΠΡΙΝ από οποιαδήποτε μετάθεση/indexing.
    root_pc: int 0-11, ή 'UNKNOWN_ROOT' (π.χ. Roman numerals, άγνωστα strings)."""
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

# Krumhansl-Schmuckler key-finding: profile-correlation, standard textbook algorithm.
# ΠΕΙΡΑΜΑ: επιλογή προφίλ - επιβεβαιωμένο σε πραγματικό ground truth (12-key
# dataset, 5.198 δοκιμές) ότι το Temperley-Kostka-Payne έχει σαφώς υψηλότερη
# ακρίβεια τονικής (64.5%) από το τρέχον Krumhansl-Kessler (50.8%). Η στάθμιση
# κατά διάρκεια στο ιστόγραμμα ΔΕΝ αλλάζει εδώ σκόπιμα - μόνο ΕΝΑ μεταβλητό
# ανά πείραμα, ώστε η σύγκριση KK vs TKP να είναι καθαρή.
KEY_PROFILE_CHOICE = 'KK'  # <-- άλλαξε σε 'TKP' για το δεύτερο run

KS_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

TKP_MAJOR_PROFILE = [0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400]
TKP_MINOR_PROFILE = [0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330]

ACTIVE_MAJOR_PROFILE, ACTIVE_MINOR_PROFILE = (
    (TKP_MAJOR_PROFILE, TKP_MINOR_PROFILE) if KEY_PROFILE_CHOICE == 'TKP'
    else (KS_MAJOR_PROFILE, KS_MINOR_PROFILE)
)
print(f"   ℹ️ Ενεργό προφίλ ανίχνευσης key: {KEY_PROFILE_CHOICE}")

def _pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((x - mb) ** 2 for x in b) ** 0.5
    return 0.0 if da == 0 or db == 0 else num / (da * db)

def detect_key(pitch_classes):
    """pitch_classes: λίστα από ints 0-11. Επιστρέφει (tonic_pc, mode). Χρησιμοποιεί
    το ΕΝΕΡΓΟ προφίλ (KEY_PROFILE_CHOICE) - ΙΔΙΟΣ αλγόριθμος/στάθμιση με πριν,
    ΜΟΝΟ το προφίλ αλλάζει, ώστε το πείραμα να συγκρίνει ΕΝΑ πράγμα τη φορά."""
    hist = [0.0] * 12
    for pc in pitch_classes:
        hist[pc] += 1.0
    if sum(hist) == 0:
        return 0, 'major'
    best = (-2.0, 0, 'major')
    for tonic in range(12):
        rotated = [hist[(tonic + k) % 12] for k in range(12)]
        for mode, profile in (('major', ACTIVE_MAJOR_PROFILE), ('minor', ACTIVE_MINOR_PROFILE)):
            c = _pearson(rotated, profile)
            if c > best[0]:
                best = (c, tonic, mode)
    return best[1], best[2]

# Μουσική θεωρία: quality -> semitone intervals από τη ρίζα. 'NC'/'OTHER' λείπουν
# σκόπιμα -> άγνωστο περιεχόμενο -> πάντα "συμβατό" (δεν τιμωρούμε κάτι που δεν ξέρουμε).
QUALITY_INTERVALS = {
    'maj': [0, 4, 7], 'm': [0, 3, 7], '7': [0, 4, 7, 10], 'm7': [0, 3, 7, 10],
    'maj7': [0, 4, 7, 11], 'dim': [0, 3, 6], '6': [0, 4, 7, 9], '9': [0, 4, 7, 10, 2],
    'ø7': [0, 3, 6, 10], 'sus': [0, 5, 7], 'm6': [0, 3, 7, 9], 'o7': [0, 3, 6, 9],
    'aug': [0, 4, 8], '7aug': [0, 4, 8, 10], 'm9': [0, 3, 7, 10, 2], 'power': [0, 7],
    '13': [0, 4, 7, 10, 2, 9], 'sus2': [0, 2, 7], 'M9': [0, 4, 7, 11, 2],
    'm11': [0, 3, 7, 10, 2, 5], 'mM7': [0, 3, 7, 11], '11': [0, 4, 7, 10, 2, 5],
}

def chord_pitch_classes(root_pc, quality):
    if not isinstance(root_pc, int) or quality not in QUALITY_INTERVALS:
        return None  # άγνωστο περιεχόμενο -> πάντα συμβατό
    return {(root_pc + iv) % 12 for iv in QUALITY_INTERVALS[quality]}

# --- 1Γ. ΑΝΙΧΝΕΥΣΗ KEY & ΜΕΤΑΘΕΣΗ ΑΝΑ ΤΡΑΓΟΥΔΙ ---
# ΚΑΘΕ τραγούδι μεταφέρεται στο ίδιο κανονικό key (C major / A minor - ίδια σύμβαση
# με το paper) πριν την εκπαίδευση. Αυτό ΔΕΝ περιορίζει ποια τραγούδια μπορεί να
# αρμονικοποιήσει το μοντέλο - η μετάθεση είναι πλήρως αντιστρέψιμη: σε πραγματική
# χρήση, ανιχνεύουμε το key του input, μεταθέτουμε προς τα μέσα, τρέχουμε το
# μοντέλο, μεταθέτουμε τις προβλεπόμενες ρίζες προς τα έξω.
print("⏳ Ανίχνευση key & μετάθεση ανά τραγούδι (Krumhansl-Schmuckler)...")
note_pitch_class_transposed = {}
transposed_pitch_name = {}
chord_root_pc_transposed = {}
chord_quality_raw = {}
chord_is_major = {}  # ΚΑΝΟΝΑΣ 6: mode ανά χορδή (True=major, False=minor)
note_is_leading_tone = {}   # ΚΑΝΟΝΑΣ 5: True αν η νότα ΕΙΝΑΙ η leading tone (mode-aware)
note_expected_tonic_pc = {}  # ΚΑΝΟΝΑΣ 5: ποια pitch class μετράει σαν "τονική" γι' αυτή τη νότα (0=major, 9=minor)
unparsed_pitch_count = 0
major_count, minor_count = 0, 0

for melody_iris, chord_iris in raw_songs:
    # ΔΙΟΡΘΩΣΗ μετά το verification βήμα (Part B) του chord-melody-dataset merge:
    # η ΕΠΑΝΑ-ανίχνευση key πάνω σε ήδη-κανονικά τραγούδια διαφώνησε με τη δική
    # μας αρχική ανίχνευση στο 20.5% των νοτών - πιθανή αιτία, η δική μας
    # στάθμιση κατά διάρκεια στο ιστόγραμμα pitch-class, ΕΝΩ το detect_key() εδώ
    # μετράει απλές εμφανίσεις (χωρίς στάθμιση). Αντί να βασιστούμε σε συμφωνία
    # μεταξύ δύο ΔΙΑΦΟΡΕΤΙΚΩΝ υλοποιήσεων, χρησιμοποιούμε ΑΠΕΥΘΕΙΑΣ το ήδη
    # γνωστό, ήδη επικυρωμένο (μέσω 12-key cross-check στο δικό μας extraction
    # pipeline) mode για αυτά τα τραγούδια - shift=0 εγγυημένα, καμία εικασία.
    if melody_iris and melody_iris[0] in _synthetic_iri_to_is_major:
        mode = 'major' if _synthetic_iri_to_is_major[melody_iris[0]] else 'minor'
        shift = 0
    else:
        raw_pcs = []
        for n in melody_iris:
            st = pitch_name_to_semitone(note_features.get(n, {}).get('pitch'))
            if st is not None:
                raw_pcs.append(st % 12)
        tonic, mode = detect_key(raw_pcs)
        shift = ((0 - tonic) % 12) if mode == 'major' else ((9 - tonic) % 12)
    if mode == 'major':
        major_count += 1
    else:
        minor_count += 1

    # ΚΑΝΟΝΑΣ 5: η "leading tone" εξαρτάται από το mode. Σε major, η 7η βαθμίδα
    # (11, μισό τόνο κάτω από την τονική) είναι πραγματική leading tone. Σε minor
    # (φυσική ελάσσονα, ίδια συλλογή pitch classes με τη σχετική major), η 7η
    # βαθμίδα (7, ΕΝΑΝ τόνο κάτω) είναι "subtonic" - ΔΕΝ έχει την ίδια έλξη. Η
    # πραγματική (ανεβασμένη) leading tone σε minor είναι στο 8 (μισό τόνο κάτω
    # από την τονική 9).
    if mode == 'major':
        leading_tone_pc, tonic_pc = 11, 0
    else:
        leading_tone_pc, tonic_pc = 8, 9

    for n in melody_iris:
        if n in note_pitch_class_transposed:
            continue
        pname = note_features.get(n, {}).get('pitch')
        st = pitch_name_to_semitone(pname)
        if st is None:
            note_pitch_class_transposed[n] = None
            transposed_pitch_name[n] = pname if pname else '__UNKNOWN__'
            unparsed_pitch_count += 1
        else:
            note_pitch_class_transposed[n] = (st + shift) % 12
            transposed_pitch_name[n] = transpose_pitch_name(pname, shift)
        note_is_leading_tone[n] = (note_pitch_class_transposed[n] == leading_tone_pc)
        note_expected_tonic_pc[n] = tonic_pc

    for c in chord_iris:
        if c in chord_root_pc_transposed:
            continue
        name = simplify_chord_name(chord_names.get(c, ''))
        root_pc, quality = parse_root_quality_raw(name)
        chord_quality_raw[c] = quality
        chord_is_major[c] = (mode == 'major')  # ΚΑΝΟΝΑΣ 6: χρειάζεται mode-aware function
        if isinstance(root_pc, int):
            chord_root_pc_transposed[c] = (root_pc + shift) % 12
        else:
            chord_root_pc_transposed[c] = 'UNKNOWN_ROOT'

print(f"   Ανιχνεύθηκαν {major_count} τραγούδια σε major, {minor_count} σε minor.")
if unparsed_pitch_count > 0:
    print(f"   ⚠️ {unparsed_pitch_count} pitch names δεν αναγνωρίστηκαν - ουδέτερο (πάντα συμβατό) βάρος.")

# ΚΑΝΟΝΑΣ 2 χρειάζεται το ΠΛΗΡΕΣ semitone (ΜΕ octave, όχι μόνο pitch class mod 12)
# για να ξεχωρίσει "βήμα" από "άλμα οκτάβας". Παίρνουμε το από το ΗΔΗ μεταθεμένο
# όνομα ώστε να μείνει συνεπές με όλα τα υπόλοιπα.
note_semitone = {iri: pitch_name_to_semitone(transposed_pitch_name.get(iri)) for iri in note_pitch_class_transposed}

# --- 1Δ. ΛΕΞΙΚΑ (από τα ΜΕΤΑΤΕΘΕΙΜΕΝΑ δεδομένα) ---
all_notes_iris = sorted(note_pitch_class_transposed.keys())
note_class_to_idx = {}
note_to_idx = {}
for iri in all_notes_iris:
    key = note_pitch_class_transposed[iri]
    key = key if key is not None else '__UNKNOWN__'
    if key not in note_class_to_idx:
        note_class_to_idx[key] = len(note_class_to_idx)
    note_to_idx[iri] = note_class_to_idx[key]
NUM_NOTE_CLASSES = len(note_class_to_idx)

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

print(f"   Εμπλουτισμένα features: {NUM_PITCH_OCT_CLASSES} pitch(+octave, μεταθεμένο), {NUM_DURATION_CLASSES} duration, "
      f"{NUM_BEAT_TYPE_CLASSES} beat_type, {NUM_POSITION_CLASSES} position, {NUM_INTERVAL_CLASSES} interval")
print(f"✅ Λεξικά έτοιμα: {len(all_notes_iris)} raw notes -> {NUM_NOTE_CLASSES} pitch-class ομάδες (κανονικοποιημένο key)")

all_chords_iris = sorted(chord_root_pc_transposed.keys())
root_to_idx = {}
quality_to_idx = {}
chord_root_idx = {}
chord_quality_idx = {}
for c in all_chords_iris:
    root_key = chord_root_pc_transposed[c]
    qual_key = chord_quality_raw[c]
    if root_key not in root_to_idx:
        root_to_idx[root_key] = len(root_to_idx)
    if qual_key not in quality_to_idx:
        quality_to_idx[qual_key] = len(quality_to_idx)
    chord_root_idx[c] = root_to_idx[root_key]
    chord_quality_idx[c] = quality_to_idx[qual_key]
NUM_ROOT_CLASSES = len(root_to_idx)
NUM_QUALITY_CLASSES = len(quality_to_idx)
print(f"   Auxiliary heads: {NUM_ROOT_CLASSES} ρίζες, {NUM_QUALITY_CLASSES} qualities (μετά το κανονικό key)")

# FLAT chord vocabulary: ΤΩΡΑ ορίζεται σαν συνδυασμός (transposed root, quality) -
# ΟΧΙ σαν raw string, ώστε να παραμένει συνεπές με το κανονικό key. Πολύ μικρότερο
# λεξιλόγιο σαν φυσικό επακόλουθο (<= NUM_ROOT_CLASSES x NUM_QUALITY_CLASSES).
chord_to_idx = {}
flat_key_to_idx = {}
for c in all_chords_iris:
    combo = (chord_root_idx[c], chord_quality_idx[c])
    if combo not in flat_key_to_idx:
        flat_key_to_idx[combo] = len(flat_key_to_idx)
    chord_to_idx[c] = flat_key_to_idx[combo]
NUM_CHORD_CLASSES = len(flat_key_to_idx)
print(f"   {len(all_chords_iris)} raw chords -> {NUM_CHORD_CLASSES} flat κλάσεις (root x quality, κανονικοποιημένο key)")

chord_root_mapping = torch.zeros(NUM_CHORD_CLASSES, dtype=torch.long)
chord_qual_mapping = torch.zeros(NUM_CHORD_CLASSES, dtype=torch.long)
for (r_idx, q_idx), flat_idx in flat_key_to_idx.items():
    chord_root_mapping[flat_idx] = r_idx
    chord_qual_mapping[flat_idx] = q_idx
chord_root_mapping = chord_root_mapping.to(device)
chord_qual_mapping = chord_qual_mapping.to(device)

# --- 1Ε. ΠΙΝΑΚΑΣ ΣΥΜΒΑΤΟΤΗΤΑΣ (από μουσική θεωρία, όχι από ontology CorePitch names) ---
idx_to_note_pc = {idx: pc for pc, idx in note_class_to_idx.items()}
idx_to_root_pc = {idx: pc for pc, idx in root_to_idx.items()}
idx_to_quality = {idx: q for q, idx in quality_to_idx.items()}

print("⏳ Κατασκευή πίνακα συμβατότητας από μουσική θεωρία (root+quality -> intervals)...")
compatibility_matrix = np.zeros((NUM_NOTE_CLASSES, NUM_CHORD_CLASSES), dtype=np.float32)
for note_idx in range(NUM_NOTE_CLASSES):
    note_pc = idx_to_note_pc[note_idx]
    for (r_idx, q_idx), flat_idx in flat_key_to_idx.items():
        root_pc = idx_to_root_pc[r_idx]
        quality = idx_to_quality[q_idx]
        pitches = chord_pitch_classes(root_pc, quality)
        if pitches is None or note_pc == '__UNKNOWN__':
            compatibility_matrix[note_idx, flat_idx] = 1.0  # άγνωστο περιεχόμενο -> πάντα συμβατό
        elif note_pc in pitches:
            compatibility_matrix[note_idx, flat_idx] = 1.0

# --- 1ΣΤ. ΜΕΤΡΗΣΗ ΠΑΡΑΒΙΑΣΗΣ (νέος, θεωρητικά-θεμελιωμένος έλεγχος) + ΚΑΝΟΝΑΣ 1 ---
beat_type_totals = Counter()
beat_type_violations = Counter()
beat_type_same_totals = Counter()
beat_type_same_counts = Counter()
matched_pairs, total_pairs = 0, 0

for melody_iris, chord_iris in raw_songs:
    for t, (note_iri, chord_iri) in enumerate(zip(melody_iris, chord_iris)):
        total_pairs += 1
        n_idx = note_to_idx[note_iri]
        c_idx = chord_to_idx[chord_iri]
        is_valid = compatibility_matrix[n_idx, c_idx] > 0.5
        if is_valid:
            matched_pairs += 1
        bt = note_features.get(note_iri, {}).get('beat_type') or '__UNKNOWN__'
        beat_type_totals[bt] += 1
        if not is_valid:
            beat_type_violations[bt] += 1
        beat_type_same_totals[bt] += 1
        if t > 0 and chord_iris[t] == chord_iris[t - 1]:
            beat_type_same_counts[bt] += 1

match_rate = (matched_pairs / total_pairs * 100) if total_pairs > 0 else 0
baseline_error = 100.0 - match_rate
PENALTY_WEIGHT = (1.0 / ((baseline_error / 100.0) + 0.01)) - 1
print(f"✅ Πίνακας συμβατότητας: {matched_pairs}/{total_pairs} ζεύγη ({match_rate:.1f}%) έγκυρα "
      f"-> baseline_error={baseline_error:.2f}% -> PENALTY_WEIGHT={PENALTY_WEIGHT:.2f}")

# ΔΙΟΡΘΩΣΗ: το beat_type_to_idx (από build_feature_vocab) είναι keyed ΑΝΑ NOTE IRI,
# ΟΧΙ ανά raw string τιμή ("Beat_Strong" κλπ) - η ΙΔΙΑ κατηγορία σφάλματος που ήδη
# βρήκαμε και διορθώσαμε στο harmonize_musicxml.py. Τα beat_type_scale_tensor,
# expected_same_tensor, rule3_strictness_tensor παρακάτω ΧΡΕΙΑΖΟΝΤΑΙ ένα raw-value
# -> index mapping, ΟΧΙ IRI -> index. Το φτιάχνουμε ΕΔΩ, ΜΙΑ φορά: για κάθε
# μοναδική raw τιμή beat_type, βρίσκουμε ΜΙΑ νότα με αυτή την τιμή και παίρνουμε
# το index ΤΗΣ (ΟΛΕΣ οι νότες με την ΙΔΙΑ raw τιμή έχουν το ΙΔΙΟ index, εξ
# ορισμού του build_feature_vocab, άρα αυτό είναι ακριβές, όχι προσέγγιση).
beat_type_value_to_idx = {}
for _iri in all_notes_iris:
    _raw_bt_val = note_features.get(_iri, {}).get('beat_type')
    _raw_bt = _raw_bt_val if _raw_bt_val is not None else "__UNKNOWN__"
    if _raw_bt not in beat_type_value_to_idx:
        beat_type_value_to_idx[_raw_bt] = beat_type_to_idx[_iri]
        if len(beat_type_value_to_idx) >= NUM_BEAT_TYPE_CLASSES:
            break

beat_type_rate = {bt: beat_type_violations.get(bt, 0) / total for bt, total in beat_type_totals.items()}
raw_scale = {bt: 1.0 / (rate + 0.05) for bt, rate in beat_type_rate.items()}
mean_scale = sum(raw_scale.values()) / len(raw_scale) if raw_scale else 1.0
beat_type_leniency = {bt: v / mean_scale for bt, v in raw_scale.items()}
print("   Κανόνας 1 (μετρικό βάρος): ποσοστό παραβίασης & leniency scale ανά beat_type:")
for bt in sorted(beat_type_totals, key=lambda k: -beat_type_totals[k]):
    print(f"      {bt:15} rate={beat_type_rate[bt]*100:5.1f}%  scale={beat_type_leniency[bt]:.2f}  (n={beat_type_totals[bt]})")

beat_type_scale_tensor = torch.ones(NUM_BEAT_TYPE_CLASSES + 1, dtype=torch.float32)
for bt_name, idx in beat_type_value_to_idx.items():
    beat_type_scale_tensor[idx] = beat_type_leniency.get(bt_name, 1.0)
beat_type_scale_tensor = beat_type_scale_tensor.to(device)
print(f"      [ΕΛΕΓΧΟΣ] πραγματικός beat_type_scale_tensor: {beat_type_scale_tensor.tolist()} "
      f"(ΠΡΕΠΕΙ να ΔΙΑΦΕΡΕΙ ανά θέση - αν είναι όλα ίδια, κάτι έσπασε)")

# --- 1Ζ. ΚΑΝΟΝΑΣ 3 (σταθερότητα αρμονικού ρυθμού) ---
beat_type_same_rate = {bt: beat_type_same_counts.get(bt, 0) / total
                        for bt, total in beat_type_same_totals.items()}
print("   Κανόνας 3 (σταθερότητα αρμονικού ρυθμού): P(ίδια χορδή με προηγούμενη) ανά beat_type:")
for bt in sorted(beat_type_same_totals, key=lambda k: -beat_type_same_totals[k]):
    print(f"      {bt:15} P(same)={beat_type_same_rate[bt]*100:5.1f}%  (n={beat_type_same_totals[bt]})")

overall_same = sum(beat_type_same_counts.values())
overall_total = sum(beat_type_same_totals.values())
overall_change_rate = 1.0 - (overall_same / overall_total if overall_total > 0 else 0.0)
RULE3_WEIGHT = (1.0 / (overall_change_rate + 0.01)) - 1
print(f"   Συνολικό ποσοστό αλλαγής χορδής (νότα->νότα): {overall_change_rate*100:.2f}% -> "
      f"RULE3_WEIGHT={RULE3_WEIGHT:.2f} (ίδια φόρμουλα με το PENALTY_WEIGHT={PENALTY_WEIGHT:.2f})")

expected_same_tensor = torch.full((NUM_BEAT_TYPE_CLASSES + 1,), 0.5, dtype=torch.float32)
for bt_name, idx in beat_type_value_to_idx.items():
    expected_same_tensor[idx] = beat_type_same_rate.get(bt_name, 0.5)
expected_same_tensor = expected_same_tensor.to(device)
print(f"      [ΕΛΕΓΧΟΣ] πραγματικός expected_same_tensor: {expected_same_tensor.tolist()} "
      f"(ΠΡΕΠΕΙ να ΔΙΑΦΕΡΕΙ ανά θέση - αν είναι όλα 0.5, κάτι έσπασε)")

# --- 1Ζ.2. ΕΝΙΣΧΥΣΗ Κανόνα 3: ανά-beat_type ΑΥΣΤΗΡΟΤΗΤΑ (ΟΧΙ μόνο στόχος) ---
# Μέχρι εδώ ο Κανόνας 3 έχει ΔΙΑΦΟΡΕΤΙΚΟ ΣΤΟΧΟ ανά beat_type (expected_same_tensor
# παραπάνω), αλλά το ΙΔΙΟ ΒΑΡΟΣ ποινής παντού - μια απόκλιση από το 97% στο Weak
# τιμωρείται ΤΟ ΙΔΙΟ όσο μια απόκλιση από το 63% στο Strong, παρόλο που η πρώτη
# είναι ΠΟΛΥ πιο "σίγουρη"/προβλέψιμη συμπεριφορά στα δεδομένα. Εδώ προσθέτουμε
# ΠΟΛΛΑΠΛΑΣΙΑΣΤΙΚΗ αυστηρότητα ανά beat_type, βασισμένη στη ΔΙΑΚΥΜΑΝΣΗ Bernoulli
# p*(1-p) του στόχου: όσο πιο κοντά στο 0/1 (σχεδόν ντετερμινιστικό, π.χ. Weak/
# Offbeat ~97%) τόσο ΜΙΚΡΟΤΕΡΗ η διακύμανση -> ΜΕΓΑΛΥΤΕΡΗ αυστηρότητα εκεί· όσο πιο
# κοντά στο 50% (πραγματικά αβέβαιο, π.χ. Strong ~63%) τόσο ΜΕΓΑΛΥΤΕΡΗ η διακύμανση
# -> ΜΙΚΡΟΤΕΡΗ αυστηρότητα εκεί. ΙΔΙΑ φόρμουλα/κανονικοποίηση (μέσος όρος=1.0) με
# το beat_type_leniency του Κανόνα 1 παραπάνω, ώστε το συνολικό effective βάρος να
# παραμένει συγκρίσιμο με πριν.
rule3_variance = {bt: p * (1.0 - p) for bt, p in beat_type_same_rate.items()}
rule3_raw_strictness = {bt: 1.0 / (v + 0.05) for bt, v in rule3_variance.items()}
rule3_mean_strictness = sum(rule3_raw_strictness.values()) / len(rule3_raw_strictness) if rule3_raw_strictness else 1.0
rule3_strictness = {bt: v / rule3_mean_strictness for bt, v in rule3_raw_strictness.items()}
print("   Κανόνας 3, ενισχυμένη αυστηρότητα ανά beat_type (1.0=ουδέτερο, βάσει διακύμανσης στόχου):")
for bt in sorted(rule3_strictness, key=lambda k: -rule3_strictness[k]):
    print(f"      {bt:15} strictness={rule3_strictness[bt]:.2f}")

rule3_strictness_tensor = torch.ones(NUM_BEAT_TYPE_CLASSES + 1, dtype=torch.float32)
for bt_name, idx in beat_type_value_to_idx.items():
    rule3_strictness_tensor[idx] = rule3_strictness.get(bt_name, 1.0)
rule3_strictness_tensor = rule3_strictness_tensor.to(device)
print(f"      [ΕΛΕΓΧΟΣ] πραγματικός rule3_strictness_tensor: {rule3_strictness_tensor.tolist()} "
      f"(ΠΡΕΠΕΙ να ΔΙΑΦΕΡΕΙ ανά θέση - αν είναι όλα 1.0, κάτι έσπασε)")

# --- 1Η. ΚΑΝΟΝΑΣ 4 (κίνηση ρίζας) — ΔΙΟΡΘΩΜΕΝΗ ΕΚΔΟΣΗ: μετράμε ΜΟΝΟ πραγματικές
# ΑΛΛΑΓΕΣ ρίζας (prev != curr), όχι επαναλήψεις. Η προηγούμενη εκδοχή μέτραγε ΚΑΙ
# τις επαναλήψεις, οπότε ο πίνακας κυριαρχούνταν από "ρίζα -> ίδια ρίζα" (~80-85%)
# - ουσιαστικά επανέλαβε ό,τι ήδη κάνει ο Κανόνας 3, με αντικρουόμενο βάρος, γι'
# αυτό χειροτέρευε τα αποτελέσματα. Τώρα: η διαγώνιος (ρίζα -> ίδια ρίζα) είναι
# ΟΥΔΕΤΕΡΗ (1.0, "πάντα εντάξει να μείνει ίδια" - αυτό το αποφασίζει ο Κανόνας 3),
# και μόνο το ΕΚΤΟΣ διαγωνίου μέρος (η ΚΑΤΕΥΘΥΝΣΗ όταν όντως αλλάζει η ρίζα, π.χ.
# V -> I) προέρχεται από μέτρηση.
root_trans_counts = np.zeros((NUM_ROOT_CLASSES, NUM_ROOT_CLASSES), dtype=np.float64)
for _, chord_seq in raw_songs:
    for t in range(1, len(chord_seq)):
        prev_idx = chord_root_idx[chord_seq[t - 1]]
        curr_idx = chord_root_idx[chord_seq[t]]
        if prev_idx != curr_idx:
            root_trans_counts[prev_idx, curr_idx] += 1

row_sums = root_trans_counts.sum(axis=1, keepdims=True)
offdiag_target_np = np.where(
    row_sums > 0, root_trans_counts / np.where(row_sums == 0, 1, row_sums),
    1.0 / max(NUM_ROOT_CLASSES - 1, 1)
).astype(np.float32)

root_transition_target_np = offdiag_target_np.copy()
for i in range(NUM_ROOT_CLASSES):
    root_transition_target_np[i, i] = 1.0  # ουδέτερο: "μένει ίδια" -> δεν τιμωρείται/επιβραβεύεται εδώ
root_transition_target = torch.tensor(root_transition_target_np, dtype=torch.float32).to(device)

row_weights = row_sums.flatten()
row_max_probs = offdiag_target_np.max(axis=1)
total_w = row_weights.sum()
avg_max_prob = float((row_weights * row_max_probs).sum() / total_w) if total_w > 0 else 1.0 / max(NUM_ROOT_CLASSES - 1, 1)
root_unpredictability = 1.0 - avg_max_prob
RULE4_WEIGHT = (1.0 / (root_unpredictability + 0.01)) - 1
top_transitions = []
for i in range(NUM_ROOT_CLASSES):
    j = int(offdiag_target_np[i].argmax())
    top_transitions.append((idx_to_root_pc.get(i, '?'), idx_to_root_pc.get(j, '?'), offdiag_target_np[i, j]))
print(f"   Κανόνας 4 (κίνηση ρίζας, ΜΟΝΟ πραγματικές αλλαγές): μέση μέγιστη πιθανότητα κατεύθυνσης={avg_max_prob*100:.1f}% -> "
      f"RULE4_WEIGHT={RULE4_WEIGHT:.2f}")
print("   Πιο συχνή ΚΑΤΕΥΘΥΝΣΗ αλλαγής ανά ρίζα (όταν όντως αλλάζει - όχι επανάληψη):")
for prev_pc, next_pc, prob in sorted(top_transitions, key=lambda x: -x[2])[:6]:
    print(f"      {prev_pc} -> {next_pc}  ({prob*100:.1f}%)")

# --- 1Θ. ΚΑΝΟΝΑΣ 5 (επίλυση leading tone -> τονική) ---
# Όταν μια νότα ΕΙΝΑΙ leading tone (mode-aware, βλ. παραπάνω) ΚΑΙ όντως επιλύεται
# (η επόμενη νότα φτάνει στην τονική), αυτό είναι ισχυρή ένδειξη ότι η αρμονία εκεί
# είναι dominant-function - άρα η νότα ΠΡΕΠΕΙ να είναι πραγματική chord tone πιο
# αξιόπιστα απ' ό,τι κατά μέσο όρο. Μετράμε το ΠΡΑΓΜΑΤΙΚΟ ποσοστό παραβίασης ΣΕ
# ΑΥΤΕΣ ΤΙΣ ΘΕΣΕΙΣ και το συγκρίνουμε με το γενικό, για να βγάλουμε πόσο πιο
# ΑΥΣΤΗΡΟΣ πρέπει να είναι ο βασικός κανόνας εκεί (ίδια λογική με τον Κανόνα 1,
# απλά προς την αντίθετη κατεύθυνση - αυστηρότητα αντί για επιείκεια).
lt_violation_count, lt_total_count = 0, 0
for melody_iris, chord_iris in raw_songs:
    for t in range(len(melody_iris) - 1):
        n = melody_iris[t]
        if not note_is_leading_tone.get(n):
            continue
        next_pc = note_pitch_class_transposed.get(melody_iris[t + 1])
        if next_pc != note_expected_tonic_pc.get(n):
            continue  # δεν επιλύθηκε πραγματικά -> δεν μετράει σαν "resolving leading tone"
        lt_total_count += 1
        n_idx = note_to_idx[n]
        c_idx = chord_to_idx[chord_iris[t]]
        if compatibility_matrix[n_idx, c_idx] <= 0.5:
            lt_violation_count += 1

lt_violation_rate = (lt_violation_count / lt_total_count) if lt_total_count > 0 else (baseline_error / 100.0)
lt_scale = 1.0 / (lt_violation_rate + 0.05)
overall_scale_reference = 1.0 / ((baseline_error / 100.0) + 0.05)
RULE5_LENIENCY_FACTOR = lt_scale / overall_scale_reference  # >1 -> αυστηρότερος έλεγχος εδώ, <1 -> πιο χαλαρός
print(f"   Κανόνας 5 (επίλυση leading tone): {lt_total_count} περιπτώσεις resolving leading tone στο corpus, "
      f"ποσοστό παραβίασης εκεί={lt_violation_rate*100:.1f}% (γενικό: {baseline_error:.1f}%) -> "
      f"strictness factor={RULE5_LENIENCY_FACTOR:.2f}")

# --- 1Ι. ΚΑΝΟΝΑΣ 6 (λειτουργική αρμονία: Tonic / Subdominant / Dominant) ---
# Πρότυπη λειτουργική ταξινόμηση (T-S-D), mode-aware: το ΙΔΙΟ transposed pitch class
# αντιστοιχεί σε ΔΙΑΦΟΡΕΤΙΚΗ βαθμίδα ανάλογα με το mode του τραγουδιού (ίδιο θέμα
# με τον Κανόνα 5). Οι αντιστοιχίσεις VI/VII σε minor είναι μια λογική, καθιερωμένη
# σύμβαση (submediant/subtonic ως tonic-substitute / dominant-preparation
# αντίστοιχα) - μπορεί να αλλάξει αν διαφωνείς.
MAJOR_FUNCTION_BY_OFFSET = {0: 'T', 2: 'S', 4: 'T', 5: 'S', 7: 'D', 9: 'T', 11: 'D'}
MINOR_FUNCTION_BY_OFFSET = {0: 'T', 2: 'S', 3: 'T', 5: 'S', 7: 'D', 8: 'T', 10: 'D', 11: 'D'}
FUNCTION_TO_IDX = {'T': 0, 'S': 1, 'D': 2, 'X': 3}  # X = μη-διατονική/άγνωστη βαθμίδα
NUM_FUNCTION_CLASSES = len(FUNCTION_TO_IDX)

def chord_function_label(root_pc, is_major):
    if not isinstance(root_pc, int):
        return 'X'
    tonic_pc = 0 if is_major else 9
    offset = (root_pc - tonic_pc) % 12
    table = MAJOR_FUNCTION_BY_OFFSET if is_major else MINOR_FUNCTION_BY_OFFSET
    return table.get(offset, 'X')

chord_function = {c: chord_function_label(chord_root_pc_transposed[c], chord_is_major[c]) for c in all_chords_iris}

# Μετράμε ΜΟΝΟ πραγματικές αλλαγές function (ίδια λογική διόρθωση με τον Κανόνα 4 -
# αγνοούμε επαναλήψεις, ώστε να μην κυριαρχεί ξανά το "μένει ίδιο").
# ΔΙΟΡΘΩΣΗ (μετά από εμπειρικό έλεγχο - βλ. check_rule6_by_mode.py): οι ΠΡΟΤΙΜΩΜΕΝΕΣ
# κατευθύνσεις διαφέρουν ΠΟΙΟΤΙΚΑ ανάμεσα σε major/minor (major: T<->S ταλάντωση,
# minor: κατευθείαν προς D και από τα δύο T ΚΑΙ S - πιθανότατα λόγω της ισχυρότερης
# έλξης της dominant στην ελάσσονα, με την αναβαθμισμένη leading tone). Ένας ΜΙΚΤΟΣ
# πίνακας-στόχος (όπως πριν) κυριαρχείται από το major (75% των μεταβάσεων), άρα
# εκπαιδεύει τα minor τραγούδια προς ΛΑΘΟΣ (κυρίως major-shaped) στόχο. Χτίζουμε
# ΞΕΧΩΡΙΣΤΟΥΣ πίνακες-στόχους ανά mode, ΙΔΙΑ σύμβαση [2,F,F] (0=minor, 1=major) με
# το ήδη υπάρχον root_to_func_by_mode.
def build_function_transition_target(songs, want_major):
    counts = np.zeros((NUM_FUNCTION_CLASSES, NUM_FUNCTION_CLASSES), dtype=np.float64)
    for _, chord_seq in songs:
        if not chord_seq:
            continue
        if chord_is_major.get(chord_seq[0], True) != want_major:
            continue
        for t in range(1, len(chord_seq)):
            f_prev = FUNCTION_TO_IDX[chord_function[chord_seq[t - 1]]]
            f_curr = FUNCTION_TO_IDX[chord_function[chord_seq[t]]]
            if f_prev != f_curr:
                counts[f_prev, f_curr] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    offdiag = np.where(row_sums > 0, counts / np.where(row_sums == 0, 1, row_sums),
                        1.0 / max(NUM_FUNCTION_CLASSES - 1, 1)).astype(np.float32)
    target = offdiag.copy()
    for i in range(NUM_FUNCTION_CLASSES):
        target[i, i] = 1.0  # ουδέτερο: "μένει ίδια function" -> δεν αφορά τον Κανόνα 6
    return target, counts, row_sums, offdiag

function_transition_target_major_np, func_trans_counts_major, func_row_sums_major, func_offdiag_major = \
    build_function_transition_target(raw_songs, want_major=True)
function_transition_target_minor_np, func_trans_counts_minor, func_row_sums_minor, func_offdiag_minor = \
    build_function_transition_target(raw_songs, want_major=False)

# Για το print/βαθμονόμηση του RULE6_WEIGHT, ΣΥΝΕΧΙΖΟΥΜΕ να χρησιμοποιούμε το
# ΜΙΚΤΟ στατιστικό (η ΣΥΝΟΛΙΚΗ αυστηρότητα βγήκε παρόμοια ανά mode στον έλεγχο -
# βλ. check_rule6_by_mode.py - άρα ΕΝΑ βάρος παραμένει λογικό, μόνο ο ΣΤΟΧΟΣ
# χρειαζόταν διαχωρισμό, όχι η ΑΥΣΤΗΡΟΤΗΤΑ).
func_trans_counts = func_trans_counts_major + func_trans_counts_minor
func_row_sums = func_trans_counts.sum(axis=1, keepdims=True)
func_offdiag_target_np = np.where(
    func_row_sums > 0, func_trans_counts / np.where(func_row_sums == 0, 1, func_row_sums),
    1.0 / max(NUM_FUNCTION_CLASSES - 1, 1)
).astype(np.float32)

func_row_weights = func_row_sums.flatten()
func_row_max_probs = func_offdiag_target_np.max(axis=1)
func_total_w = func_row_weights.sum()
func_avg_max_prob = float((func_row_weights * func_row_max_probs).sum() / func_total_w) if func_total_w > 0 else 1.0 / max(NUM_FUNCTION_CLASSES - 1, 1)
func_unpredictability = 1.0 - func_avg_max_prob
RULE6_WEIGHT = (1.0 / (func_unpredictability + 0.01)) - 1
idx_to_function = {v: k for k, v in FUNCTION_TO_IDX.items()}
print(f"   Κανόνας 6 (λειτουργική αρμονία T/S/D): μέση μέγιστη πιθανότητα κατεύθυνσης={func_avg_max_prob*100:.1f}% -> "
      f"RULE6_WEIGHT={RULE6_WEIGHT:.2f}")
print("   Πιο συχνή ΚΑΤΕΥΘΥΝΣΗ αλλαγής ανά function (όταν όντως αλλάζει, ΜΙΚΤΟ - μόνο για το βάρος):")
for i in range(NUM_FUNCTION_CLASSES):
    j = int(func_offdiag_target_np[i].argmax())
    print(f"      {idx_to_function[i]} -> {idx_to_function[j]}  ({func_offdiag_target_np[i, j]*100:.1f}%)")
print("   Κανόνας 6, στόχος ΑΝΑ mode (major/minor) - βλ. παρακάτω ποιος από τους δύο χρησιμοποιείται τελικά στο loss:")
print("      MAJOR:")
for i in range(NUM_FUNCTION_CLASSES):
    j = int(func_offdiag_major[i].argmax())
    print(f"         {idx_to_function[i]} -> {idx_to_function[j]}  ({func_offdiag_major[i, j]*100:.1f}%)")
print("      MINOR:")
for i in range(NUM_FUNCTION_CLASSES):
    j = int(func_offdiag_minor[i].argmax())
    print(f"         {idx_to_function[i]} -> {idx_to_function[j]}  ({func_offdiag_minor[i, j]*100:.1f}%)")

if ENABLE_RULE6_MODE_SPLIT:
    function_transition_target_by_mode = torch.stack([
        torch.tensor(function_transition_target_minor_np, dtype=torch.float32),
        torch.tensor(function_transition_target_major_np, dtype=torch.float32),
    ], dim=0).to(device)  # idx 0=minor, 1=major - ΙΔΙΑ σύμβαση με root_to_func_by_mode
    print("   Κανόνας 6: mode-split στόχος ΕΝΕΡΓΟΣ - το loss χρησιμοποιεί τον ΞΕΧΩΡΙΣΤΟ πίνακα ανά mode (πάνω).")
else:
    # ΙΔΙΟ blended target ΚΑΙ στους δύο δείκτες mode, ώστε το self.function_transition_target[is_major]
    # στο forward() να επιλέγει ΠΑΝΤΑ τον ίδιο πίνακα - αναπαράγει ακριβώς τη
    # συμπεριφορά πριν το mode-split (καμία αλλαγή χρειάζεται στο forward pass).
    _blended_target = func_offdiag_target_np.copy()
    for _i in range(NUM_FUNCTION_CLASSES):
        _blended_target[_i, _i] = 1.0  # ίδιο ουδέτερο diagonal-fill με τους per-mode πίνακες
    function_transition_target_by_mode = torch.stack([
        torch.tensor(_blended_target, dtype=torch.float32),
        torch.tensor(_blended_target, dtype=torch.float32),
    ], dim=0).to(device)
    print("   Κανόνας 6: mode-split στόχος ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ - το loss χρησιμοποιεί το ΚΟΙΝΟ/μικτό target (πάνω, 'μέσος όρος').")

# Tensor lookup: root_idx -> function_idx, ξεχωριστά για major/minor τραγούδια
# (χρειάζεται στο loss για να "μεταφράσει" τις προβλέψεις root σε function ανά
# batch item, ανάλογα με το mode ΤΟΥ ΣΥΓΚΕΚΡΙΜΕΝΟΥ chunk).
major_root_to_func = torch.zeros(NUM_ROOT_CLASSES, NUM_FUNCTION_CLASSES)
minor_root_to_func = torch.zeros(NUM_ROOT_CLASSES, NUM_FUNCTION_CLASSES)
for r_idx in range(NUM_ROOT_CLASSES):
    root_pc = idx_to_root_pc.get(r_idx)
    major_root_to_func[r_idx, FUNCTION_TO_IDX[chord_function_label(root_pc, True)]] = 1.0
    minor_root_to_func[r_idx, FUNCTION_TO_IDX[chord_function_label(root_pc, False)]] = 1.0
root_to_func_by_mode = torch.stack([minor_root_to_func, major_root_to_func], dim=0).to(device)  # idx 0=minor, 1=major

# --- 1Κ. ΚΑΝΟΝΑΣ 7 (τονική στην αρχή και στο τέλος του τραγουδιού) ---
# Δομικός κανόνας - ΟΧΙ για γειτονικές θέσεις (όπως οι Κανόνες 1-6) αλλά για τη
# ΠΡΩΤΗ και ΤΕΛΕΥΤΑΙΑ νότα ΟΛΟΚΛΗΡΟΥ του τραγουδιού. Μετράμε αρχή/τέλος ΞΕΧΩΡΙΣΤΑ,
# αφού μπορεί να διαφέρουν (π.χ. μια εισαγωγή/pickup να ξεκινά αλλού πιο συχνά απ'
# ό,τι ένα τραγούδι τελειώνει εκτός τονικής). Μετρήσαμε ΚΑΙ τα δύο - η αρχή αποδείχθηκε
# ΟΧΙ πλειοψηφική τάση (31.7%, όχι πραγματική σύμβαση) ενώ το τέλος είναι (75.2%,
# ξεκάθαρη τάση "τονικής επίλυσης") - άρα ο Κανόνας 7 τώρα αφορά ΜΟΝΟ το τέλος. Η
# στατιστική της αρχής μένει μόνο σαν ενημερωτικό στοιχείο (print), όχι στο loss.
start_tonic_count, end_tonic_count, n_songs_checked = 0, 0, 0
for _, chord_seq in raw_songs:
    if len(chord_seq) == 0:
        continue
    n_songs_checked += 1
    if chord_function[chord_seq[0]] == 'T':
        start_tonic_count += 1
    if chord_function[chord_seq[-1]] == 'T':
        end_tonic_count += 1

start_tonic_rate = start_tonic_count / n_songs_checked if n_songs_checked > 0 else 0.0
end_tonic_rate = end_tonic_count / n_songs_checked if n_songs_checked > 0 else 0.0
RULE7_END_WEIGHT = (1.0 / ((1.0 - end_tonic_rate) + 0.05)) - 1
print(f"   Κανόνας 7 (τονική στο τέλος): {n_songs_checked} τραγούδια -> "
      f"αρχή σε tonic={start_tonic_rate*100:.1f}% (ενημερωτικά, ΔΕΝ επιβάλλεται - όχι πλειοψηφία), "
      f"τέλος σε tonic={end_tonic_rate*100:.1f}% (βάρος={RULE7_END_WEIGHT:.2f})")

# --- 1Λ. ΚΑΝΟΝΑΣ 8 (ποιότητα χορδής ανάλογα με τη λειτουργία) ---
# ΔΙΑΦΟΡΕΤΙΚΟΣ τύπος κανόνα από τους 3/4/6/7: ΔΕΝ συγκρίνει γειτονικές θέσεις στο
# χρόνο - συγκρίνει τα ΔΥΟ heads (root, quality) ΣΤΗΝ ΙΔΙΑ θέση. Μετράμε P(quality |
# function) ανά ΝΟΤΑ (όχι ανά μοναδική χορδή), ίδια λογική με τα quality class weights.
qual_given_func_counts = np.zeros((NUM_FUNCTION_CLASSES, NUM_QUALITY_CLASSES), dtype=np.float64)
for _, chord_seq in raw_songs:
    for c in chord_seq:
        f_idx = FUNCTION_TO_IDX[chord_function[c]]
        q_idx = chord_quality_idx[c]
        qual_given_func_counts[f_idx, q_idx] += 1

qual_func_row_sums = qual_given_func_counts.sum(axis=1, keepdims=True)
quality_given_function_target_np = np.where(
    qual_func_row_sums > 0, qual_given_func_counts / np.where(qual_func_row_sums == 0, 1, qual_func_row_sums),
    1.0 / NUM_QUALITY_CLASSES
).astype(np.float32)
quality_given_function_target = torch.tensor(quality_given_function_target_np, dtype=torch.float32).to(device)

qual_func_row_weights = qual_func_row_sums.flatten()
qual_func_row_max_probs = quality_given_function_target_np.max(axis=1)
qual_func_total_w = qual_func_row_weights.sum()
qual_func_avg_max_prob = float((qual_func_row_weights * qual_func_row_max_probs).sum() / qual_func_total_w) if qual_func_total_w > 0 else 1.0 / NUM_QUALITY_CLASSES
qual_func_unpredictability = 1.0 - qual_func_avg_max_prob
RULE8_WEIGHT = (1.0 / (qual_func_unpredictability + 0.01)) - 1
idx_to_quality_r8 = {idx: q for q, idx in quality_to_idx.items()}
print(f"   Κανόνας 8 (ποιότητα ανά λειτουργία): μέση μέγιστη πιθανότητα={qual_func_avg_max_prob*100:.1f}% -> "
      f"RULE8_WEIGHT={RULE8_WEIGHT:.2f}")
print("   Πιο συχνή ποιότητα ανά λειτουργία:")
for f_idx in range(NUM_FUNCTION_CLASSES):
    top3 = sorted(range(NUM_QUALITY_CLASSES), key=lambda q: -quality_given_function_target_np[f_idx, q])[:3]
    top3_str = ", ".join(f"{idx_to_quality_r8.get(q,'?')}={quality_given_function_target_np[f_idx,q]*100:.1f}%" for q in top3)
    print(f"      {idx_to_function[f_idx]}: {top3_str}")

print(f"📦 Κανονικοποίηση key ολοκληρώθηκε: όλα τα τραγούδια μεταφέρθηκαν σε C major / A minor.")

# =============================================================================
# ΕΝΣΩΜΑΤΩΣΗ chord-melody-dataset - PART B: ΕΠΑΛΗΘΕΥΣΗ
# =============================================================================
# Τα τραγούδια του chord-melody-dataset ΗΔΗ μεταφέρθηκαν σε C major/A minor
# στο δικό μας extraction pipeline. Η ΥΠΑΡΧΟΥΣΑ ανίχνευση key μόλις έτρεξε
# ΞΑΝΑ πάνω τους (δεν διακρίνει "νέα" από "παλιά" τραγούδια) - αν συμφώνησε
# ότι είναι ήδη κανονικά, η μετάθεση ήταν 0 και όλα εντάξει. Το ελέγχουμε
# ΑΜΕΣΑ αντί να το υποθέσουμε.
if INCLUDE_CHORD_MELODY_DATASET and _synthetic_note_iris:
    import re as _re
    _PITCH_NAME_RE_CHECK = _re.compile(r'^Pitch_([A-G])(_sharp_|_flat_)?(-?\d+)$')
    _LETTER_SEMITONE_CHECK = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

    def _original_pc(pitch_str):
        m = _PITCH_NAME_RE_CHECK.match(pitch_str)
        letter, accidental = m.group(1), m.group(2)
        base = _LETTER_SEMITONE_CHECK[letter]
        if accidental == '_sharp_':
            base += 1
        elif accidental == '_flat_':
            base -= 1
        return base % 12

    _shift_mismatches = 0
    _checked = 0
    for _iri in _synthetic_note_iris:
        _orig_pc = _original_pc(note_features[_iri]['pitch'])
        _transposed_pc = note_pitch_class_transposed.get(_iri)
        if _transposed_pc is None:
            continue
        _checked += 1
        if _transposed_pc != _orig_pc:
            _shift_mismatches += 1

    print("\n" + "=" * 78)
    print("ΕΠΑΛΗΘΕΥΣΗ: παρέμειναν τα νέα τραγούδια κανονικά μετά την ΥΠΑΡΧΟΥΣΑ ανίχνευση key;")
    print("=" * 78)
    print(f"   Ελέγχθηκαν: {_checked} νότες | shift=0 (αναμενόμενο): {_checked - _shift_mismatches} | "
          f"μετατέθηκαν ξανά: {_shift_mismatches}")
    if _shift_mismatches == 0:
        print("   ✅ Όλα εντάξει.")
    else:
        print(f"   ⚠️ {_shift_mismatches/_checked*100:.1f}% μετατέθηκαν ξανά - άξιζε έλεγχος πριν εμπιστευτούμε πλήρως.")

# =============================================================================
# ΕΝΣΩΜΑΤΩΣΗ HookTheory - PART B: ΕΠΑΛΗΘΕΥΣΗ (ΔΙΑΦΟΡΕΤΙΚΗ λογική από πάνω)
# =============================================================================
# ΣΗΜΑΝΤΙΚΗ ΔΙΑΦΟΡΑ από το chord-melody-dataset: το HookTheory ΔΕΝ μεταφέρθηκε
# σε κανονικό key κατά την εξαγωγή - μόνο η ΓΡΑΦΗ (letter names) χτίστηκε
# χρησιμοποιώντας το key της HookTheory. Άρα ΕΔΩ ΠΕΡΙΜΕΝΟΥΜΕ η υπάρχουσα
# ανίχνευση key να εφαρμόσει ΠΡΑΓΜΑΤΙΚΗ μετάθεση (shift != 0 είναι φυσιολογικό,
# ΟΧΙ πρόβλημα). Αντί γι' αυτό, ελέγχουμε κάτι ουσιαστικό: παράγει η ανίχνευση
# key ΛΟΓΙΚΟ αποτέλεσμα σε αυτή τη νέα πηγή - δηλαδή, καταλήγουν τα τραγούδια
# HookTheory να "τελειώνουν σε τονική" σε ΠΑΡΟΜΟΙΟ ποσοστό με την υπόλοιπη
# corpus (~75%, βλ. Κανόνας 7 παραπάνω); Αν είναι πολύ χαμηλότερο, αυτό θα
# ήταν ένδειξη ότι κάτι δεν λειτουργεί σωστά σε αυτή τη νέα πηγή.
if INCLUDE_HOOKTHEORY_DATASET and _hk_synthetic_note_iris:
    _hk_synthetic_set = set(_hk_synthetic_note_iris)
    _hk_tonic_hits, _hk_tonic_total = 0, 0
    for _melody_iris, _chord_iris in raw_songs:
        if not _melody_iris or _melody_iris[0] not in _hk_synthetic_set:
            continue
        if not _chord_iris:
            continue
        _last_chord_iri = _chord_iris[-1]
        _func = chord_function.get(_last_chord_iri)
        if _func is None:
            continue
        _hk_tonic_total += 1
        if _func == 'T':
            _hk_tonic_hits += 1

    print("\n" + "=" * 78)
    print("ΕΠΑΛΗΘΕΥΣΗ HookTheory: ποσοστό τραγουδιών που ΤΕΛΕΙΩΝΟΥΝ σε τονική")
    print("=" * 78)
    if _hk_tonic_total > 0:
        _hk_rate = _hk_tonic_hits / _hk_tonic_total * 100
        print(f"   {_hk_tonic_hits}/{_hk_tonic_total} τραγούδια HookTheory ({_hk_rate:.1f}%) "
              f"τελειώνουν σε λειτουργία τονικής")
        print(f"   Για σύγκριση, όλη η corpus (Κανόνας 7 παραπάνω): ~75% (βλ. τιμή 'τέλος σε tonic' πιο πάνω)")
        if _hk_rate < 55:
            print("   ⚠️ ΣΗΜΑΝΤΙΚΑ χαμηλότερο - άξιζε έλεγχος πριν εμπιστευτούμε πλήρως το merge.")
        else:
            print("   ✅ Λογικό εύρος - η ανίχνευση key φαίνεται να δουλεύει σωστά σε αυτή τη νέα πηγή.")
    else:
        print("   ⚠️ Δεν βρέθηκαν έγκυρα τραγούδια HookTheory για έλεγχο.")

# --- 2. POSITIONAL ENCODING ---
# --- 2. (ΠΑΛΙΑ ΑΠΟΛΥΤΗ PositionalEncoding ΑΦΑΙΡΕΘΗΚΕ) ---
# Το x-transformers Encoder (βλ. παρακάτω) χειρίζεται τη θέση ΜΕΣΑ στο attention
# (σχετική απόσταση), όχι προσθέτοντας ένα διάνυσμα θέσης στο input όπως πριν - οι
# δύο μηχανισμοί δεν πρέπει να συνδυαστούν (θα ήταν πλεονασμός/σύγκρουση σημάτων).

# --- 3. HARMONIZER: bidirectional encoder μόνο, ΧΩΡΙΣ chord-history conditioning ---
# ΓΙΑΤΙ γυρίζουμε πίσω σε αυτό το απλούστερο σχήμα: το πρώτο πετυχημένο run (πριν
# από masking/autoregressive πειράματα) έδωσε ~40% joint accuracy με ΑΚΡΙΒΩΣ αυτή
# την αρχιτεκτονική. Κάθε αυτοπαλινδρομική παραλλαγή έκτοτε (masked bidirectional,
# seq2seq με causal decoder) έδωσε ΧΕΙΡΟΤΕΡΑ αποτελέσματα, επειδή η τροφοδότηση της
# προηγούμενης χορδής σαν input δίνει στο μοντέλο μια "διαρροή": με δεδομένο ότι μια
# χορδή συχνά κρατάει για πολλές διαδοχικές νότες, το μοντέλο μαθαίνει να αντιγράφει
# την προηγούμενη χορδή αντί να ακούει τη μελωδία — και υπό greedy autoregressive
# decoding αυτό εκφυλίζεται σε μόνιμη επανάληψη μίας τιμής (ό,τι είδαμε στο log: 100%
# των προβλέψεων ήταν η ίδια κλάση, ΛΑΝΘΑΣΜΕΝΗ 97.5% των φορών, ενώ μια απλή "μάντεψε
# την πιο συχνή κλάση" στρατηγική θα πετύχαινε 56.2% δωρεάν). Η λύση: αφαιρούμε
# εντελώς το chord-history conditioning· κάθε θέση προβλέπεται ανεξάρτητα από τη
# μελωδία γύρω της, όπως στο αρχικό, επιτυχημένο σχήμα.
class HarmonizerTransformer(nn.Module):
    def __init__(self, num_pitch_classes, pad_pitch_id, num_duration_classes, pad_duration_id,
                 num_beat_type_classes, pad_beat_type_id, num_position_classes, pad_position_id,
                 num_interval_classes, pad_interval_id, embed_dim, num_heads, num_layers,
                 num_root_classes, num_quality_classes, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        # Πέντε ξεχωριστά embeddings, ένα ανά feature, αθροίζονται σε μία αναπαράσταση
        # ανά νότα (ίδια λογική με token+position+segment embeddings σε BERT).
        self.pitch_embedding = nn.Embedding(num_pitch_classes + 1, embed_dim, padding_idx=pad_pitch_id)
        self.duration_embedding = nn.Embedding(num_duration_classes + 1, embed_dim, padding_idx=pad_duration_id)
        self.beat_type_embedding = nn.Embedding(num_beat_type_classes + 1, embed_dim, padding_idx=pad_beat_type_id)
        self.position_embedding = nn.Embedding(num_position_classes + 1, embed_dim, padding_idx=pad_position_id)
        self.interval_embedding = nn.Embedding(num_interval_classes + 1, embed_dim, padding_idx=pad_interval_id)
        self.beat_strength_proj = nn.Linear(1, embed_dim)  # συνεχής τιμή -> ίδιο embedding space

        # x-transformers Encoder: rel_pos_bias=True -> T5-style ΣΧΕΤΙΚΗ θετική
        # μεροληψία στο attention (αντί για την παλιά απόλυτη sinusoidal). ff_mult=1
        # για να ταιριάξει ΑΚΡΙΒΩΣ το παλιό dim_feedforward=256 (=embed_dim, όχι το
        # σύνηθες 4x) - ΕΝΑ μεταβλητό πείραμα τη φορά, όχι αλλαγή χωρητικότητας.
        # ΣΗΜΕΙΩΣΗ: αν το ff_mult πετάξει TypeError, έλεγξε το τρέχον README του
        # x-transformers για το σωστό όνομα ορίσματος πλάτους feedforward - δεν
        # μπόρεσα να το επιβεβαιώσω 100% χωρίς να τρέξω τον κώδικα.
        #
        # use_rmsnorm=True: ΔΙΟΡΘΩΣΗ μετά από εκτεταμένο διαγνωστικό (state_dict
        # σύγκριση φορτωμένου vs φρέσκου μοντέλου) που βρήκε τις παραμέτρους
        # ".gamma" (default norm του x-transformers, ScaleNorm-style πολλαπλασιαστικό
        # gamma) να έχουν "εκραγεί" σε τιμές ~10^7-10^8 μετά την εκπαίδευση -
        # πιθανότατα η αιτία της κατάρρευσης πρόβλεψης ("πάντα A/C ανεξαρτήτου
        # input"). Η τεκμηρίωση του x-transformers ΙΔΙΑ σημειώνει ότι το gamma
        # αυτό είναι εύθραυστο ("removing gamma led to no performance degradation",
        # χρειάζεται ειδικό offset trick για ασφαλές weight decay). Το RMSNorm
        # (χρησιμοποιείται σε LLaMA/Retro/Gopher) είναι η σταθερή, καθιερωμένη
        # εναλλακτική που προτείνει η ίδια η βιβλιοθήκη.
        self.transformer_encoder = XTEncoder(
            dim=embed_dim, depth=num_layers, heads=num_heads,
            attn_dropout=dropout, ff_dropout=dropout, ff_mult=1,
            rel_pos_bias=True, use_rmsnorm=True
        )
        self.fc_root = nn.Linear(embed_dim, num_root_classes)
        self.fc_quality = nn.Linear(embed_dim, num_quality_classes)

    def forward(self, pitch_ids, duration_ids, beat_type_ids, position_ids, interval_ids,
                beat_strength, pad_mask=None):
        x = (self.pitch_embedding(pitch_ids) + self.duration_embedding(duration_ids)
             + self.beat_type_embedding(beat_type_ids) + self.position_embedding(position_ids)
             + self.interval_embedding(interval_ids) + self.beat_strength_proj(beat_strength.unsqueeze(-1)))
        x = x * math.sqrt(self.embed_dim)
        # x-transformers: mask=True σημαίνει "έγκυρη θέση, πρόσεξέ τη" - ΑΝΤΙΘΕΤΟ από
        # το PyTorch src_key_padding_mask (True=αγνόησε). Το pad_mask μας ΕΙΝΑΙ στη
        # σύμβαση PyTorch (True=padding), άρα ΑΝΤΙΣΤΡΕΦΟΥΜΕ το με ~pad_mask.
        attn_mask = ~pad_mask if pad_mask is not None else None
        shared = self.transformer_encoder(x, mask=attn_mask)
        return self.fc_root(shared), self.fc_quality(shared)


# =============================================================================
# ΝΕΟ, ΞΕΧΩΡΙΣΤΟ μοντέλο - ΔΕΝ αγγίζει το HarmonizerTransformer παραπάνω, ΔΕΝ
# επηρεάζει κανένα υπάρχον checkpoint/κώδικα. Χτίζεται για δύο ιδέες από την
# έρευνα βιβλιογραφίας (Kaliakatsos-Papakostas et al.):
#   #6 - FF curriculum: κατά την εκπαίδευση, το μοντέλο βλέπει ΚΑΙ τη μελωδία
#        ΚΑΙ ΕΝΑ ΤΜΗΜΑ των πραγματικών χορδών (οι υπόλοιπες = MASK token) -
#        στην αρχή ΟΛΕΣ masked, σταδιακά αποκαλύπτονται όλο και περισσότερες.
#   #1 - iterative "certain" unmasking: στο inference, αντί για μία πρόβλεψη σε
#        ΟΛΕΣ τις θέσεις μαζί, προβλέπουμε, κρατάμε τις πιο σίγουρες, τις
#        δίνουμε πίσω ως input, ξαναπροβλέπουμε τις υπόλοιπες, επαναλαμβάνουμε.
# ΚΑΙ οι δύο ιδέες χρειάζονται το ΙΔΙΟ νέο input channel (ποιες χορδές είναι
# ήδη γνωστές) - γι' αυτό χτίζονται ΜΑΖΙ, σε ΕΝΑ μοντέλο, με ΞΕΧΩΡΙΣΤΗ όμως
# επαλήθευση της καθεμιάς πριν εμπιστευτούμε το σύνολο.
class HarmonizerTransformerIterative(nn.Module):
    def __init__(self, num_pitch_classes, pad_pitch_id, num_duration_classes, pad_duration_id,
                 num_beat_type_classes, pad_beat_type_id, num_position_classes, pad_position_id,
                 num_interval_classes, pad_interval_id, num_chord_classes, mask_chord_id,
                 embed_dim, num_heads, num_layers, num_root_classes, num_quality_classes, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.pitch_embedding = nn.Embedding(num_pitch_classes + 1, embed_dim, padding_idx=pad_pitch_id)
        self.duration_embedding = nn.Embedding(num_duration_classes + 1, embed_dim, padding_idx=pad_duration_id)
        self.beat_type_embedding = nn.Embedding(num_beat_type_classes + 1, embed_dim, padding_idx=pad_beat_type_id)
        self.position_embedding = nn.Embedding(num_position_classes + 1, embed_dim, padding_idx=pad_position_id)
        self.interval_embedding = nn.Embedding(num_interval_classes + 1, embed_dim, padding_idx=pad_interval_id)
        self.beat_strength_proj = nn.Linear(1, embed_dim)
        # ΝΕΟ: chord-context embedding. mask_chord_id = num_chord_classes (ΙΔΙΑ
        # σύμβαση "pad/mask id = vocab_size" με ΟΛΑ τα άλλα embeddings παραπάνω -
        # ΕΝΑ ΠΑΡΑΠΑΝΩ index, ΜΕΤΑ τις πραγματικές κλάσεις, δεν μπερδεύεται ΠΟΤΕ
        # με πραγματική χορδή).
        self.chord_context_embedding = nn.Embedding(num_chord_classes + 1, embed_dim, padding_idx=mask_chord_id)
        self.mask_chord_id = mask_chord_id

        self.transformer_encoder = XTEncoder(
            dim=embed_dim, depth=num_layers, heads=num_heads,
            attn_dropout=dropout, ff_dropout=dropout, ff_mult=1,
            rel_pos_bias=True, use_rmsnorm=True
        )
        self.fc_root = nn.Linear(embed_dim, num_root_classes)
        self.fc_quality = nn.Linear(embed_dim, num_quality_classes)

    def forward(self, pitch_ids, duration_ids, beat_type_ids, position_ids, interval_ids,
                beat_strength, chord_context_ids, pad_mask=None):
        x = (self.pitch_embedding(pitch_ids) + self.duration_embedding(duration_ids)
             + self.beat_type_embedding(beat_type_ids) + self.position_embedding(position_ids)
             + self.interval_embedding(interval_ids) + self.beat_strength_proj(beat_strength.unsqueeze(-1))
             + self.chord_context_embedding(chord_context_ids))
        x = x * math.sqrt(self.embed_dim)
        attn_mask = ~pad_mask if pad_mask is not None else None
        shared = self.transformer_encoder(x, mask=attn_mask)
        return self.fc_root(shared), self.fc_quality(shared)


# =============================================================================
# ΝΕΟ, ΞΕΧΩΡΙΣΤΟ μοντέλο - πιστό αντίγραφο της ΠΡΑΓΜΑΤΙΚΗΣ αρχιτεκτονικής
# SEModular από το επίσημο repo του paper (train_utils.py/models.py στο
# github.com/NeuraLLMuse/SingleEncoderMHCrossAttention). ΒΑΣΙΚΗ διαφορά από το
# HarmonizerTransformerIterative παραπάνω: εκεί ΑΘΡΟΙΖΑΜΕ chord-context με
# μελωδία ΣΤΗΝ ΙΔΙΑ θέση - ΕΔΩ τα ΣΥΝΕΝΩΝΟΥΜΕ σε ΜΙΑ ακολουθία διπλάσιου
# μήκους (μελωδία πρώτο μισό, αρμονία δεύτερο) και τρέχουμε ΚΑΝΟΝΙΚΟ
# nn.TransformerEncoder πάνω σε ΟΛΟ - το "cross-attention" τους ΔΕΝ είναι
# ξεχωριστός μηχανισμός, προκύπτει από ΑΠΛΟ self-attention πάνω στην
# ενωμένη ακολουθία. ΔΙΑΤΗΡΟΥΜΕ ξεχωριστά fc_root/fc_quality (όχι ενιαίο flat
# λεξιλόγιο 348 κλάσεων όπως αυτοί) - ΑΠΑΡΑΙΤΗΤΟ για τους 8 κανόνες μας.
class HarmonizerConcatEncoder(nn.Module):
    def __init__(self, num_pitch_classes, pad_pitch_id, num_duration_classes, pad_duration_id,
                 num_beat_type_classes, pad_beat_type_id, num_position_classes, pad_position_id,
                 num_interval_classes, pad_interval_id, num_chord_classes, mask_chord_id,
                 embed_dim, num_heads, num_layers, num_root_classes, num_quality_classes,
                 max_seq_len, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Μελωδία: ΙΔΙΑ 6 embeddings/άθροισμα με πριν - αυτό ΔΕΝ αλλάζει.
        self.pitch_embedding = nn.Embedding(num_pitch_classes + 1, embed_dim, padding_idx=pad_pitch_id)
        self.duration_embedding = nn.Embedding(num_duration_classes + 1, embed_dim, padding_idx=pad_duration_id)
        self.beat_type_embedding = nn.Embedding(num_beat_type_classes + 1, embed_dim, padding_idx=pad_beat_type_id)
        self.position_embedding = nn.Embedding(num_position_classes + 1, embed_dim, padding_idx=pad_position_id)
        self.interval_embedding = nn.Embedding(num_interval_classes + 1, embed_dim, padding_idx=pad_interval_id)
        self.beat_strength_proj = nn.Linear(1, embed_dim)

        # Αρμονία: ΞΕΧΩΡΙΣΤΗ ενσωμάτωση - ΘΑ ΣΥΝΕΝΩΘΕΙ, ΔΕΝ θα αθροιστεί.
        self.chord_context_embedding = nn.Embedding(num_chord_classes + 1, embed_dim, padding_idx=mask_chord_id)
        self.mask_chord_id = mask_chord_id

        # ΝΕΟ: σταθερή (ΟΧΙ εκπαιδεύσιμη) ημιτονοειδής θεσιακή κωδικοποίηση,
        # επαναλαμβανόμενη και στα δύο μισά - ΑΚΡΙΒΩΣ όπως το SEModular τους.
        pe = self._sinusoidal_positional_encoding(max_seq_len, embed_dim)
        self.register_buffer('shared_pos', pe)  # [1, max_seq_len, embed_dim]

        # ΝΕΟ: ΚΑΝΟΝΙΚΟ PyTorch nn.TransformerEncoder (ΟΧΙ x_transformers) με
        # ff_mult=4 (dim_feedforward=embed_dim*4) - ΑΚΡΙΒΩΣ όπως αυτοί
        # (d_model=512, dim_feedforward=2048 => 2048/512=4).
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

    def forward(self, pitch_ids, duration_ids, beat_type_ids, position_ids, interval_ids,
                beat_strength, chord_context_ids, pad_mask=None):
        B, L = pitch_ids.shape
        melody_emb = (self.pitch_embedding(pitch_ids) + self.duration_embedding(duration_ids)
                      + self.beat_type_embedding(beat_type_ids) + self.position_embedding(position_ids)
                      + self.interval_embedding(interval_ids)
                      + self.beat_strength_proj(beat_strength.unsqueeze(-1)))
        harmony_emb = self.chord_context_embedding(chord_context_ids)

        # ΝΕΟ: ΣΥΝΕΝΩΣΗ (concat), ΟΧΙ άθροισμα - η ακολουθία γίνεται 2×L.
        full_seq = torch.cat([melody_emb, harmony_emb], dim=1)  # [B, 2L, D]
        full_pos = torch.cat([self.shared_pos[:, :L, :], self.shared_pos[:, :L, :]], dim=1)
        full_seq = full_seq + full_pos

        full_seq = self.input_norm(full_seq)
        full_seq = self.dropout(full_seq)

        # nn.TransformerEncoder: True=αγνόησε (ΙΔΙΑ σύμβαση με το δικό μας
        # pad_mask ήδη - ΔΕΝ χρειάζεται αντιστροφή, σε αντίθεση με το
        # x_transformers που χρησιμοποιούσαμε πριν).
        full_pad_mask = torch.cat([pad_mask, pad_mask], dim=1) if pad_mask is not None else None

        encoded = self.encoder(full_seq, src_key_padding_mask=full_pad_mask)
        encoded = self.output_norm(encoded)

        # ΝΕΟ: ΜΟΝΟ το δεύτερο μισό (αρμονία) χρησιμοποιείται για την έξοδο -
        # ΑΚΡΙΒΩΣ όπως encoded[:, -grid_length:, :] στο δικό τους μοντέλο.
        harmony_encoded = encoded[:, L:, :]
        return self.fc_root(harmony_encoded), self.fc_quality(harmony_encoded)


# --- 3Β. RULE-INFORMED LOSS ---
class RuleInformedLoss(nn.Module):
    def __init__(self, penalty_weight, rule_matrix, root_weights, quality_weights,
                 root_mapping, qual_mapping, beat_type_scale, step_leniency_factor=0.3,
                 enable_rules_1_2=True, expected_same_scale=None, rule3_weight=0.5,
                 enable_rule_3=True, root_transition_target=None, rule4_weight=0.5,
                 enable_rule_4=True, rule5_leniency_factor=1.0, enable_rule_5=True,
                 root_to_func_by_mode=None, function_transition_target=None, rule6_weight=0.5,
                 enable_rule_6=True, rule7_end_weight=0.5,
                 tonic_function_idx=0, enable_rule_7=True, quality_given_function_target=None,
                 rule8_weight=0.5, enable_rule_8=True, rule3_strictness_scale=None,
                 centroid_table=None, centroid_mask_table=None,
                 tonal_distance_weight=0.3, enable_tonal_distance_loss=True):
        super().__init__()
        self._rule3_debug_done = False  # ΔΙΑΓΝΩΣΤΙΚΟ: τυπώνει ΠΡΑΓΜΑΤΙΚΕΣ τιμές runtime
        # ΜΙΑ φορά (πρώτο forward() call) - ελέγχει αν το strictness ΟΝΤΩΣ διαφέρει
        # ανά θέση, όχι μόνο θεωρητικά στον κώδικα.
        self.penalty_weight = penalty_weight
        self.comp_matrix = rule_matrix.to(device)
        self.root_weights = root_weights
        self.quality_weights = quality_weights
        self.root_mapping = root_mapping
        self.qual_mapping = qual_mapping
        self.beat_type_scale = beat_type_scale  # [num_beat_type_classes+1] tensor, Κανόνας 1
        self.step_leniency_factor = step_leniency_factor  # Κανόνας 2: πόσο "ελαφραίνει" μια σωστά επιλυμένη παραβίαση
        # ABLATION SWITCH: enable_rules_1_2=False -> leniency πάντα 1.0, δηλαδή
        # ακριβώς η συμπεριφορά ΠΡΙΝ τους Κανόνες 1 & 2 (μόνο το βασικό chord-tone
        # rule). Χρήσιμο για να απομονώσουμε αν η πτώση ακρίβειας οφείλεται στο
        # vocabulary cleanup ή στους δύο νέους κανόνες.
        self.enable_rules_1_2 = enable_rules_1_2
        self.expected_same_scale = expected_same_scale  # [num_beat_type_classes+1] tensor, Κανόνας 3
        self.rule3_weight = rule3_weight
        self.enable_rule_3 = enable_rule_3
        # ΕΝΙΣΧΥΣΗ Κανόνα 3: πολλαπλασιαστική αυστηρότητα ανά beat_type (βλ. σχόλιο
        # στην κατασκευή του rule3_strictness_tensor παραπάνω στο script) - None ->
        # συμπεριφορά ταυτόσημη με πριν (ουδέτερο 1.0 παντού), ώστε να μπορεί να
        # απενεργοποιηθεί εύκολα σε ablation χωρίς να αγγίξουμε τον υπόλοιπο κώδικα.
        self.rule3_strictness_scale = rule3_strictness_scale
        self.root_transition_target = root_transition_target  # [num_root_classes, num_root_classes], Κανόνας 4
        self.rule4_weight = rule4_weight
        self.enable_rule_4 = enable_rule_4
        self.rule5_leniency_factor = rule5_leniency_factor  # Κανόνας 5: αυστηρότητα σε resolving leading tones
        self.enable_rule_5 = enable_rule_5
        self.root_to_func_by_mode = root_to_func_by_mode  # [2, num_root_classes, num_function_classes], Κανόνας 6 & 7
        self.function_transition_target = function_transition_target  # [2, num_function_classes, num_function_classes] - ΞΕΧΩΡΙΣΤΟ ανά mode (0=minor, 1=major) μετά από εμπειρικό έλεγχο (check_rule6_by_mode.py)
        self.rule6_weight = rule6_weight
        self.enable_rule_6 = enable_rule_6
        self.rule7_end_weight = rule7_end_weight  # Κανόνας 7 (μόνο τέλος - η αρχή δεν είναι πλειοψηφική τάση)
        self.tonic_function_idx = tonic_function_idx
        self.enable_rule_7 = enable_rule_7
        self.quality_given_function_target = quality_given_function_target  # [num_function_classes, num_quality_classes], Κανόνας 8
        self.rule8_weight = rule8_weight
        self.enable_rule_8 = enable_rule_8
        # ΝΕΟ (v2): tonal-distance loss - βλ. σχόλιο στο build_centroid_table
        # παραπάνω. ΑΝΤΙΚΑΤΕΣΤΗΣΕ το προηγούμενο "απόλυτες νότες" head, που
        # δοκιμάστηκε και δεν έδειξε βελτίωση.
        self.centroid_table = centroid_table            # [num_root, num_qual, 6]
        self.centroid_mask_table = centroid_mask_table    # [num_root, num_qual] bool
        self.tonal_distance_weight = tonal_distance_weight
        self.enable_tonal_distance_loss = enable_tonal_distance_loss

    def forward(self, note_ids, pad_mask, beat_type_ids, step_resolved, resolving_lt, is_major,
                song_end, root_logits, root_targets, quality_logits, quality_targets):
        B, T = note_ids.shape

        # Joint πιθανότητα P(root, quality) = P(root) * P(quality) (γινόμενο, όχι
        # softmax του αθροίσματος των logits).
        root_probs = F.softmax(root_logits, dim=-1)
        qual_probs = F.softmax(quality_logits, dim=-1)
        combined_probs = root_probs[..., self.root_mapping] * qual_probs[..., self.qual_mapping]

        valid_mask = (~pad_mask).float()
        safe_note_ids = torch.clamp(note_ids, max=self.comp_matrix.shape[0] - 1)
        violation_mask = 1.0 - self.comp_matrix[safe_note_ids]

        # ΚΑΝΟΝΑΣ 1 + ΚΑΝΟΝΑΣ 2: leniency ανά θέση, πολλαπλασιάζεται πάνω στο βασικό
        # penalty. Επηρεάζει ΜΟΝΟ θέσεις όπου υπάρχει πραγματική παραβίαση (το
        # violation_mask είναι ήδη 0 στις έγκυρες θέσεις, άρα το leniency εκεί δεν
        # έχει καμία επίδραση).
        if self.enable_rules_1_2:
            beat_leniency = self.beat_type_scale[beat_type_ids]  # [B, T]
            step_leniency = torch.where(
                step_resolved > 0.5,
                torch.full_like(step_resolved, self.step_leniency_factor),
                torch.ones_like(step_resolved)
            )
            leniency = beat_leniency * step_leniency
        else:
            leniency = torch.ones_like(step_resolved)

        # ΚΑΝΟΝΑΣ 5: σε θέσεις όπου η νότα είναι ΚΑΙ resolving leading tone,
        # πολλαπλασιάζουμε επιπλέον με το strictness factor (>1 -> αυστηρότερο
        # έλεγχο, δηλαδή ΛΙΓΟΤΕΡΗ ανοχή σε παραβίαση εκεί - το αντίθετο από τους
        # Κανόνες 1&2, που χαλαρώνουν).
        if self.enable_rule_5:
            rule5_factor = torch.where(
                resolving_lt > 0.5,
                torch.full_like(resolving_lt, self.rule5_leniency_factor),
                torch.ones_like(resolving_lt)
            )
            leniency = leniency * rule5_factor

        per_pos_penalty = torch.sum(combined_probs * violation_mask, dim=-1) * valid_mask * leniency
        symbolic_penalty = per_pos_penalty.sum() / valid_mask.sum().clamp(min=1.0)

        # ΚΑΝΟΝΑΣ 3 (σταθερότητα αρμονικού ρυθμού): ΠΟΤΕ δεν τροφοδοτούμε την
        # προηγούμενη χορδή σαν INPUT (αυτό ήταν το πρόβλημα στα seq2seq πειράματα).
        # Αντί γι' αυτό, συγκρίνουμε τις predicted distributions δύο διαδοχικών
        # θέσεων (dot product = αναμενόμενη πιθανότητα να είναι "ίδια" χορδή κάτω
        # από το μοντέλο) με το πραγματικό εμπειρικό ποσοστό P(ίδια|beat_type).
        rule3_penalty = torch.zeros((), device=combined_probs.device)
        if self.enable_rule_3:
            prev_probs = torch.zeros_like(combined_probs)
            prev_probs[:, 1:, :] = combined_probs[:, :-1, :]
            predicted_same = torch.sum(combined_probs * prev_probs, dim=-1)  # [B, T]

            expected_same = self.expected_same_scale[beat_type_ids]  # [B, T]

            stability_mask = valid_mask.clone()
            stability_mask[:, 0] = 0.0  # η πρώτη θέση κάθε chunk δεν έχει έγκυρη προηγούμενη
            prev_valid = torch.zeros_like(valid_mask)
            prev_valid[:, 1:] = valid_mask[:, :-1]
            stability_mask = stability_mask * prev_valid

            # ΕΝΙΣΧΥΣΗ: πολλαπλασιαστική αυστηρότητα ανά beat_type (βλ. σχόλιο στην
            # κατασκευή του rule3_strictness_tensor) - τιμωρεί ΠΙΟ ΣΚΛΗΡΑ μια απόκλιση
            # στο Weak/Offbeat (όπου η σταθερότητα είναι σχεδόν ντετερμινιστική στα
            # δεδομένα) απ' ό,τι την ΙΔΙΑ απόκλιση στο Strong (όπου είναι γνήσια πιο
            # αβέβαιο). Αν δεν δόθηκε (None), 1.0 παντού -> ταυτόσημη συμπεριφορά με πριν.
            if self.rule3_strictness_scale is not None:
                strictness = self.rule3_strictness_scale[beat_type_ids]  # [B, T]
            else:
                strictness = torch.ones_like(predicted_same)

            rule3_per_pos = ((predicted_same - expected_same) ** 2) * strictness * stability_mask
            rule3_penalty = rule3_per_pos.sum() / stability_mask.sum().clamp(min=1.0)

            # ΔΙΑΓΝΩΣΤΙΚΟ (ΜΙΑ φορά): αποδεικνύει ΕΜΠΕΙΡΙΚΑ αν το strictness ΟΝΤΩΣ
            # ποικίλει ανά θέση σε πραγματικό runtime tensor, ΟΧΙ μόνο στη θεωρία.
            if not self._rule3_debug_done:
                unweighted_penalty = (((predicted_same - expected_same) ** 2) * stability_mask).sum() / stability_mask.sum().clamp(min=1.0)
                print(f"      [RULE3-DEBUG] strictness tensor (όλο): {self.rule3_strictness_scale.tolist()}")
                print(f"      [RULE3-DEBUG] strictness σε αυτό το batch: min={strictness.min().item():.4f} "
                      f"max={strictness.max().item():.4f} mean={strictness.mean().item():.4f}")
                print(f"      [RULE3-DEBUG] rule3_penalty ΜΕ strictness: {rule3_penalty.item():.6f}")
                print(f"      [RULE3-DEBUG] rule3_penalty ΧΩΡΙΣ strictness (σύγκριση): {unweighted_penalty.item():.6f}")
                self._rule3_debug_done = True

        # ΚΑΝΟΝΑΣ 4 (πιθανότητα κίνησης ρίζας): συγκρίνει τις predicted root
        # distributions δύο διαδοχικών θέσεων με τον εμπειρικό πίνακα μετάβασης
        # P(ρίζα_t | ρίζα_t-1) - bilinear μορφή, ίδιο πνεύμα με τον Κανόνα 3 αλλά
        # πάνω σε ΠΙΝΑΚΑ αντί για μονή τιμή στόχο. Ποτέ δεν βλέπει την πραγματική
        # προηγούμενη ρίζα σαν input.
        rule4_penalty = torch.zeros((), device=combined_probs.device)
        if self.enable_rule_4:
            prev_root_probs = torch.zeros_like(root_probs)
            prev_root_probs[:, 1:, :] = root_probs[:, :-1, :]
            transformed = torch.matmul(prev_root_probs, self.root_transition_target)  # [B, T, R]
            predicted_root_score = torch.sum(transformed * root_probs, dim=-1)  # [B, T]

            r4_mask = valid_mask.clone()
            r4_mask[:, 0] = 0.0
            prev_valid2 = torch.zeros_like(valid_mask)
            prev_valid2[:, 1:] = valid_mask[:, :-1]
            r4_mask = r4_mask * prev_valid2

            rule4_per_pos = (1.0 - predicted_root_score) * r4_mask
            rule4_penalty = rule4_per_pos.sum() / r4_mask.sum().clamp(min=1.0)

        # ΚΑΝΟΝΑΣ 6 (λειτουργική αρμονία T/S/D): "μεταφράζουμε" τις predicted root
        # distributions σε function distributions (mode-aware, ανά batch item - το
        # ΙΔΙΟ root_idx σημαίνει ΔΙΑΦΟΡΕΤΙΚΗ function σε major vs minor τραγούδι),
        # μετά ελέγχουμε bilinear plausibility ίδια λογική με τον Κανόνα 4.
        rule6_penalty = torch.zeros((), device=combined_probs.device)
        if self.enable_rule_6:
            root_to_func_selected = self.root_to_func_by_mode[is_major]  # [B, R, F]
            function_probs = torch.bmm(root_probs, root_to_func_selected)  # [B, T, F]

            prev_function_probs = torch.zeros_like(function_probs)
            prev_function_probs[:, 1:, :] = function_probs[:, :-1, :]
            # ΔΙΟΡΘΩΣΗ: το function_transition_target είναι ΤΩΡΑ [2,F,F] (ξεχωριστό
            # ανά mode) - επιλέγουμε τον ΣΩΣΤΟ πίνακα ΑΝΑ batch item (ΙΔΙΟ pattern με
            # το root_to_func_selected ακριβώς από πάνω), και χρησιμοποιούμε bmm
            # (per-item matrix) αντί για matmul με ΕΝΑ κοινό πίνακα.
            function_transition_selected = self.function_transition_target[is_major]  # [B, F, F]
            transformed_f = torch.bmm(prev_function_probs, function_transition_selected)  # [B, T, F]
            predicted_func_score = torch.sum(transformed_f * function_probs, dim=-1)  # [B, T]

            r6_mask = valid_mask.clone()
            r6_mask[:, 0] = 0.0
            prev_valid3 = torch.zeros_like(valid_mask)
            prev_valid3[:, 1:] = valid_mask[:, :-1]
            r6_mask = r6_mask * prev_valid3

            rule6_per_pos = (1.0 - predicted_func_score) * r6_mask
            rule6_penalty = rule6_per_pos.sum() / r6_mask.sum().clamp(min=1.0)

        # ΚΑΝΟΝΑΣ 7 (τονική στο τέλος): ΑΝΕΞΑΡΤΗΤΟΣ υπολογισμός από τον Κανόνα 6 (ίδιο
        # tensor root_to_func_by_mode, αλλά ξεχωριστό forward pass) - έμαθε το μάθημα
        # από τη σύγκρουση Κανόνα 4/6. Μόνο το τέλος - η αρχή δεν είναι πλειοψηφική τάση.
        rule7_end_penalty = torch.zeros((), device=combined_probs.device)
        if self.enable_rule_7:
            root_to_func_selected_7 = self.root_to_func_by_mode[is_major]  # [B, R, F]
            function_probs_7 = torch.bmm(root_probs, root_to_func_selected_7)  # [B, T, F]
            tonic_prob = function_probs_7[..., self.tonic_function_idx]  # [B, T]

            end_valid = song_end * valid_mask
            rule7_end_penalty = ((1.0 - tonic_prob) * end_valid).sum() / end_valid.sum().clamp(min=1.0)

        # ΚΑΝΟΝΑΣ 8 (ποιότητα χορδής ανάλογα με τη λειτουργία): ΔΙΑΦΟΡΕΤΙΚΟΣ τύπος
        # κανόνα - συγκρίνει τα ΔΥΟ heads ΣΤΗΝ ΙΔΙΑ θέση (όχι γειτονικές θέσεις στο
        # χρόνο όπως οι Κανόνες 3/4/6/7). Ισχύει σε ΚΑΘΕ έγκυρη θέση, όχι αραιά.
        rule8_penalty = torch.zeros((), device=combined_probs.device)
        if self.enable_rule_8:
            root_to_func_selected_8 = self.root_to_func_by_mode[is_major]  # [B, R, F]
            function_probs_8 = torch.bmm(root_probs, root_to_func_selected_8)  # [B, T, F]
            transformed_8 = torch.matmul(function_probs_8, self.quality_given_function_target)  # [B, T, Q]
            predicted_score_8 = torch.sum(transformed_8 * qual_probs, dim=-1)  # [B, T]

            rule8_per_pos = (1.0 - predicted_score_8) * valid_mask
            rule8_penalty = rule8_per_pos.sum() / valid_mask.sum().clamp(min=1.0)

        Rn, Qn = root_logits.shape[-1], quality_logits.shape[-1]
        root_loss = F.cross_entropy(root_logits.reshape(B * T, Rn), root_targets.reshape(B * T),
                                     weight=self.root_weights, ignore_index=-100, label_smoothing=0.1)
        quality_loss = F.cross_entropy(quality_logits.reshape(B * T, Qn), quality_targets.reshape(B * T),
                                        weight=self.quality_weights, ignore_index=-100, label_smoothing=0.1)

        # ΝΕΟ (v2): ΑΜΕΣΟ tonal-distance loss - υπολογίζει το ΑΝΑΜΕΝΟΜΕΝΟ σημείο
        # της ΠΡΑΓΜΑΤΙΚΗΣ πρόβλεψης του μοντέλου (όχι ξεχωριστό head) στον 6D
        # τονικό χώρο, μέσω εξωτερικού γινομένου των δύο softmax κατανομών, και
        # το συγκρίνει με το σημείο της αληθινής χορδής. Βλ. σχόλιο στο
        # build_centroid_table παραπάνω για γιατί αντικατέστησε το προηγούμενο.
        # ΙΔΙΑ προσοχή στο -100 padding index όπως πριν: clamp πριν το lookup,
        # αποκλεισμός μέσω μάσκας.
        tonal_distance_loss = torch.tensor(0.0, device=root_logits.device)
        if self.enable_tonal_distance_loss and self.centroid_table is not None:
            root_probs = F.softmax(root_logits, dim=-1)        # [B,T,Rn]
            quality_probs = F.softmax(quality_logits, dim=-1)  # [B,T,Qn]
            # εξωτερικό γινόμενο ανά θέση -> κοινή (ανεξάρτητη) κατανομή πάνω
            # σε ΚΑΘΕ (root,quality) συνδυασμό
            joint_probs = root_probs.unsqueeze(-1) * quality_probs.unsqueeze(-2)  # [B,T,Rn,Qn]
            expected_centroid = torch.einsum('btrq,rqc->btc', joint_probs, self.centroid_table)  # [B,T,6]

            safe_root_t = root_targets.clamp(min=0, max=self.centroid_table.shape[0] - 1)
            safe_qual_t = quality_targets.clamp(min=0, max=self.centroid_table.shape[1] - 1)
            true_centroid = self.centroid_table[safe_root_t, safe_qual_t]        # [B,T,6]
            tone_valid = self.centroid_mask_table[safe_root_t, safe_qual_t]      # [B,T] bool
            tone_valid = tone_valid & (~pad_mask)

            per_pos = ((expected_centroid - true_centroid) ** 2).sum(dim=-1)  # [B,T] squared distance
            denom = tone_valid.float().sum().clamp(min=1.0)
            tonal_distance_loss = (per_pos * tone_valid.float()).sum() / denom

        total_loss = (root_loss + quality_loss + (self.penalty_weight * symbolic_penalty)
                      + (self.rule3_weight * rule3_penalty) + (self.rule4_weight * rule4_penalty)
                      + (self.rule6_weight * rule6_penalty)
                      + (self.rule7_end_weight * rule7_end_penalty)
                      + (self.rule8_weight * rule8_penalty)
                      + (self.tonal_distance_weight * tonal_distance_loss))
        return total_loss, combined_probs


# --- 4. ΣΥΝΑΡΤΗΣΗ ΕΚΠΑΙΔΕΥΣΗΣ ---
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class SongDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def make_collate_fn(pad_note_id, pad_pitch_id, pad_duration_id, pad_beat_type_id, pad_position_id, pad_interval_id):
    def collate_batch(batch):
        (n_ids_list, pitch_list, duration_list, beat_type_list, position_list,
         interval_list, beat_strength_list, step_resolved_list, resolving_lt_list,
         is_major_list, song_end_list, root_list, quality_list) = zip(*batch)
        n_ids_padded = pad_sequence(n_ids_list, batch_first=True, padding_value=pad_note_id)
        pitch_padded = pad_sequence(pitch_list, batch_first=True, padding_value=pad_pitch_id)
        duration_padded = pad_sequence(duration_list, batch_first=True, padding_value=pad_duration_id)
        beat_type_padded = pad_sequence(beat_type_list, batch_first=True, padding_value=pad_beat_type_id)
        position_padded = pad_sequence(position_list, batch_first=True, padding_value=pad_position_id)
        interval_padded = pad_sequence(interval_list, batch_first=True, padding_value=pad_interval_id)
        beat_strength_padded = pad_sequence(beat_strength_list, batch_first=True, padding_value=0.0)
        step_resolved_padded = pad_sequence(step_resolved_list, batch_first=True, padding_value=1.0)
        resolving_lt_padded = pad_sequence(resolving_lt_list, batch_first=True, padding_value=0.0)
        is_major_batched = torch.stack(is_major_list)  # [B], ένα σταθερό mode ανά chunk, όχι padded ανά θέση
        song_end_padded = pad_sequence(song_end_list, batch_first=True, padding_value=0.0)
        root_padded = pad_sequence(root_list, batch_first=True, padding_value=-100)
        quality_padded = pad_sequence(quality_list, batch_first=True, padding_value=-100)
        pad_mask = (n_ids_padded == pad_note_id)
        return (n_ids_padded, pitch_padded, duration_padded, beat_type_padded, position_padded,
                interval_padded, beat_strength_padded, step_resolved_padded, resolving_lt_padded,
                is_major_batched, song_end_padded, root_padded, quality_padded, pad_mask)
    return collate_batch

def train_model(raw, chord_to_idx, note_to_idx, chord_root_idx, chord_quality_idx,
                 comp_matrix_np, num_note_classes, num_classes, num_root_classes, num_quality_classes,
                 batch_size=64, lr=5e-4, max_epochs=300, patience_limit=30, warmup_epochs=3,
                 enable_rules_1_2=True, enable_rule_3=True, enable_rule_4=True, enable_rule_5=True,
                 enable_rule_6=True, enable_rule_7=True, enable_rule_8=True,
                 enable_tonal_distance_loss=True):
    comp_matrix_tensor = torch.tensor(comp_matrix_np, dtype=torch.float32)
    PAD_NOTE_ID = num_note_classes
    PAD_PITCH_ID = NUM_PITCH_OCT_CLASSES
    PAD_DURATION_ID = NUM_DURATION_CLASSES
    PAD_BEAT_TYPE_ID = NUM_BEAT_TYPE_CLASSES
    PAD_POSITION_ID = NUM_POSITION_CLASSES
    PAD_INTERVAL_ID = NUM_INTERVAL_CLASSES

    # ΠΡΙΝ αποφασίσουμε το όριο μήκους, το ΜΕΤΡΑΜΕ (ίδια λογική με όλους τους
    # κανόνες) - θέλουμε ένα όριο αρκετά γενναιόδωρο ώστε ΣΧΕΔΟΝ ΚΑΝΕΝΑ τραγούδι
    # να μην κοπεί, όχι ένα αυθαίρετο νούμερο.
    song_lengths = sorted(len(melody_iris) for melody_iris, _ in raw)
    n_songs_total = len(song_lengths)
    p50 = song_lengths[n_songs_total // 2]
    p95 = song_lengths[int(n_songs_total * 0.95)]
    p99 = song_lengths[int(n_songs_total * 0.99)]
    max_len_seen = song_lengths[-1]
    # ΔΙΟΡΘΩΣΗ: το προηγούμενο όριο βασιζόταν στο ΑΠΟΛΥΤΟ max, το οποίο αποδείχθηκε
    # ένα ακραίο outlier (16623 νότες - 37x πάνω από το 99ό εκατοστημόριο!), σχεδόν
    # σίγουρα data artifact όχι πραγματικό τραγούδι. Αυτό ανάγκαζε ΚΑΘΕ batch που
    # (ατυχώς) το περιείχε να γεμίσει (pad) ΟΛΑ τα άλλα τραγούδια στο batch μέχρι
    # εκεί - αυτό προκάλεσε το OOM, όχι το whole-song training καθαυτό. Τώρα το
    # όριο βασίζεται στο 99ό εκατοστημόριο (με περιθώριο) - κόβει μόνο το πιο
    # ακραίο <=1% του corpus, επιτρέπει σε λογικό batch_size να δουλέψει ξανά.
    max_seq_len = max(64, int(p99 * 1.2))
    n_truncated = sum(1 for L in song_lengths if L > max_seq_len)
    print(f"   Μήκη τραγουδιών (νότες): median={p50}, 95ό εκατοστημόριο={p95}, 99ό={p99}, "
          f"max={max_len_seen} -> max_seq_len={max_seq_len} (θα κοπούν {n_truncated} τραγούδια, "
          f"δηλαδή {n_truncated/n_songs_total*100:.2f}% του corpus)")

    all_data = []
    for melody_iris, chord_iris in raw:
        n_ids = [note_to_idx[n] for n in melody_iris]
        pitch_ids = [pitch_oct_to_idx[n] for n in melody_iris]
        duration_ids = [duration_to_idx[n] for n in melody_iris]
        beat_type_ids = [beat_type_to_idx[n] for n in melody_iris]
        position_ids = [position_to_idx[n] for n in melody_iris]
        interval_ids = [interval_to_idx[n] for n in melody_iris]
        beat_strength_vals = [note_beat_strength[n] for n in melody_iris]
        root_y = [chord_root_idx[c] for c in chord_iris]
        quality_y = [chord_quality_idx[c] for c in chord_iris]

        # Κανόνας 2: "σωστά επιλυμένη" νότα = πλησιάστηκε Ή έφυγε με βήμα (<=2
        # semitones). Άγνωστο ύψος (parsing failure) -> ουδέτερο/lenient (1.0),
        # ώστε να μην τιμωρούμε λάθος κάτι που απλά δεν μπορέσαμε να διαβάσουμε.
        semis = [note_semitone[n] for n in melody_iris]
        step_resolved = [1.0] * len(semis)
        for t in range(len(semis)):
            if semis[t] is None:
                continue
            approached_ok = (t == 0) or (semis[t - 1] is None) or (abs(semis[t] - semis[t - 1]) <= 2)
            left_ok = (t == len(semis) - 1) or (semis[t + 1] is None) or (abs(semis[t + 1] - semis[t]) <= 2)
            step_resolved[t] = 1.0 if (approached_ok or left_ok) else 0.0

        # Κανόνας 5: "resolving leading tone" = η νότα ΕΙΝΑΙ leading tone (mode-aware)
        # ΚΑΙ η επόμενη νότα πράγματι φτάνει στην τονική.
        resolving_leading_tone = [0.0] * len(melody_iris)
        for t in range(len(melody_iris) - 1):
            n = melody_iris[t]
            if note_is_leading_tone.get(n):
                next_pc = note_pitch_class_transposed.get(melody_iris[t + 1])
                if next_pc == note_expected_tonic_pc.get(n):
                    resolving_leading_tone[t] = 1.0

        # Κανόνας 6: το mode είναι σταθερό ανά τραγούδι - το παίρνουμε από την πρώτη
        # χορδή του τραγουδιού.
        song_is_major = 1.0 if chord_is_major.get(chord_iris[0], True) else 0.0

        # Κανόνας 7: 1.0 ΜΟΝΟ στην τελευταία θέση ΟΛΟΚΛΗΡΟΥ του τραγουδιού.
        song_end_mask = [0.0] * len(melody_iris)
        if len(melody_iris) > 0:
            song_end_mask[-1] = 1.0

        # ΟΛΟΚΛΗΡΟ το τραγούδι σαν ΕΝΑ training example - όχι κομμένο σε chunks των
        # 64. Αυτό διορθώνει ΔΥΟ προβλήματα ταυτόχρονα: (1) το bidirectional
        # attention τώρα βλέπει ΠΡΑΓΜΑΤΙΚΑ όλο το τραγούδι, όχι ένα αυθαίρετο
        # παράθυρο, (2) το positional encoding δεν "νομίζει" πλέον ότι κάθε
        # εσωτερικό κομμάτι είναι η αρχή ενός τραγουδιού. Κόβουμε ΜΟΝΟ στα ελάχιστα
        # ακραία outliers (βλ. max_seq_len παραπάνω, μετρημένο από τα δεδομένα).
        n_ids_final = n_ids[:max_seq_len]
        if len(n_ids_final) >= 8:
            all_data.append((
                torch.tensor(n_ids_final, dtype=torch.long),
                torch.tensor(pitch_ids[:max_seq_len], dtype=torch.long),
                torch.tensor(duration_ids[:max_seq_len], dtype=torch.long),
                torch.tensor(beat_type_ids[:max_seq_len], dtype=torch.long),
                torch.tensor(position_ids[:max_seq_len], dtype=torch.long),
                torch.tensor(interval_ids[:max_seq_len], dtype=torch.long),
                torch.tensor(beat_strength_vals[:max_seq_len], dtype=torch.float32),
                torch.tensor(step_resolved[:max_seq_len], dtype=torch.float32),
                torch.tensor(resolving_leading_tone[:max_seq_len], dtype=torch.float32),
                torch.tensor(song_is_major, dtype=torch.long),
                torch.tensor(song_end_mask[:max_seq_len], dtype=torch.float32),
                torch.tensor(root_y[:max_seq_len], dtype=torch.long),
                torch.tensor(quality_y[:max_seq_len], dtype=torch.long),
            ))

    # Το shuffle/split είναι ΤΩΡΑ αυτόματα σε επίπεδο ΤΡΑΓΟΥΔΙΟΥ (αφού κάθε
    # στοιχείο του all_data ΕΙΝΑΙ ένα ολόκληρο τραγούδι) - διορθώνει και το
    # πρόβλημα διαρροής train/val που είχαμε με το chunking.
    random.seed(42)
    random.shuffle(all_data)
    split_idx = int(0.8 * len(all_data))
    train_data, val_data = all_data[:split_idx], all_data[split_idx:]
    print(f"📦 {len(train_data)} τραγούδια εκπαίδευσης, {len(val_data)} τραγούδια επικύρωσης "
          f"(ολόκληρα, max_seq_len={max_seq_len})")

    all_root_indices = [chord_root_idx[c] for _, chord_labels in raw for c in chord_labels]
    all_quality_indices = [chord_quality_idx[c] for _, chord_labels in raw for c in chord_labels]

    def make_class_weights(indices, n_classes, clip_max=10.0):
        counts = Counter(indices)
        total = len(indices)
        w = [0.0] * n_classes
        for i in range(n_classes):
            count = counts.get(i, 0)
            raw_w = total / (n_classes * count) if count > 0 else 1.0
            w[i] = math.sqrt(raw_w)
        w = np.clip(w, 0.1, clip_max)
        return torch.tensor(w, dtype=torch.float32).to(device), counts

    # ΥΠΟΘΕΣΗ: με λιγότερες κλάσεις (254 -> 24 μετά το vocabulary cleanup), ο ίδιος
    # τύπος βάρους (1/n_classes στον παρονομαστή) δίνει ΠΙΟ ακραία βάρη σε σπάνιες
    # (ΣΧΕΤΙΚΑ, όχι απόλυτα) κλάσεις - πιθανή αιτία της πτώσης στο quality accuracy
    # (47%->38%) παρόλο που το λεξιλόγιο έγινε "πιο εύκολο". Σφίγγουμε το clip_max
    # για το quality head, ΚΑΙ τυπώνουμε τα πραγματικά βάρη ώστε να το επιβεβαιώσουμε.
    root_weights_tensor, root_counts = make_class_weights(all_root_indices, num_root_classes, clip_max=10.0)
    quality_weights_tensor, quality_counts_dbg = make_class_weights(all_quality_indices, num_quality_classes, clip_max=5.0)

    print("   Βάρη κλάσεων quality (idx: count -> weight), ταξινομημένα κατά βάρος:")
    idx_to_quality = {idx: q for q, idx in quality_to_idx.items()}
    for i in sorted(range(num_quality_classes), key=lambda i: -quality_weights_tensor[i].item()):
        q_name = idx_to_quality.get(i, '?')
        print(f"      idx {i:2d} ({q_name:8}): count={quality_counts_dbg.get(i, 0):>8}  weight={quality_weights_tensor[i].item():.2f}")

    # ΑΝΑΠΑΡΑΓΩΓΙΜΟΤΗΤΑ: μέχρι τώρα μόνο το random.seed(42) για το data split ήταν
    # σταθερό - η ΑΡΧΙΚΟΠΟΙΗΣΗ των βαρών του μοντέλου (nn.Embedding, nn.Linear,
    # εσωτερικά του XTEncoder) ΔΕΝ ήταν seeded, άρα κάθε run ξεκινούσε από
    # διαφορετικά τυχαία βάρη - πιθανή εξήγηση για το γιατί δύο runs με τον ΙΔΙΟ
    # κώδικα έδειξαν διαφορετική πορεία σύγκλισης. torch.manual_seed() ΔΕΝ εγγυάται
    # απόλυτη, bit-for-bit αναπαραγωγιμότητα σε GPU (ορισμένοι πυρήνες CUDA
    # παραμένουν μη-ντετερμινιστικοί εξ ορισμού), αλλά καθιστά την αρχικοποίηση
    # βαρών συνεπή, που είναι ήδη μεγάλο βήμα προς πιο συγκρίσιμα πειράματα.
    MODEL_SEED = int(os.environ.get('MODEL_SEED', 1))
    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)

    model = HarmonizerTransformer(
        num_pitch_classes=NUM_PITCH_OCT_CLASSES, pad_pitch_id=PAD_PITCH_ID,
        num_duration_classes=NUM_DURATION_CLASSES, pad_duration_id=PAD_DURATION_ID,
        num_beat_type_classes=NUM_BEAT_TYPE_CLASSES, pad_beat_type_id=PAD_BEAT_TYPE_ID,
        num_position_classes=NUM_POSITION_CLASSES, pad_position_id=PAD_POSITION_ID,
        num_interval_classes=NUM_INTERVAL_CLASSES, pad_interval_id=PAD_INTERVAL_ID,
        embed_dim=EMBEDDING_DIM, num_heads=8, num_layers=4, num_root_classes=num_root_classes,
        num_quality_classes=num_quality_classes, dropout=0.3
    ).to(device)

    # ΝΕΟ (v2): πίνακας κεντροειδών για το tonal-distance loss - ΜΙΑ φορά, πριν
    # ξεκινήσει η εκπαίδευση (όχι ανά batch - θα ήταν σπατάλη υπολογισμού).
    centroid_table, centroid_mask_table = build_centroid_table(
        idx_to_root_pc, idx_to_quality, num_root_classes, num_quality_classes
    )
    centroid_table = centroid_table.to(device)
    centroid_mask_table = centroid_mask_table.to(device)

    # ΝΕΟ: πίνακας για ΖΩΝΤΑΝΗ μέτρηση 'κοντινής' ακρίβειας δίπλα στο Val Acc -
    # ΚΑΘΑΡΑ μέτρηση, ΔΕΝ συμμετέχει στο loss/gradient.
    tone_vector_table, tone_vector_mask_table = build_tone_vector_table(
        idx_to_root_pc, idx_to_quality, num_root_classes, num_quality_classes
    )
    tone_vector_table = tone_vector_table.to(device)
    tone_vector_mask_table = tone_vector_mask_table.to(device)

    collate_fn = make_collate_fn(PAD_NOTE_ID, PAD_PITCH_ID, PAD_DURATION_ID, PAD_BEAT_TYPE_ID,
                                  PAD_POSITION_ID, PAD_INTERVAL_ID)
    train_loader = DataLoader(SongDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(SongDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"   Κανόνες 1 & 2 (μετρικό βάρος + step-resolution): {'ΕΝΕΡΓΟΙ' if enable_rules_1_2 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΙ (ablation)'}")
    print(f"   Κανόνας 3 (σταθερότητα αρμονικού ρυθμού): {'ΕΝΕΡΓΟΣ' if enable_rule_3 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ (ablation)'} (βάρος={RULE3_WEIGHT:.2f}, από τα δεδομένα)")
    print(f"   Κανόνας 4 (κίνηση ρίζας): {'ΕΝΕΡΓΟΣ' if enable_rule_4 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ (ablation)'} (βάρος={RULE4_WEIGHT:.2f}, από τα δεδομένα)")
    print(f"   Κανόνας 5 (επίλυση leading tone): {'ΕΝΕΡΓΟΣ' if enable_rule_5 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ (ablation)'} (strictness={RULE5_LENIENCY_FACTOR:.2f}, από τα δεδομένα)")
    print(f"   Κανόνας 6 (λειτουργική αρμονία T/S/D): {'ΕΝΕΡΓΟΣ' if enable_rule_6 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ (ablation)'} (βάρος={RULE6_WEIGHT:.2f}, από τα δεδομένα)")
    print(f"   Κανόνας 7 (τονική στο τέλος): {'ΕΝΕΡΓΟΣ' if enable_rule_7 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ (ablation)'} (τέλος={RULE7_END_WEIGHT:.2f}, από τα δεδομένα)")
    print(f"   Κανόνας 8 (ποιότητα ανά λειτουργία): {'ΕΝΕΡΓΟΣ' if enable_rule_8 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ (ablation)'} (βάρος={RULE8_WEIGHT:.2f}, από τα δεδομένα)")
    print(f"   Tonal-distance loss (Harte κεντροειδές): {'ΕΝΕΡΓΟ' if enable_tonal_distance_loss else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟ (ablation)'} (βάρος={TONAL_DISTANCE_WEIGHT:.2f}, σταθερό)")

    criterion = RuleInformedLoss(
        penalty_weight=PENALTY_WEIGHT, rule_matrix=comp_matrix_tensor,
        root_weights=root_weights_tensor, quality_weights=quality_weights_tensor,
        root_mapping=chord_root_mapping, qual_mapping=chord_qual_mapping,
        beat_type_scale=beat_type_scale_tensor, step_leniency_factor=0.3,
        enable_rules_1_2=enable_rules_1_2,
        expected_same_scale=expected_same_tensor, rule3_weight=RULE3_WEIGHT, enable_rule_3=enable_rule_3,
        rule3_strictness_scale=rule3_strictness_tensor,
        root_transition_target=root_transition_target, rule4_weight=RULE4_WEIGHT, enable_rule_4=enable_rule_4,
        rule5_leniency_factor=RULE5_LENIENCY_FACTOR, enable_rule_5=enable_rule_5,
        root_to_func_by_mode=root_to_func_by_mode, function_transition_target=function_transition_target_by_mode,
        rule6_weight=RULE6_WEIGHT, enable_rule_6=enable_rule_6,
        rule7_end_weight=RULE7_END_WEIGHT,
        tonic_function_idx=FUNCTION_TO_IDX['T'], enable_rule_7=enable_rule_7,
        quality_given_function_target=quality_given_function_target, rule8_weight=RULE8_WEIGHT,
        enable_rule_8=enable_rule_8,
        centroid_table=centroid_table,
        centroid_mask_table=centroid_mask_table,
        tonal_distance_weight=TONAL_DISTANCE_WEIGHT, enable_tonal_distance_loss=enable_tonal_distance_loss
    )

    # AdamW: σωστό decoupled weight decay (το plain Adam εφαρμόζει weight decay
    # λάθος, μπλέκοντάς το με τα gradient moving averages) - καθαρή βελτίωση χωρίς
    # μειονέκτημα. Weight decay ανέβηκε ελαφρά (1e-4 -> 2e-4) λόγω του ήπιου
    # overfitting gap που φαινόταν στο προηγούμενο run (train loss ξέφευγε κάτω
    # από το val loss προς το τέλος).
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=16)
    # Γραμμικό warmup: οι πρώτες λίγες εποχές ξεκινούν από πολύ χαμηλό LR και
    # ανεβαίνουν σταδιακά στο πλήρες lr, πριν ενεργοποιηθεί το ReduceLROnPlateau.
    # Transformers είναι γνωστό ότι μπορούν να αποσταθεροποιηθούν με πλήρες LR από
    # την αρχή, πριν προλάβουν να "ζεσταθούν" τα attention layers.

    patience_counter = 0
    best_val_loss = float('inf')
    best_model_path = os.path.join(OUT_DIR, "best_transformer_model.pth")
    # Ξεχωριστό tracking ΜΟΝΟ για αναφορά - ΔΕΝ επηρεάζει ποιο checkpoint σώζεται/
    # επιστρέφεται (αυτό παραμένει βάσει val loss, η πιο σωστή επιλογή ώστε να μην
    # "διαλέγουμε" αυθαίρετα το καλύτερο-ακρίβεια epoch μετά τα γεγονότα). Απλά μας
    # δείχνει πόσο διαφέρει η επιλεγμένη ακρίβεια από την ΚΑΛΥΤΕΡΗ που φάνηκε ποτέ.
    peak_val_acc, peak_val_acc_epoch = 0.0, -1

    print("\nΞεκινάει η εκπαίδευση (bidirectional encoder, εμπλουτισμένα features)...")
    for epoch in range(max_epochs):
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for g in optimizer.param_groups:
                g['lr'] = warmup_lr

        model.train()
        train_loss = 0.0
        n_clamped_this_epoch = 0  # πόσες τιμές .g χτύπησαν το όριο [-10,10] αυτή την εποχή
        for (n_ids, pitch_ids, duration_ids, beat_type_ids, position_ids, interval_ids,
             beat_strength, step_resolved, resolving_lt, is_major, song_end,
             root_y, quality_y, pad_mask) in train_loader:
            n_ids, pad_mask = n_ids.to(device), pad_mask.to(device)
            pitch_ids, duration_ids = pitch_ids.to(device), duration_ids.to(device)
            beat_type_ids, position_ids = beat_type_ids.to(device), position_ids.to(device)
            interval_ids, beat_strength = interval_ids.to(device), beat_strength.to(device)
            step_resolved = step_resolved.to(device)
            resolving_lt = resolving_lt.to(device)
            is_major = is_major.to(device)
            song_end = song_end.to(device)
            root_y, quality_y = root_y.to(device), quality_y.to(device)

            optimizer.zero_grad()
            root_out, quality_out = model(pitch_ids, duration_ids, beat_type_ids, position_ids,
                                           interval_ids, beat_strength, pad_mask)
            loss, _ = criterion(n_ids, pad_mask, beat_type_ids, step_resolved, resolving_lt, is_major,
                                 song_end, root_out, root_y, quality_out, quality_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # ΔΙΟΡΘΩΣΗ μετά το Διαγνωστικό 3: τα RMSNorm ".g" (πριν, ScaleNorm
            # ".gamma") βρέθηκαν να "εκρήγνυνται" σε τιμές ~10^7-10^8 μετά από πολλές
            # εποχές - ΑΝΕΞΑΡΤΗΤΑ από το ποιο norm variant χρησιμοποιείται (το ίδιο
            # πρόβλημα εμφανίστηκε ΚΑΙ στα δύο, άρα δεν είναι κάτι ειδικό στο
            # ScaleNorm όπως πιστεύαμε αρχικά). Το gradient clipping δεσμεύει μόνο
            # το ΜΕΓΕΘΟΣ κάθε μεμονωμένου βήματος, όχι τη ΣΩΡΕΥΤΙΚΗ κίνηση σε
            # εκατοντάδες εποχές - αν η κατεύθυνση του gradient παραμένει σταθερή
            # για πολύ καιρό, η παράμετρος μπορεί ακόμα να "περπατήσει" σε παράλογες
            # τιμές. Εδώ περιορίζουμε ΑΜΕΣΑ αυτές τις παραμέτρους μετά από κάθε
            # βήμα - εγγυημένη λύση, ανεξάρτητα από το τι ακριβώς προκαλεί την
            # ανοδική πίεση.
            with torch.no_grad():
                for param_name, param in model.named_parameters():
                    if param_name.endswith('.g'):
                        n_clamped_this_epoch += (param.abs() > 10.0).sum().item()
                        param.clamp_(-10.0, 10.0)
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct, root_correct, qual_correct, total = 0, 0, 0, 0
        near1_correct, near2_correct, near_total = 0, 0, 0  # ΝΕΟ: 'κοντινή' ακρίβεια
        qual_pred_counts = Counter()
        with torch.no_grad():
            for (n_ids, pitch_ids, duration_ids, beat_type_ids, position_ids, interval_ids,
                 beat_strength, step_resolved, resolving_lt, is_major, song_end,
                 root_y, quality_y, pad_mask) in val_loader:
                n_ids, pad_mask = n_ids.to(device), pad_mask.to(device)
                pitch_ids, duration_ids = pitch_ids.to(device), duration_ids.to(device)
                beat_type_ids, position_ids = beat_type_ids.to(device), position_ids.to(device)
                interval_ids, beat_strength = interval_ids.to(device), beat_strength.to(device)
                step_resolved = step_resolved.to(device)
                resolving_lt = resolving_lt.to(device)
                is_major = is_major.to(device)
                song_end = song_end.to(device)
                root_y, quality_y = root_y.to(device), quality_y.to(device)
                root_out, quality_out = model(pitch_ids, duration_ids, beat_type_ids, position_ids,
                                               interval_ids, beat_strength, pad_mask)
                loss, _ = criterion(n_ids, pad_mask, beat_type_ids, step_resolved, resolving_lt, is_major,
                                     song_end, root_out, root_y, quality_out, quality_y)
                val_loss += loss.item()

                root_preds = root_out.argmax(dim=-1)
                qual_preds = quality_out.argmax(dim=-1)
                valid = (~pad_mask)
                root_correct += ((root_preds == root_y) & valid).sum().item()
                qual_correct += ((qual_preds == quality_y) & valid).sum().item()
                correct += (((root_preds == root_y) & (qual_preds == quality_y)) & valid).sum().item()
                total += valid.sum().item()
                qual_pred_counts.update(qual_preds[valid].tolist())

                # ΝΕΟ: 'κοντινή' ακρίβεια - ΚΑΘΑΡΑ μέτρηση, όχι μέρος του loss.
                # root_preds/qual_preds είναι ΠΑΝΤΑ έγκυροι δείκτες (από argmax,
                # ποτέ -100) - ΜΟΝΟ τα root_y/quality_y χρειάζονται clamp λόγω
                # του ignore_index στα padded θέσεις.
                safe_root_y = root_y.clamp(min=0, max=tone_vector_table.shape[0] - 1)
                safe_qual_y = quality_y.clamp(min=0, max=tone_vector_table.shape[1] - 1)
                pred_tones = tone_vector_table[root_preds, qual_preds]      # [B,T,12]
                true_tones = tone_vector_table[safe_root_y, safe_qual_y]    # [B,T,12]
                pred_tones_valid = tone_vector_mask_table[root_preds, qual_preds]
                true_tones_valid = tone_vector_mask_table[safe_root_y, safe_qual_y]
                near_valid = pred_tones_valid & true_tones_valid & valid  # αγνοεί OTHER + padding
                distance = (pred_tones != true_tones).sum(dim=-1)  # Hamming, [B,T]
                near1_correct += ((distance <= 1) & near_valid).sum().item()
                near2_correct += ((distance <= 2) & near_valid).sum().item()
                near_total += near_valid.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = (correct / total * 100) if total > 0 else 0.0
        root_acc = (root_correct / total * 100) if total > 0 else 0.0
        qual_acc = (qual_correct / total * 100) if total > 0 else 0.0
        near1_acc = (near1_correct / near_total * 100) if near_total > 0 else 0.0
        near2_acc = (near2_correct / near_total * 100) if near_total > 0 else 0.0
        if val_acc > peak_val_acc:
            peak_val_acc, peak_val_acc_epoch = val_acc, epoch
        if epoch >= warmup_epochs:
            scheduler.step(avg_val_loss)

        n_unique_pred = len(qual_pred_counts)
        top_pred = qual_pred_counts.most_common(3)
        pred_summary = ", ".join(f"cls{c}:{n/total*100:.1f}%" for c, n in top_pred)
        print(f"   [diag] quality: {n_unique_pred} διαφορετικές κλάσεις προβλέφθηκαν | top: {pred_summary}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            status_msg = "⭐ Νέο ρεκόρ Val Loss!"
        else:
            patience_counter += 1
            status_msg = f"⚠️ Καμία βελτίωση ({patience_counter}/{patience_limit})"

        clamp_note = f" | .g clamped: {n_clamped_this_epoch}x" if n_clamped_this_epoch > 0 else ""
        # ΝΕΟ, ΠΑΝΤΑ ενεργό διαγνωστικό: τυπώνουμε την ΠΡΑΓΜΑΤΙΚΗ τιμή (όχι μόνο
        # μέτρημα ενεργοποίησης) της .g παραμέτρου ΑΠΕΥΘΕΙΑΣ από το live model object,
        # ΑΜΕΣΩΣ μετά το clamp. Αν αυτό δείξει σταθερά <=10 σε ΚΑΘΕ εποχή αλλά το
        # ΤΕΛΙΚΟ αποθηκευμένο checkpoint δείχνει εκατομμύρια (Διαγνωστικό 3), αυτό θα
        # αποδείξει ότι το πρόβλημα ΔΕΝ είναι στο ίδιο το clamp - θα είναι κάπου
        # αλλού (π.χ. πιθανή εσωτερική αναπαραμετροποίηση του RMSNorm στο
        # x-transformers, όπου το named_parameters() ΔΕΝ επιστρέφει το ΙΔΙΟ tensor
        # που χρησιμοποιείται στην πραγματικότητα στον υπολογισμό forward()).
        g_debug = ""
        for param_name, param in model.named_parameters():
            if param_name.endswith('.g'):
                g_debug = f" | live .g max|.|: {param.abs().max().item():.4f}"
                break
        print(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% (Root: {root_acc:.1f}%, Qual: {qual_acc:.1f}%) | "
              f"Near(≤1/≤2 νότες): {near1_acc:.1f}%/{near2_acc:.1f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | {status_msg}{clamp_note}{g_debug}")

        if patience_counter >= patience_limit:
            print(f"\n🛑 EARLY STOPPING στην Εποχή {epoch}!")
            model.load_state_dict(torch.load(best_model_path))
            print(f"   ℹ️ Το επιστρεφόμενο checkpoint επιλέχθηκε βάσει val loss. Η ΚΑΛΥΤΕΡΗ ακρίβεια "
                  f"που φάνηκε σε ΟΠΟΙΟΔΗΠΟΤΕ epoch ήταν {peak_val_acc:.2f}% (εποχή {peak_val_acc_epoch}) - "
                  f"δες αυτό το κενό πριν συγκρίνεις ρυθμίσεις πολύ κοντά μεταξύ τους.")
            break
    else:
        # Το loop έφτασε στο max_epochs ΧΩΡΙΣ να ενεργοποιηθεί early stopping - πρέπει
        # ΚΑΙ ΕΔΩ να φορτωθεί το καλύτερο (βάσει val loss) checkpoint, αλλιώς η
        # συνάρτηση επιστρέφει τα βάρη της ΤΕΛΕΥΤΑΙΑΣ εποχής, όχι του καλύτερου.
        print(f"\n📦 Ολοκληρώθηκαν όλες οι {max_epochs} εποχές χωρίς early stopping.")
        model.load_state_dict(torch.load(best_model_path))
        print(f"   ℹ️ Η ΚΑΛΥΤΕΡΗ ακρίβεια που φάνηκε σε ΟΠΟΙΟΔΗΠΟΤΕ epoch ήταν {peak_val_acc:.2f}% "
              f"(εποχή {peak_val_acc_epoch}).")

    return model


def train_model_iterative(raw, chord_to_idx, note_to_idx, chord_root_idx, chord_quality_idx,
                 comp_matrix_np, num_note_classes, num_classes, num_root_classes, num_quality_classes,
                 num_chord_classes, flat_key_to_idx_arg,
                 batch_size=64, lr=5e-4, max_epochs=300, patience_limit=30, warmup_epochs=3,
                 enable_rules_1_2=True, enable_rule_3=True, enable_rule_4=True, enable_rule_5=True,
                 enable_rule_6=True, enable_rule_7=True, enable_rule_8=True,
                 enable_tonal_distance_loss=True, curriculum_total_epochs=80):
    # ΣΗΜΑΝΤΙΚΟ: curriculum_total_epochs (default 80) ΕΙΝΑΙ ΣΚΟΠΙΜΑ ΞΕΧΩΡΙΣΤΟ από
    # το max_epochs (300, σχεδόν ΠΟΤΕ δεν φτάνουμε εκεί λόγω early stopping - ΟΛΑ
    # τα runs μέχρι τώρα σταμάτησαν στις 50-80 εποχές). Αν το curriculum βάσιζε
    # την πρόοδό του στο max_epochs=300, το reveal_fraction θα ήταν ΣΧΕΔΟΝ ΜΗΔΕΝ
    # (~0.001) ακόμα και στην εποχή 80 - το μοντέλο ΣΧΕΔΟΝ ΠΟΤΕ δεν θα έβλεπε
    # αποκαλυμμένη χορδή πριν σταματήσει η εκπαίδευση. Το curriculum_total_epochs
    # βασίζεται σε ΠΡΑΓΜΑΤΙΚΗ, παρατηρημένη διάρκεια, ΟΧΙ στο θεωρητικό όριο.
    comp_matrix_tensor = torch.tensor(comp_matrix_np, dtype=torch.float32)
    PAD_NOTE_ID = num_note_classes
    PAD_PITCH_ID = NUM_PITCH_OCT_CLASSES
    PAD_DURATION_ID = NUM_DURATION_CLASSES
    PAD_BEAT_TYPE_ID = NUM_BEAT_TYPE_CLASSES
    PAD_POSITION_ID = NUM_POSITION_CLASSES
    PAD_INTERVAL_ID = NUM_INTERVAL_CLASSES

    # ΠΡΙΝ αποφασίσουμε το όριο μήκους, το ΜΕΤΡΑΜΕ (ίδια λογική με όλους τους
    # κανόνες) - θέλουμε ένα όριο αρκετά γενναιόδωρο ώστε ΣΧΕΔΟΝ ΚΑΝΕΝΑ τραγούδι
    # να μην κοπεί, όχι ένα αυθαίρετο νούμερο.
    song_lengths = sorted(len(melody_iris) for melody_iris, _ in raw)
    n_songs_total = len(song_lengths)
    p50 = song_lengths[n_songs_total // 2]
    p95 = song_lengths[int(n_songs_total * 0.95)]
    p99 = song_lengths[int(n_songs_total * 0.99)]
    max_len_seen = song_lengths[-1]
    # ΔΙΟΡΘΩΣΗ: το προηγούμενο όριο βασιζόταν στο ΑΠΟΛΥΤΟ max, το οποίο αποδείχθηκε
    # ένα ακραίο outlier (16623 νότες - 37x πάνω από το 99ό εκατοστημόριο!), σχεδόν
    # σίγουρα data artifact όχι πραγματικό τραγούδι. Αυτό ανάγκαζε ΚΑΘΕ batch που
    # (ατυχώς) το περιείχε να γεμίσει (pad) ΟΛΑ τα άλλα τραγούδια στο batch μέχρι
    # εκεί - αυτό προκάλεσε το OOM, όχι το whole-song training καθαυτό. Τώρα το
    # όριο βασίζεται στο 99ό εκατοστημόριο (με περιθώριο) - κόβει μόνο το πιο
    # ακραίο <=1% του corpus, επιτρέπει σε λογικό batch_size να δουλέψει ξανά.
    max_seq_len = max(64, int(p99 * 1.2))
    n_truncated = sum(1 for L in song_lengths if L > max_seq_len)
    print(f"   Μήκη τραγουδιών (νότες): median={p50}, 95ό εκατοστημόριο={p95}, 99ό={p99}, "
          f"max={max_len_seen} -> max_seq_len={max_seq_len} (θα κοπούν {n_truncated} τραγούδια, "
          f"δηλαδή {n_truncated/n_songs_total*100:.2f}% του corpus)")

    all_data = []
    for melody_iris, chord_iris in raw:
        n_ids = [note_to_idx[n] for n in melody_iris]
        pitch_ids = [pitch_oct_to_idx[n] for n in melody_iris]
        duration_ids = [duration_to_idx[n] for n in melody_iris]
        beat_type_ids = [beat_type_to_idx[n] for n in melody_iris]
        position_ids = [position_to_idx[n] for n in melody_iris]
        interval_ids = [interval_to_idx[n] for n in melody_iris]
        beat_strength_vals = [note_beat_strength[n] for n in melody_iris]
        root_y = [chord_root_idx[c] for c in chord_iris]
        quality_y = [chord_quality_idx[c] for c in chord_iris]

        # Κανόνας 2: "σωστά επιλυμένη" νότα = πλησιάστηκε Ή έφυγε με βήμα (<=2
        # semitones). Άγνωστο ύψος (parsing failure) -> ουδέτερο/lenient (1.0),
        # ώστε να μην τιμωρούμε λάθος κάτι που απλά δεν μπορέσαμε να διαβάσουμε.
        semis = [note_semitone[n] for n in melody_iris]
        step_resolved = [1.0] * len(semis)
        for t in range(len(semis)):
            if semis[t] is None:
                continue
            approached_ok = (t == 0) or (semis[t - 1] is None) or (abs(semis[t] - semis[t - 1]) <= 2)
            left_ok = (t == len(semis) - 1) or (semis[t + 1] is None) or (abs(semis[t + 1] - semis[t]) <= 2)
            step_resolved[t] = 1.0 if (approached_ok or left_ok) else 0.0

        # Κανόνας 5: "resolving leading tone" = η νότα ΕΙΝΑΙ leading tone (mode-aware)
        # ΚΑΙ η επόμενη νότα πράγματι φτάνει στην τονική.
        resolving_leading_tone = [0.0] * len(melody_iris)
        for t in range(len(melody_iris) - 1):
            n = melody_iris[t]
            if note_is_leading_tone.get(n):
                next_pc = note_pitch_class_transposed.get(melody_iris[t + 1])
                if next_pc == note_expected_tonic_pc.get(n):
                    resolving_leading_tone[t] = 1.0

        # Κανόνας 6: το mode είναι σταθερό ανά τραγούδι - το παίρνουμε από την πρώτη
        # χορδή του τραγουδιού.
        song_is_major = 1.0 if chord_is_major.get(chord_iris[0], True) else 0.0

        # Κανόνας 7: 1.0 ΜΟΝΟ στην τελευταία θέση ΟΛΟΚΛΗΡΟΥ του τραγουδιού.
        song_end_mask = [0.0] * len(melody_iris)
        if len(melody_iris) > 0:
            song_end_mask[-1] = 1.0

        # ΟΛΟΚΛΗΡΟ το τραγούδι σαν ΕΝΑ training example - όχι κομμένο σε chunks των
        # 64. Αυτό διορθώνει ΔΥΟ προβλήματα ταυτόχρονα: (1) το bidirectional
        # attention τώρα βλέπει ΠΡΑΓΜΑΤΙΚΑ όλο το τραγούδι, όχι ένα αυθαίρετο
        # παράθυρο, (2) το positional encoding δεν "νομίζει" πλέον ότι κάθε
        # εσωτερικό κομμάτι είναι η αρχή ενός τραγουδιού. Κόβουμε ΜΟΝΟ στα ελάχιστα
        # ακραία outliers (βλ. max_seq_len παραπάνω, μετρημένο από τα δεδομένα).
        n_ids_final = n_ids[:max_seq_len]
        if len(n_ids_final) >= 8:
            all_data.append((
                torch.tensor(n_ids_final, dtype=torch.long),
                torch.tensor(pitch_ids[:max_seq_len], dtype=torch.long),
                torch.tensor(duration_ids[:max_seq_len], dtype=torch.long),
                torch.tensor(beat_type_ids[:max_seq_len], dtype=torch.long),
                torch.tensor(position_ids[:max_seq_len], dtype=torch.long),
                torch.tensor(interval_ids[:max_seq_len], dtype=torch.long),
                torch.tensor(beat_strength_vals[:max_seq_len], dtype=torch.float32),
                torch.tensor(step_resolved[:max_seq_len], dtype=torch.float32),
                torch.tensor(resolving_leading_tone[:max_seq_len], dtype=torch.float32),
                torch.tensor(song_is_major, dtype=torch.long),
                torch.tensor(song_end_mask[:max_seq_len], dtype=torch.float32),
                torch.tensor(root_y[:max_seq_len], dtype=torch.long),
                torch.tensor(quality_y[:max_seq_len], dtype=torch.long),
            ))

    # Το shuffle/split είναι ΤΩΡΑ αυτόματα σε επίπεδο ΤΡΑΓΟΥΔΙΟΥ (αφού κάθε
    # στοιχείο του all_data ΕΙΝΑΙ ένα ολόκληρο τραγούδι) - διορθώνει και το
    # πρόβλημα διαρροής train/val που είχαμε με το chunking.
    random.seed(42)
    random.shuffle(all_data)
    split_idx = int(0.8 * len(all_data))
    train_data, val_data = all_data[:split_idx], all_data[split_idx:]
    print(f"📦 {len(train_data)} τραγούδια εκπαίδευσης, {len(val_data)} τραγούδια επικύρωσης "
          f"(ολόκληρα, max_seq_len={max_seq_len})")

    all_root_indices = [chord_root_idx[c] for _, chord_labels in raw for c in chord_labels]
    all_quality_indices = [chord_quality_idx[c] for _, chord_labels in raw for c in chord_labels]

    def make_class_weights(indices, n_classes, clip_max=10.0):
        counts = Counter(indices)
        total = len(indices)
        w = [0.0] * n_classes
        for i in range(n_classes):
            count = counts.get(i, 0)
            raw_w = total / (n_classes * count) if count > 0 else 1.0
            w[i] = math.sqrt(raw_w)
        w = np.clip(w, 0.1, clip_max)
        return torch.tensor(w, dtype=torch.float32).to(device), counts

    # ΥΠΟΘΕΣΗ: με λιγότερες κλάσεις (254 -> 24 μετά το vocabulary cleanup), ο ίδιος
    # τύπος βάρους (1/n_classes στον παρονομαστή) δίνει ΠΙΟ ακραία βάρη σε σπάνιες
    # (ΣΧΕΤΙΚΑ, όχι απόλυτα) κλάσεις - πιθανή αιτία της πτώσης στο quality accuracy
    # (47%->38%) παρόλο που το λεξιλόγιο έγινε "πιο εύκολο". Σφίγγουμε το clip_max
    # για το quality head, ΚΑΙ τυπώνουμε τα πραγματικά βάρη ώστε να το επιβεβαιώσουμε.
    root_weights_tensor, root_counts = make_class_weights(all_root_indices, num_root_classes, clip_max=10.0)
    quality_weights_tensor, quality_counts_dbg = make_class_weights(all_quality_indices, num_quality_classes, clip_max=5.0)

    print("   Βάρη κλάσεων quality (idx: count -> weight), ταξινομημένα κατά βάρος:")
    idx_to_quality = {idx: q for q, idx in quality_to_idx.items()}
    for i in sorted(range(num_quality_classes), key=lambda i: -quality_weights_tensor[i].item()):
        q_name = idx_to_quality.get(i, '?')
        print(f"      idx {i:2d} ({q_name:8}): count={quality_counts_dbg.get(i, 0):>8}  weight={quality_weights_tensor[i].item():.2f}")

    # ΑΝΑΠΑΡΑΓΩΓΙΜΟΤΗΤΑ: μέχρι τώρα μόνο το random.seed(42) για το data split ήταν
    # σταθερό - η ΑΡΧΙΚΟΠΟΙΗΣΗ των βαρών του μοντέλου (nn.Embedding, nn.Linear,
    # εσωτερικά του XTEncoder) ΔΕΝ ήταν seeded, άρα κάθε run ξεκινούσε από
    # διαφορετικά τυχαία βάρη - πιθανή εξήγηση για το γιατί δύο runs με τον ΙΔΙΟ
    # κώδικα έδειξαν διαφορετική πορεία σύγκλισης. torch.manual_seed() ΔΕΝ εγγυάται
    # απόλυτη, bit-for-bit αναπαραγωγιμότητα σε GPU (ορισμένοι πυρήνες CUDA
    # παραμένουν μη-ντετερμινιστικοί εξ ορισμού), αλλά καθιστά την αρχικοποίηση
    # βαρών συνεπή, που είναι ήδη μεγάλο βήμα προς πιο συγκρίσιμα πειράματα.
    MODEL_SEED = int(os.environ.get('MODEL_SEED', 1))
    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)

    MASK_CHORD_ID = num_chord_classes  # ΙΔΙΑ σύμβαση "mask/pad id = vocab_size"
    model = HarmonizerTransformerIterative(
        num_pitch_classes=NUM_PITCH_OCT_CLASSES, pad_pitch_id=PAD_PITCH_ID,
        num_duration_classes=NUM_DURATION_CLASSES, pad_duration_id=PAD_DURATION_ID,
        num_beat_type_classes=NUM_BEAT_TYPE_CLASSES, pad_beat_type_id=PAD_BEAT_TYPE_ID,
        num_position_classes=NUM_POSITION_CLASSES, pad_position_id=PAD_POSITION_ID,
        num_interval_classes=NUM_INTERVAL_CLASSES, pad_interval_id=PAD_INTERVAL_ID,
        num_chord_classes=num_chord_classes, mask_chord_id=MASK_CHORD_ID,
        embed_dim=EMBEDDING_DIM, num_heads=8, num_layers=4, num_root_classes=num_root_classes,
        num_quality_classes=num_quality_classes, dropout=0.3
    ).to(device)

    # ΝΕΟ: πίνακας (root,quality)->flat chord id, για την κατασκευή του
    # chord_context_ids σε κάθε batch (βλ. build_chord_context_ids).
    root_qual_to_flat_table, _ = build_root_qual_to_flat_table(
        flat_key_to_idx_arg, num_root_classes, num_quality_classes, MASK_CHORD_ID
    )
    root_qual_to_flat_table = root_qual_to_flat_table.to(device)

    # ΝΕΟ (v2): πίνακας κεντροειδών για το tonal-distance loss - ΜΙΑ φορά, πριν
    # ξεκινήσει η εκπαίδευση (όχι ανά batch - θα ήταν σπατάλη υπολογισμού).
    centroid_table, centroid_mask_table = build_centroid_table(
        idx_to_root_pc, idx_to_quality, num_root_classes, num_quality_classes
    )
    centroid_table = centroid_table.to(device)
    centroid_mask_table = centroid_mask_table.to(device)

    # ΝΕΟ: πίνακας για ΖΩΝΤΑΝΗ μέτρηση 'κοντινής' ακρίβειας δίπλα στο Val Acc -
    # ΚΑΘΑΡΑ μέτρηση, ΔΕΝ συμμετέχει στο loss/gradient.
    tone_vector_table, tone_vector_mask_table = build_tone_vector_table(
        idx_to_root_pc, idx_to_quality, num_root_classes, num_quality_classes
    )
    tone_vector_table = tone_vector_table.to(device)
    tone_vector_mask_table = tone_vector_mask_table.to(device)

    collate_fn = make_collate_fn(PAD_NOTE_ID, PAD_PITCH_ID, PAD_DURATION_ID, PAD_BEAT_TYPE_ID,
                                  PAD_POSITION_ID, PAD_INTERVAL_ID)
    train_loader = DataLoader(SongDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(SongDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"   Κανόνες 1 & 2 (μετρικό βάρος + step-resolution): {'ΕΝΕΡΓΟΙ' if enable_rules_1_2 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΙ (ablation)'}")
    print(f"   Κανόνας 3 (σταθερότητα αρμονικού ρυθμού): {'ΕΝΕΡΓΟΣ' if enable_rule_3 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ (ablation)'} (βάρος={RULE3_WEIGHT:.2f}, από τα δεδομένα)")
    print(f"   Κανόνας 4 (κίνηση ρίζας): {'ΕΝΕΡΓΟΣ' if enable_rule_4 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ (ablation)'} (βάρος={RULE4_WEIGHT:.2f}, από τα δεδομένα)")
    print(f"   Κανόνας 5 (επίλυση leading tone): {'ΕΝΕΡΓΟΣ' if enable_rule_5 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ (ablation)'} (strictness={RULE5_LENIENCY_FACTOR:.2f}, από τα δεδομένα)")
    print(f"   Κανόνας 6 (λειτουργική αρμονία T/S/D): {'ΕΝΕΡΓΟΣ' if enable_rule_6 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ (ablation)'} (βάρος={RULE6_WEIGHT:.2f}, από τα δεδομένα)")
    print(f"   Κανόνας 7 (τονική στο τέλος): {'ΕΝΕΡΓΟΣ' if enable_rule_7 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ (ablation)'} (τέλος={RULE7_END_WEIGHT:.2f}, από τα δεδομένα)")
    print(f"   Κανόνας 8 (ποιότητα ανά λειτουργία): {'ΕΝΕΡΓΟΣ' if enable_rule_8 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ (ablation)'} (βάρος={RULE8_WEIGHT:.2f}, από τα δεδομένα)")
    print(f"   Tonal-distance loss (Harte κεντροειδές): {'ΕΝΕΡΓΟ' if enable_tonal_distance_loss else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟ (ablation)'} (βάρος={TONAL_DISTANCE_WEIGHT:.2f}, σταθερό)")

    criterion = RuleInformedLoss(
        penalty_weight=PENALTY_WEIGHT, rule_matrix=comp_matrix_tensor,
        root_weights=root_weights_tensor, quality_weights=quality_weights_tensor,
        root_mapping=chord_root_mapping, qual_mapping=chord_qual_mapping,
        beat_type_scale=beat_type_scale_tensor, step_leniency_factor=0.3,
        enable_rules_1_2=enable_rules_1_2,
        expected_same_scale=expected_same_tensor, rule3_weight=RULE3_WEIGHT, enable_rule_3=enable_rule_3,
        rule3_strictness_scale=rule3_strictness_tensor,
        root_transition_target=root_transition_target, rule4_weight=RULE4_WEIGHT, enable_rule_4=enable_rule_4,
        rule5_leniency_factor=RULE5_LENIENCY_FACTOR, enable_rule_5=enable_rule_5,
        root_to_func_by_mode=root_to_func_by_mode, function_transition_target=function_transition_target_by_mode,
        rule6_weight=RULE6_WEIGHT, enable_rule_6=enable_rule_6,
        rule7_end_weight=RULE7_END_WEIGHT,
        tonic_function_idx=FUNCTION_TO_IDX['T'], enable_rule_7=enable_rule_7,
        quality_given_function_target=quality_given_function_target, rule8_weight=RULE8_WEIGHT,
        enable_rule_8=enable_rule_8,
        centroid_table=centroid_table,
        centroid_mask_table=centroid_mask_table,
        tonal_distance_weight=TONAL_DISTANCE_WEIGHT, enable_tonal_distance_loss=enable_tonal_distance_loss
    )

    # AdamW: σωστό decoupled weight decay (το plain Adam εφαρμόζει weight decay
    # λάθος, μπλέκοντάς το με τα gradient moving averages) - καθαρή βελτίωση χωρίς
    # μειονέκτημα. Weight decay ανέβηκε ελαφρά (1e-4 -> 2e-4) λόγω του ήπιου
    # overfitting gap που φαινόταν στο προηγούμενο run (train loss ξέφευγε κάτω
    # από το val loss προς το τέλος).
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=16)
    # Γραμμικό warmup: οι πρώτες λίγες εποχές ξεκινούν από πολύ χαμηλό LR και
    # ανεβαίνουν σταδιακά στο πλήρες lr, πριν ενεργοποιηθεί το ReduceLROnPlateau.
    # Transformers είναι γνωστό ότι μπορούν να αποσταθεροποιηθούν με πλήρες LR από
    # την αρχή, πριν προλάβουν να "ζεσταθούν" τα attention layers.

    patience_counter = 0
    best_val_loss = float('inf')
    best_model_path = os.path.join(OUT_DIR, "best_transformer_model.pth")
    # Ξεχωριστό tracking ΜΟΝΟ για αναφορά - ΔΕΝ επηρεάζει ποιο checkpoint σώζεται/
    # επιστρέφεται (αυτό παραμένει βάσει val loss, η πιο σωστή επιλογή ώστε να μην
    # "διαλέγουμε" αυθαίρετα το καλύτερο-ακρίβεια epoch μετά τα γεγονότα). Απλά μας
    # δείχνει πόσο διαφέρει η επιλεγμένη ακρίβεια από την ΚΑΛΥΤΕΡΗ που φάνηκε ποτέ.
    peak_val_acc, peak_val_acc_epoch = 0.0, -1

    # ΝΕΟ: global step counter (ανά batch, ΟΧΙ ανά εποχή) - βλ. σχόλιο στην
    # παράμετρο curriculum_total_epochs παραπάνω για γιατί ΔΕΝ χρησιμοποιούμε
    # το max_epochs εδώ.
    global_step = 0
    total_curriculum_steps = curriculum_total_epochs * len(train_loader)
    print(f"   Curriculum: {curriculum_total_epochs} εποχές × {len(train_loader)} batches "
          f"= {total_curriculum_steps} συνολικά βήματα (βάση του χρονοδιαγράμματος αποκάλυψης)")
    print("\nΞεκινάει η εκπαίδευση (bidirectional encoder, εμπλουτισμένα features, iterative)...")
    for epoch in range(max_epochs):
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for g in optimizer.param_groups:
                g['lr'] = warmup_lr

        model.train()
        train_loss = 0.0
        n_clamped_this_epoch = 0  # πόσες τιμές .g χτύπησαν το όριο [-10,10] αυτή την εποχή
        for (n_ids, pitch_ids, duration_ids, beat_type_ids, position_ids, interval_ids,
             beat_strength, step_resolved, resolving_lt, is_major, song_end,
             root_y, quality_y, pad_mask) in train_loader:
            n_ids, pad_mask = n_ids.to(device), pad_mask.to(device)
            pitch_ids, duration_ids = pitch_ids.to(device), duration_ids.to(device)
            beat_type_ids, position_ids = beat_type_ids.to(device), position_ids.to(device)
            interval_ids, beat_strength = interval_ids.to(device), beat_strength.to(device)
            step_resolved = step_resolved.to(device)
            resolving_lt = resolving_lt.to(device)
            is_major = is_major.to(device)
            song_end = song_end.to(device)
            root_y, quality_y = root_y.to(device), quality_y.to(device)

            optimizer.zero_grad()
            reveal_frac = curriculum_reveal_fraction(global_step, total_curriculum_steps)
            chord_context_ids, should_reveal = build_chord_context_ids(
                root_y, quality_y, pad_mask, reveal_frac, root_qual_to_flat_table, MASK_CHORD_ID
            )
            root_out, quality_out = model(pitch_ids, duration_ids, beat_type_ids, position_ids,
                                           interval_ids, beat_strength, chord_context_ids, pad_mask)
            global_step += 1

            # ΚΡΙΣΙΜΗ ΔΙΟΡΘΩΣΗ: το loss ΔΕΝ πρέπει να υπολογίζεται στις ήδη-
            # αποκαλυμμένες θέσεις - το paper υπολογίζει loss ΜΟΝΟ στις masked
            # (βλ. σχόλιο στο build_chord_context_ids). Χωρίς αυτό, το μοντέλο
            # έχει ένα trivial "αντέγραψε το input" shortcut στις αποκαλυμμένες
            # θέσεις - πιθανή αιτία της κατάρρευσης που παρατηρήσαμε.
            loss_root_y = root_y.masked_fill(should_reveal, -100)
            loss_quality_y = quality_y.masked_fill(should_reveal, -100)
            effective_pad_mask = pad_mask | should_reveal  # οι κανόνες βλέπουν τις
            # αποκαλυμμένες θέσεις σαν "padding" - αγνοούνται παντού, όχι μόνο στο
            # κύριο cross-entropy.

            loss, _ = criterion(n_ids, effective_pad_mask, beat_type_ids, step_resolved, resolving_lt, is_major,
                                 song_end, root_out, loss_root_y, quality_out, loss_quality_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # ΔΙΟΡΘΩΣΗ μετά το Διαγνωστικό 3: τα RMSNorm ".g" (πριν, ScaleNorm
            # ".gamma") βρέθηκαν να "εκρήγνυνται" σε τιμές ~10^7-10^8 μετά από πολλές
            # εποχές - ΑΝΕΞΑΡΤΗΤΑ από το ποιο norm variant χρησιμοποιείται (το ίδιο
            # πρόβλημα εμφανίστηκε ΚΑΙ στα δύο, άρα δεν είναι κάτι ειδικό στο
            # ScaleNorm όπως πιστεύαμε αρχικά). Το gradient clipping δεσμεύει μόνο
            # το ΜΕΓΕΘΟΣ κάθε μεμονωμένου βήματος, όχι τη ΣΩΡΕΥΤΙΚΗ κίνηση σε
            # εκατοντάδες εποχές - αν η κατεύθυνση του gradient παραμένει σταθερή
            # για πολύ καιρό, η παράμετρος μπορεί ακόμα να "περπατήσει" σε παράλογες
            # τιμές. Εδώ περιορίζουμε ΑΜΕΣΑ αυτές τις παραμέτρους μετά από κάθε
            # βήμα - εγγυημένη λύση, ανεξάρτητα από το τι ακριβώς προκαλεί την
            # ανοδική πίεση.
            with torch.no_grad():
                for param_name, param in model.named_parameters():
                    if param_name.endswith('.g'):
                        n_clamped_this_epoch += (param.abs() > 10.0).sum().item()
                        param.clamp_(-10.0, 10.0)
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct, root_correct, qual_correct, total = 0, 0, 0, 0
        near1_correct, near2_correct, near_total = 0, 0, 0  # ΝΕΟ: 'κοντινή' ακρίβεια
        qual_pred_counts = Counter()
        with torch.no_grad():
            for (n_ids, pitch_ids, duration_ids, beat_type_ids, position_ids, interval_ids,
                 beat_strength, step_resolved, resolving_lt, is_major, song_end,
                 root_y, quality_y, pad_mask) in val_loader:
                n_ids, pad_mask = n_ids.to(device), pad_mask.to(device)
                pitch_ids, duration_ids = pitch_ids.to(device), duration_ids.to(device)
                beat_type_ids, position_ids = beat_type_ids.to(device), position_ids.to(device)
                interval_ids, beat_strength = interval_ids.to(device), beat_strength.to(device)
                step_resolved = step_resolved.to(device)
                resolving_lt = resolving_lt.to(device)
                is_major = is_major.to(device)
                song_end = song_end.to(device)
                root_y, quality_y = root_y.to(device), quality_y.to(device)
                # ΠΑΝΤΑ πλήρως masked (v=0) - η επικύρωση προσομοιώνει ΠΡΑΓΜΑΤΙΚΟ
                # inference, όπου ΔΕΝ ξέρουμε τις αληθινές χορδές εκ των προτέρων.
                # Αυτό κρατάει το Val Acc ΑΜΕΣΑ συγκρίσιμο με το υπάρχον μοντέλο.
                val_chord_context_ids = torch.full_like(root_y, MASK_CHORD_ID)  # μόνο σχήμα/dtype χρειάζεται εδώ
                root_out, quality_out = model(pitch_ids, duration_ids, beat_type_ids, position_ids,
                                               interval_ids, beat_strength, val_chord_context_ids, pad_mask)
                loss, _ = criterion(n_ids, pad_mask, beat_type_ids, step_resolved, resolving_lt, is_major,
                                     song_end, root_out, root_y, quality_out, quality_y)
                val_loss += loss.item()

                root_preds = root_out.argmax(dim=-1)
                qual_preds = quality_out.argmax(dim=-1)
                valid = (~pad_mask)
                root_correct += ((root_preds == root_y) & valid).sum().item()
                qual_correct += ((qual_preds == quality_y) & valid).sum().item()
                correct += (((root_preds == root_y) & (qual_preds == quality_y)) & valid).sum().item()
                total += valid.sum().item()
                qual_pred_counts.update(qual_preds[valid].tolist())

                # ΝΕΟ: 'κοντινή' ακρίβεια - ΚΑΘΑΡΑ μέτρηση, όχι μέρος του loss.
                # root_preds/qual_preds είναι ΠΑΝΤΑ έγκυροι δείκτες (από argmax,
                # ποτέ -100) - ΜΟΝΟ τα root_y/quality_y χρειάζονται clamp λόγω
                # του ignore_index στα padded θέσεις.
                safe_root_y = root_y.clamp(min=0, max=tone_vector_table.shape[0] - 1)
                safe_qual_y = quality_y.clamp(min=0, max=tone_vector_table.shape[1] - 1)
                pred_tones = tone_vector_table[root_preds, qual_preds]      # [B,T,12]
                true_tones = tone_vector_table[safe_root_y, safe_qual_y]    # [B,T,12]
                pred_tones_valid = tone_vector_mask_table[root_preds, qual_preds]
                true_tones_valid = tone_vector_mask_table[safe_root_y, safe_qual_y]
                near_valid = pred_tones_valid & true_tones_valid & valid  # αγνοεί OTHER + padding
                distance = (pred_tones != true_tones).sum(dim=-1)  # Hamming, [B,T]
                near1_correct += ((distance <= 1) & near_valid).sum().item()
                near2_correct += ((distance <= 2) & near_valid).sum().item()
                near_total += near_valid.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = (correct / total * 100) if total > 0 else 0.0
        root_acc = (root_correct / total * 100) if total > 0 else 0.0
        qual_acc = (qual_correct / total * 100) if total > 0 else 0.0
        near1_acc = (near1_correct / near_total * 100) if near_total > 0 else 0.0
        near2_acc = (near2_correct / near_total * 100) if near_total > 0 else 0.0
        if val_acc > peak_val_acc:
            peak_val_acc, peak_val_acc_epoch = val_acc, epoch
        if epoch >= warmup_epochs:
            scheduler.step(avg_val_loss)

        n_unique_pred = len(qual_pred_counts)
        top_pred = qual_pred_counts.most_common(3)
        pred_summary = ", ".join(f"cls{c}:{n/total*100:.1f}%" for c, n in top_pred)
        print(f"   [diag] quality: {n_unique_pred} διαφορετικές κλάσεις προβλέφθηκαν | top: {pred_summary}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            status_msg = "⭐ Νέο ρεκόρ Val Loss!"
        else:
            patience_counter += 1
            status_msg = f"⚠️ Καμία βελτίωση ({patience_counter}/{patience_limit})"

        clamp_note = f" | .g clamped: {n_clamped_this_epoch}x" if n_clamped_this_epoch > 0 else ""
        # ΝΕΟ, ΠΑΝΤΑ ενεργό διαγνωστικό: τυπώνουμε την ΠΡΑΓΜΑΤΙΚΗ τιμή (όχι μόνο
        # μέτρημα ενεργοποίησης) της .g παραμέτρου ΑΠΕΥΘΕΙΑΣ από το live model object,
        # ΑΜΕΣΩΣ μετά το clamp. Αν αυτό δείξει σταθερά <=10 σε ΚΑΘΕ εποχή αλλά το
        # ΤΕΛΙΚΟ αποθηκευμένο checkpoint δείχνει εκατομμύρια (Διαγνωστικό 3), αυτό θα
        # αποδείξει ότι το πρόβλημα ΔΕΝ είναι στο ίδιο το clamp - θα είναι κάπου
        # αλλού (π.χ. πιθανή εσωτερική αναπαραμετροποίηση του RMSNorm στο
        # x-transformers, όπου το named_parameters() ΔΕΝ επιστρέφει το ΙΔΙΟ tensor
        # που χρησιμοποιείται στην πραγματικότητα στον υπολογισμό forward()).
        g_debug = ""
        for param_name, param in model.named_parameters():
            if param_name.endswith('.g'):
                g_debug = f" | live .g max|.|: {param.abs().max().item():.4f}"
                break
        print(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% (Root: {root_acc:.1f}%, Qual: {qual_acc:.1f}%) | "
              f"Near(≤1/≤2 νότες): {near1_acc:.1f}%/{near2_acc:.1f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | {status_msg}{clamp_note}{g_debug}")

        if patience_counter >= patience_limit:
            print(f"\n🛑 EARLY STOPPING στην Εποχή {epoch}!")
            model.load_state_dict(torch.load(best_model_path))
            print(f"   ℹ️ Το επιστρεφόμενο checkpoint επιλέχθηκε βάσει val loss. Η ΚΑΛΥΤΕΡΗ ακρίβεια "
                  f"που φάνηκε σε ΟΠΟΙΟΔΗΠΟΤΕ epoch ήταν {peak_val_acc:.2f}% (εποχή {peak_val_acc_epoch}) - "
                  f"δες αυτό το κενό πριν συγκρίνεις ρυθμίσεις πολύ κοντά μεταξύ τους.")
            break
    else:
        # Το loop έφτασε στο max_epochs ΧΩΡΙΣ να ενεργοποιηθεί early stopping - πρέπει
        # ΚΑΙ ΕΔΩ να φορτωθεί το καλύτερο (βάσει val loss) checkpoint, αλλιώς η
        # συνάρτηση επιστρέφει τα βάρη της ΤΕΛΕΥΤΑΙΑΣ εποχής, όχι του καλύτερου.
        print(f"\n📦 Ολοκληρώθηκαν όλες οι {max_epochs} εποχές χωρίς early stopping.")
        model.load_state_dict(torch.load(best_model_path))
        print(f"   ℹ️ Η ΚΑΛΥΤΕΡΗ ακρίβεια που φάνηκε σε ΟΠΟΙΟΔΗΠΟΤΕ epoch ήταν {peak_val_acc:.2f}% "
              f"(εποχή {peak_val_acc_epoch}).")

    # ΕΠΙΣΤΡΕΦΟΥΜΕ ΚΑΙ τα δύο αυτά - το harmonize_song_iterative τα χρειάζεται
    # ΩΣ globals (ίδια σύμβαση με το trained_model). Αν δεν τα επιστρέφαμε εδώ,
    # θα χρειαζόταν να τα ξαναχτίσεις χειροκίνητα αργότερα - ίδιο πρόβλημα με
    # το PAD_PITCH_ID νωρίτερα σε αυτό το project.
    return model, root_qual_to_flat_table, MASK_CHORD_ID


def train_model_concat_encoder(raw, chord_to_idx, note_to_idx, chord_root_idx, chord_quality_idx,
                 comp_matrix_np, num_note_classes, num_classes, num_root_classes, num_quality_classes,
                 num_chord_classes, flat_key_to_idx_arg,
                 batch_size=16, lr=1e-4, max_epochs=200, warmup_epochs=0,
                 # ΣΗΜΕΙΩΣΗ: batch_size=16, lr=1e-4, max_epochs=200 ΤΩΡΑ ταιριάζουν με
                 # τις default τιμές του ΠΡΑΓΜΑΤΙΚΟΥ κώδικά τους (train_semh.py). ΚΑΝΕΝΑ
                 # patience_limit πλέον - ΔΕΝ κάνουμε early stopping, ΑΚΡΙΒΩΣ όπως αυτοί
                 # για το f2f curriculum (βλ. σχόλιο στο τέλος της συνάρτησης).
                 enable_rules_1_2=True, enable_rule_3=True, enable_rule_4=True, enable_rule_5=True,
                 enable_rule_6=True, enable_rule_7=True, enable_rule_8=True,
                 enable_tonal_distance_loss=True, curriculum_total_epochs=80):
    # ΣΗΜΑΝΤΙΚΟ: curriculum_total_epochs (default 80) ΕΙΝΑΙ ΣΚΟΠΙΜΑ ΞΕΧΩΡΙΣΤΟ από
    # το max_epochs (300, σχεδόν ΠΟΤΕ δεν φτάνουμε εκεί λόγω early stopping - ΟΛΑ
    # τα runs μέχρι τώρα σταμάτησαν στις 50-80 εποχές). Αν το curriculum βάσιζε
    # την πρόοδό του στο max_epochs=300, το reveal_fraction θα ήταν ΣΧΕΔΟΝ ΜΗΔΕΝ
    # (~0.001) ακόμα και στην εποχή 80 - το μοντέλο ΣΧΕΔΟΝ ΠΟΤΕ δεν θα έβλεπε
    # αποκαλυμμένη χορδή πριν σταματήσει η εκπαίδευση. Το curriculum_total_epochs
    # βασίζεται σε ΠΡΑΓΜΑΤΙΚΗ, παρατηρημένη διάρκεια, ΟΧΙ στο θεωρητικό όριο.
    comp_matrix_tensor = torch.tensor(comp_matrix_np, dtype=torch.float32)
    PAD_NOTE_ID = num_note_classes
    PAD_PITCH_ID = NUM_PITCH_OCT_CLASSES
    PAD_DURATION_ID = NUM_DURATION_CLASSES
    PAD_BEAT_TYPE_ID = NUM_BEAT_TYPE_CLASSES
    PAD_POSITION_ID = NUM_POSITION_CLASSES
    PAD_INTERVAL_ID = NUM_INTERVAL_CLASSES

    # ΠΡΙΝ αποφασίσουμε το όριο μήκους, το ΜΕΤΡΑΜΕ (ίδια λογική με όλους τους
    # κανόνες) - θέλουμε ένα όριο αρκετά γενναιόδωρο ώστε ΣΧΕΔΟΝ ΚΑΝΕΝΑ τραγούδι
    # να μην κοπεί, όχι ένα αυθαίρετο νούμερο.
    song_lengths = sorted(len(melody_iris) for melody_iris, _ in raw)
    n_songs_total = len(song_lengths)
    p50 = song_lengths[n_songs_total // 2]
    p95 = song_lengths[int(n_songs_total * 0.95)]
    p99 = song_lengths[int(n_songs_total * 0.99)]
    max_len_seen = song_lengths[-1]
    # ΔΙΟΡΘΩΣΗ: το προηγούμενο όριο βασιζόταν στο ΑΠΟΛΥΤΟ max, το οποίο αποδείχθηκε
    # ένα ακραίο outlier (16623 νότες - 37x πάνω από το 99ό εκατοστημόριο!), σχεδόν
    # σίγουρα data artifact όχι πραγματικό τραγούδι. Αυτό ανάγκαζε ΚΑΘΕ batch που
    # (ατυχώς) το περιείχε να γεμίσει (pad) ΟΛΑ τα άλλα τραγούδια στο batch μέχρι
    # εκεί - αυτό προκάλεσε το OOM, όχι το whole-song training καθαυτό. Τώρα το
    # όριο βασίζεται στο 99ό εκατοστημόριο (με περιθώριο) - κόβει μόνο το πιο
    # ακραίο <=1% του corpus, επιτρέπει σε λογικό batch_size να δουλέψει ξανά.
    max_seq_len = max(64, int(p99 * 1.2))
    n_truncated = sum(1 for L in song_lengths if L > max_seq_len)
    print(f"   Μήκη τραγουδιών (νότες): median={p50}, 95ό εκατοστημόριο={p95}, 99ό={p99}, "
          f"max={max_len_seen} -> max_seq_len={max_seq_len} (θα κοπούν {n_truncated} τραγούδια, "
          f"δηλαδή {n_truncated/n_songs_total*100:.2f}% του corpus)")

    all_data = []
    for melody_iris, chord_iris in raw:
        n_ids = [note_to_idx[n] for n in melody_iris]
        pitch_ids = [pitch_oct_to_idx[n] for n in melody_iris]
        duration_ids = [duration_to_idx[n] for n in melody_iris]
        beat_type_ids = [beat_type_to_idx[n] for n in melody_iris]
        position_ids = [position_to_idx[n] for n in melody_iris]
        interval_ids = [interval_to_idx[n] for n in melody_iris]
        beat_strength_vals = [note_beat_strength[n] for n in melody_iris]
        root_y = [chord_root_idx[c] for c in chord_iris]
        quality_y = [chord_quality_idx[c] for c in chord_iris]

        # Κανόνας 2: "σωστά επιλυμένη" νότα = πλησιάστηκε Ή έφυγε με βήμα (<=2
        # semitones). Άγνωστο ύψος (parsing failure) -> ουδέτερο/lenient (1.0),
        # ώστε να μην τιμωρούμε λάθος κάτι που απλά δεν μπορέσαμε να διαβάσουμε.
        semis = [note_semitone[n] for n in melody_iris]
        step_resolved = [1.0] * len(semis)
        for t in range(len(semis)):
            if semis[t] is None:
                continue
            approached_ok = (t == 0) or (semis[t - 1] is None) or (abs(semis[t] - semis[t - 1]) <= 2)
            left_ok = (t == len(semis) - 1) or (semis[t + 1] is None) or (abs(semis[t + 1] - semis[t]) <= 2)
            step_resolved[t] = 1.0 if (approached_ok or left_ok) else 0.0

        # Κανόνας 5: "resolving leading tone" = η νότα ΕΙΝΑΙ leading tone (mode-aware)
        # ΚΑΙ η επόμενη νότα πράγματι φτάνει στην τονική.
        resolving_leading_tone = [0.0] * len(melody_iris)
        for t in range(len(melody_iris) - 1):
            n = melody_iris[t]
            if note_is_leading_tone.get(n):
                next_pc = note_pitch_class_transposed.get(melody_iris[t + 1])
                if next_pc == note_expected_tonic_pc.get(n):
                    resolving_leading_tone[t] = 1.0

        # Κανόνας 6: το mode είναι σταθερό ανά τραγούδι - το παίρνουμε από την πρώτη
        # χορδή του τραγουδιού.
        song_is_major = 1.0 if chord_is_major.get(chord_iris[0], True) else 0.0

        # Κανόνας 7: 1.0 ΜΟΝΟ στην τελευταία θέση ΟΛΟΚΛΗΡΟΥ του τραγουδιού.
        song_end_mask = [0.0] * len(melody_iris)
        if len(melody_iris) > 0:
            song_end_mask[-1] = 1.0

        # ΟΛΟΚΛΗΡΟ το τραγούδι σαν ΕΝΑ training example - όχι κομμένο σε chunks των
        # 64. Αυτό διορθώνει ΔΥΟ προβλήματα ταυτόχρονα: (1) το bidirectional
        # attention τώρα βλέπει ΠΡΑΓΜΑΤΙΚΑ όλο το τραγούδι, όχι ένα αυθαίρετο
        # παράθυρο, (2) το positional encoding δεν "νομίζει" πλέον ότι κάθε
        # εσωτερικό κομμάτι είναι η αρχή ενός τραγουδιού. Κόβουμε ΜΟΝΟ στα ελάχιστα
        # ακραία outliers (βλ. max_seq_len παραπάνω, μετρημένο από τα δεδομένα).
        n_ids_final = n_ids[:max_seq_len]
        if len(n_ids_final) >= 8:
            all_data.append((
                torch.tensor(n_ids_final, dtype=torch.long),
                torch.tensor(pitch_ids[:max_seq_len], dtype=torch.long),
                torch.tensor(duration_ids[:max_seq_len], dtype=torch.long),
                torch.tensor(beat_type_ids[:max_seq_len], dtype=torch.long),
                torch.tensor(position_ids[:max_seq_len], dtype=torch.long),
                torch.tensor(interval_ids[:max_seq_len], dtype=torch.long),
                torch.tensor(beat_strength_vals[:max_seq_len], dtype=torch.float32),
                torch.tensor(step_resolved[:max_seq_len], dtype=torch.float32),
                torch.tensor(resolving_leading_tone[:max_seq_len], dtype=torch.float32),
                torch.tensor(song_is_major, dtype=torch.long),
                torch.tensor(song_end_mask[:max_seq_len], dtype=torch.float32),
                torch.tensor(root_y[:max_seq_len], dtype=torch.long),
                torch.tensor(quality_y[:max_seq_len], dtype=torch.long),
            ))

    # Το shuffle/split είναι ΤΩΡΑ αυτόματα σε επίπεδο ΤΡΑΓΟΥΔΙΟΥ (αφού κάθε
    # στοιχείο του all_data ΕΙΝΑΙ ένα ολόκληρο τραγούδι) - διορθώνει και το
    # πρόβλημα διαρροής train/val που είχαμε με το chunking.
    random.seed(42)
    random.shuffle(all_data)
    split_idx = int(0.8 * len(all_data))
    train_data, val_data = all_data[:split_idx], all_data[split_idx:]
    print(f"📦 {len(train_data)} τραγούδια εκπαίδευσης, {len(val_data)} τραγούδια επικύρωσης "
          f"(ολόκληρα, max_seq_len={max_seq_len})")

    all_root_indices = [chord_root_idx[c] for _, chord_labels in raw for c in chord_labels]
    all_quality_indices = [chord_quality_idx[c] for _, chord_labels in raw for c in chord_labels]

    def make_class_weights(indices, n_classes, clip_max=10.0):
        counts = Counter(indices)
        total = len(indices)
        w = [0.0] * n_classes
        for i in range(n_classes):
            count = counts.get(i, 0)
            raw_w = total / (n_classes * count) if count > 0 else 1.0
            w[i] = math.sqrt(raw_w)
        w = np.clip(w, 0.1, clip_max)
        return torch.tensor(w, dtype=torch.float32).to(device), counts

    # ΥΠΟΘΕΣΗ: με λιγότερες κλάσεις (254 -> 24 μετά το vocabulary cleanup), ο ίδιος
    # τύπος βάρους (1/n_classes στον παρονομαστή) δίνει ΠΙΟ ακραία βάρη σε σπάνιες
    # (ΣΧΕΤΙΚΑ, όχι απόλυτα) κλάσεις - πιθανή αιτία της πτώσης στο quality accuracy
    # (47%->38%) παρόλο που το λεξιλόγιο έγινε "πιο εύκολο". Σφίγγουμε το clip_max
    # για το quality head, ΚΑΙ τυπώνουμε τα πραγματικά βάρη ώστε να το επιβεβαιώσουμε.
    root_weights_tensor, root_counts = make_class_weights(all_root_indices, num_root_classes, clip_max=10.0)
    quality_weights_tensor, quality_counts_dbg = make_class_weights(all_quality_indices, num_quality_classes, clip_max=5.0)

    print("   Βάρη κλάσεων quality (idx: count -> weight), ταξινομημένα κατά βάρος:")
    idx_to_quality = {idx: q for q, idx in quality_to_idx.items()}
    for i in sorted(range(num_quality_classes), key=lambda i: -quality_weights_tensor[i].item()):
        q_name = idx_to_quality.get(i, '?')
        print(f"      idx {i:2d} ({q_name:8}): count={quality_counts_dbg.get(i, 0):>8}  weight={quality_weights_tensor[i].item():.2f}")

    # ΑΝΑΠΑΡΑΓΩΓΙΜΟΤΗΤΑ: μέχρι τώρα μόνο το random.seed(42) για το data split ήταν
    # σταθερό - η ΑΡΧΙΚΟΠΟΙΗΣΗ των βαρών του μοντέλου (nn.Embedding, nn.Linear,
    # εσωτερικά του XTEncoder) ΔΕΝ ήταν seeded, άρα κάθε run ξεκινούσε από
    # διαφορετικά τυχαία βάρη - πιθανή εξήγηση για το γιατί δύο runs με τον ΙΔΙΟ
    # κώδικα έδειξαν διαφορετική πορεία σύγκλισης. torch.manual_seed() ΔΕΝ εγγυάται
    # απόλυτη, bit-for-bit αναπαραγωγιμότητα σε GPU (ορισμένοι πυρήνες CUDA
    # παραμένουν μη-ντετερμινιστικοί εξ ορισμού), αλλά καθιστά την αρχικοποίηση
    # βαρών συνεπή, που είναι ήδη μεγάλο βήμα προς πιο συγκρίσιμα πειράματα.
    MODEL_SEED = int(os.environ.get('MODEL_SEED', 1))
    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)

    MASK_CHORD_ID = num_chord_classes  # ΙΔΙΑ σύμβαση "mask/pad id = vocab_size"
    # ΝΕΟ μοντέλο, ΝΕΕΣ διαστάσεις - ΑΚΡΙΒΩΣ όπως το SEModular τους:
    # d_model=512 (πριν 256), num_layers=8 (πριν 4), nhead=8 (ίδιο με πριν).
    # ΣΗΜΕΙΩΣΗ: αυτό ΣΗΜΑΝΤΙΚΑ αυξάνει το υπολογιστικό κόστος (d_model 2x
    # -> ~4x περισσότερες παράμετροι/FLOPs στα γραμμικά layers).
    CONCAT_EMBED_DIM = 512
    model = HarmonizerConcatEncoder(
        num_pitch_classes=NUM_PITCH_OCT_CLASSES, pad_pitch_id=PAD_PITCH_ID,
        num_duration_classes=NUM_DURATION_CLASSES, pad_duration_id=PAD_DURATION_ID,
        num_beat_type_classes=NUM_BEAT_TYPE_CLASSES, pad_beat_type_id=PAD_BEAT_TYPE_ID,
        num_position_classes=NUM_POSITION_CLASSES, pad_position_id=PAD_POSITION_ID,
        num_interval_classes=NUM_INTERVAL_CLASSES, pad_interval_id=PAD_INTERVAL_ID,
        num_chord_classes=num_chord_classes, mask_chord_id=MASK_CHORD_ID,
        embed_dim=CONCAT_EMBED_DIM, num_heads=8, num_layers=8, num_root_classes=num_root_classes,
        num_quality_classes=num_quality_classes, max_seq_len=max_seq_len, dropout=0.3
    ).to(device)

    # ΝΕΟ: πίνακας (root,quality)->flat chord id, για την κατασκευή του
    # chord_context_ids σε κάθε batch (βλ. build_chord_context_ids).
    root_qual_to_flat_table, _ = build_root_qual_to_flat_table(
        flat_key_to_idx_arg, num_root_classes, num_quality_classes, MASK_CHORD_ID
    )
    root_qual_to_flat_table = root_qual_to_flat_table.to(device)

    # ΝΕΟ (v2): πίνακας κεντροειδών για το tonal-distance loss - ΜΙΑ φορά, πριν
    # ξεκινήσει η εκπαίδευση (όχι ανά batch - θα ήταν σπατάλη υπολογισμού).
    centroid_table, centroid_mask_table = build_centroid_table(
        idx_to_root_pc, idx_to_quality, num_root_classes, num_quality_classes
    )
    centroid_table = centroid_table.to(device)
    centroid_mask_table = centroid_mask_table.to(device)

    # ΝΕΟ: πίνακας για ΖΩΝΤΑΝΗ μέτρηση 'κοντινής' ακρίβειας δίπλα στο Val Acc -
    # ΚΑΘΑΡΑ μέτρηση, ΔΕΝ συμμετέχει στο loss/gradient.
    tone_vector_table, tone_vector_mask_table = build_tone_vector_table(
        idx_to_root_pc, idx_to_quality, num_root_classes, num_quality_classes
    )
    tone_vector_table = tone_vector_table.to(device)
    tone_vector_mask_table = tone_vector_mask_table.to(device)

    collate_fn = make_collate_fn(PAD_NOTE_ID, PAD_PITCH_ID, PAD_DURATION_ID, PAD_BEAT_TYPE_ID,
                                  PAD_POSITION_ID, PAD_INTERVAL_ID)
    train_loader = DataLoader(SongDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(SongDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"   Κανόνες 1 & 2 (μετρικό βάρος + step-resolution): {'ΕΝΕΡΓΟΙ' if enable_rules_1_2 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΙ (ablation)'}")
    print(f"   Κανόνας 3 (σταθερότητα αρμονικού ρυθμού): {'ΕΝΕΡΓΟΣ' if enable_rule_3 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ (ablation)'} (βάρος={RULE3_WEIGHT:.2f}, από τα δεδομένα)")
    print(f"   Κανόνας 4 (κίνηση ρίζας): {'ΕΝΕΡΓΟΣ' if enable_rule_4 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ (ablation)'} (βάρος={RULE4_WEIGHT:.2f}, από τα δεδομένα)")
    print(f"   Κανόνας 5 (επίλυση leading tone): {'ΕΝΕΡΓΟΣ' if enable_rule_5 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ (ablation)'} (strictness={RULE5_LENIENCY_FACTOR:.2f}, από τα δεδομένα)")
    print(f"   Κανόνας 6 (λειτουργική αρμονία T/S/D): {'ΕΝΕΡΓΟΣ' if enable_rule_6 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ (ablation)'} (βάρος={RULE6_WEIGHT:.2f}, από τα δεδομένα)")
    print(f"   Κανόνας 7 (τονική στο τέλος): {'ΕΝΕΡΓΟΣ' if enable_rule_7 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ (ablation)'} (τέλος={RULE7_END_WEIGHT:.2f}, από τα δεδομένα)")
    print(f"   Κανόνας 8 (ποιότητα ανά λειτουργία): {'ΕΝΕΡΓΟΣ' if enable_rule_8 else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ (ablation)'} (βάρος={RULE8_WEIGHT:.2f}, από τα δεδομένα)")
    print(f"   Tonal-distance loss (Harte κεντροειδές): {'ΕΝΕΡΓΟ' if enable_tonal_distance_loss else 'ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟ (ablation)'} (βάρος={TONAL_DISTANCE_WEIGHT:.2f}, σταθερό)")

    criterion = RuleInformedLoss(
        penalty_weight=PENALTY_WEIGHT, rule_matrix=comp_matrix_tensor,
        root_weights=root_weights_tensor, quality_weights=quality_weights_tensor,
        root_mapping=chord_root_mapping, qual_mapping=chord_qual_mapping,
        beat_type_scale=beat_type_scale_tensor, step_leniency_factor=0.3,
        enable_rules_1_2=enable_rules_1_2,
        expected_same_scale=expected_same_tensor, rule3_weight=RULE3_WEIGHT, enable_rule_3=enable_rule_3,
        rule3_strictness_scale=rule3_strictness_tensor,
        root_transition_target=root_transition_target, rule4_weight=RULE4_WEIGHT, enable_rule_4=enable_rule_4,
        rule5_leniency_factor=RULE5_LENIENCY_FACTOR, enable_rule_5=enable_rule_5,
        root_to_func_by_mode=root_to_func_by_mode, function_transition_target=function_transition_target_by_mode,
        rule6_weight=RULE6_WEIGHT, enable_rule_6=enable_rule_6,
        rule7_end_weight=RULE7_END_WEIGHT,
        tonic_function_idx=FUNCTION_TO_IDX['T'], enable_rule_7=enable_rule_7,
        quality_given_function_target=quality_given_function_target, rule8_weight=RULE8_WEIGHT,
        enable_rule_8=enable_rule_8,
        centroid_table=centroid_table,
        centroid_mask_table=centroid_mask_table,
        tonal_distance_weight=TONAL_DISTANCE_WEIGHT, enable_tonal_distance_loss=enable_tonal_distance_loss
    )

    # AdamW: σωστό decoupled weight decay (το plain Adam εφαρμόζει weight decay
    # λάθος, μπλέκοντάς το με τα gradient moving averages) - καθαρή βελτίωση χωρίς
    # μειονέκτημα. Weight decay ανέβηκε ελαφρά (1e-4 -> 2e-4) λόγω του ήπιου
    # overfitting gap που φαινόταν στο προηγούμενο run (train loss ξέφευγε κάτω
    # από το val loss προς το τέλος).
    # ΝΕΟ: ΚΑΝΕΝΑ scheduler, ΣΤΑΘΕΡΟ lr - ΑΚΡΙΒΩΣ όπως το train_semh.py τους
    # (η γραμμή τους με cosine scheduler είναι ΣΧΟΛΙΑΣΜΕΝΗ/ανενεργή).
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_model_path = os.path.join(OUT_DIR, "best_transformer_model.pth")
    # ΝΕΟ: ΧΩΡΙΣ early stopping - ΑΥΤΟ το tracking είναι ΜΟΝΟ για πληροφόρηση
    # (ποιο epoch είχε την καλύτερη Val Acc), ΔΕΝ επηρεάζει τι επιστρέφεται στο
    # τέλος - ΠΑΝΤΑ επιστρέφεται/σώζεται το ΤΕΛΕΥΤΑΙΟ epoch, ΑΚΡΙΒΩΣ όπως αυτοί
    # κάνουν για το f2f curriculum ("we saved the model at the last epoch").
    peak_val_acc, peak_val_acc_epoch = 0.0, -1
    best_val_loss_seen = float('inf')
    best_val_model_path = os.path.join(OUT_DIR, "best_val_transformer_model.pth")
    best_val_acc_path = os.path.join(OUT_DIR, "best_val_acc_transformer_model.pth")
    if os.path.exists(best_val_model_path):
        _bv = torch.load(best_val_model_path, map_location=device)
        best_val_loss_seen = _bv.get('val_loss', float('inf'))
        print(f"   ℹ️ Βρέθηκε best_val checkpoint με val_loss={best_val_loss_seen:.4f} (εποχή {_bv.get('epoch')})")
    # ΝΕΟ: RESUME LOGIC - 200 εποχές σε αυτό το μοντέλο δεν χωράνε σε ΕΝΑ Kaggle
    # session (12ωρο όριο εκτέλεσης). Αν βρεθεί ΗΔΗ ένα checkpoint από
    # προηγούμενο, διακομμένο session, το ΞΑΝΑΦΟΡΤΩΝΟΥΜΕ και ΣΥΝΕΧΙΖΟΥΜΕ από
    # ΕΚΕΙ - όχι από την αρχή. ΚΡΙΣΙΜΟ: το global_step ΠΡΕΠΕΙ να συνεχίσει σωστά
    # (όχι να ξαναρχίσει από 0), αλλιώς το curriculum θα "ξαναγυρίσει" σε σχεδόν
    # πλήρως masked, χάνοντας την πρόοδο που είχε ήδη γίνει.
    start_epoch = 0
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"📦 Βρέθηκε checkpoint από προηγούμενο session - συνέχεια από την "
                  f"εποχή {start_epoch} (όχι από την αρχή).")
        else:
            print("⚠️ Βρέθηκε checkpoint παλιάς μορφής (μόνο βάρη, χωρίς epoch) - "
                  "ΔΕΝ μπορεί να χρησιμοποιηθεί για resume. Ξεκινάει από την αρχή.")

    # ΝΕΟ: global step counter (ανά batch, ΟΧΙ ανά εποχή) - βλ. σχόλιο στην
    # παράμετρο curriculum_total_epochs παραπάνω για γιατί ΔΕΝ χρησιμοποιούμε
    # το max_epochs εδώ. Αν κάνουμε resume, ΞΕΚΙΝΑΕΙ από το σωστό σημείο - ΔΕΝ
    # ξαναρχίζει από 0, αλλιώς το curriculum θα χανόταν.
    global_step = start_epoch * len(train_loader)
    total_curriculum_steps = curriculum_total_epochs * len(train_loader)
    print(f"   Curriculum: {curriculum_total_epochs} εποχές × {len(train_loader)} batches "
          f"= {total_curriculum_steps} συνολικά βήματα (βάση του χρονοδιαγράμματος αποκάλυψης)")
    print("\nΞεκινάει η εκπαίδευση (concat-encoder, ΑΚΡΙΒΩΣ όπως το paper, ΧΩΡΙΣ early stopping)...")
    for epoch in range(start_epoch, max_epochs):
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for g in optimizer.param_groups:
                g['lr'] = warmup_lr

        model.train()
        train_loss = 0.0
        n_clamped_this_epoch = 0  # πόσες τιμές .g χτύπησαν το όριο [-10,10] αυτή την εποχή
        for (n_ids, pitch_ids, duration_ids, beat_type_ids, position_ids, interval_ids,
             beat_strength, step_resolved, resolving_lt, is_major, song_end,
             root_y, quality_y, pad_mask) in train_loader:
            n_ids, pad_mask = n_ids.to(device), pad_mask.to(device)
            pitch_ids, duration_ids = pitch_ids.to(device), duration_ids.to(device)
            beat_type_ids, position_ids = beat_type_ids.to(device), position_ids.to(device)
            interval_ids, beat_strength = interval_ids.to(device), beat_strength.to(device)
            step_resolved = step_resolved.to(device)
            resolving_lt = resolving_lt.to(device)
            is_major = is_major.to(device)
            song_end = song_end.to(device)
            root_y, quality_y = root_y.to(device), quality_y.to(device)

            optimizer.zero_grad()
            reveal_frac = curriculum_reveal_fraction(global_step, total_curriculum_steps)
            chord_context_ids, should_reveal = build_chord_context_ids(
                root_y, quality_y, pad_mask, reveal_frac, root_qual_to_flat_table, MASK_CHORD_ID
            )
            root_out, quality_out = model(pitch_ids, duration_ids, beat_type_ids, position_ids,
                                           interval_ids, beat_strength, chord_context_ids, pad_mask)
            global_step += 1

            # ΚΡΙΣΙΜΗ ΔΙΟΡΘΩΣΗ: το loss ΔΕΝ πρέπει να υπολογίζεται στις ήδη-
            # αποκαλυμμένες θέσεις - το paper υπολογίζει loss ΜΟΝΟ στις masked
            # (βλ. σχόλιο στο build_chord_context_ids). Χωρίς αυτό, το μοντέλο
            # έχει ένα trivial "αντέγραψε το input" shortcut στις αποκαλυμμένες
            # θέσεις - πιθανή αιτία της κατάρρευσης που παρατηρήσαμε.
            loss_root_y = root_y.masked_fill(should_reveal, -100)
            loss_quality_y = quality_y.masked_fill(should_reveal, -100)
            effective_pad_mask = pad_mask | should_reveal  # οι κανόνες βλέπουν τις
            # αποκαλυμμένες θέσεις σαν "padding" - αγνοούνται παντού, όχι μόνο στο
            # κύριο cross-entropy.

            loss, _ = criterion(n_ids, effective_pad_mask, beat_type_ids, step_resolved, resolving_lt, is_major,
                                 song_end, root_out, loss_root_y, quality_out, loss_quality_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # ΔΙΟΡΘΩΣΗ μετά το Διαγνωστικό 3: τα RMSNorm ".g" (πριν, ScaleNorm
            # ".gamma") βρέθηκαν να "εκρήγνυνται" σε τιμές ~10^7-10^8 μετά από πολλές
            # εποχές - ΑΝΕΞΑΡΤΗΤΑ από το ποιο norm variant χρησιμοποιείται (το ίδιο
            # πρόβλημα εμφανίστηκε ΚΑΙ στα δύο, άρα δεν είναι κάτι ειδικό στο
            # ScaleNorm όπως πιστεύαμε αρχικά). Το gradient clipping δεσμεύει μόνο
            # το ΜΕΓΕΘΟΣ κάθε μεμονωμένου βήματος, όχι τη ΣΩΡΕΥΤΙΚΗ κίνηση σε
            # εκατοντάδες εποχές - αν η κατεύθυνση του gradient παραμένει σταθερή
            # για πολύ καιρό, η παράμετρος μπορεί ακόμα να "περπατήσει" σε παράλογες
            # τιμές. Εδώ περιορίζουμε ΑΜΕΣΑ αυτές τις παραμέτρους μετά από κάθε
            # βήμα - εγγυημένη λύση, ανεξάρτητα από το τι ακριβώς προκαλεί την
            # ανοδική πίεση.
            with torch.no_grad():
                for param_name, param in model.named_parameters():
                    if param_name.endswith('.g'):
                        n_clamped_this_epoch += (param.abs() > 10.0).sum().item()
                        param.clamp_(-10.0, 10.0)
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct, root_correct, qual_correct, total = 0, 0, 0, 0
        near1_correct, near2_correct, near_total = 0, 0, 0  # ΝΕΟ: 'κοντινή' ακρίβεια
        qual_pred_counts = Counter()
        with torch.no_grad():
            for (n_ids, pitch_ids, duration_ids, beat_type_ids, position_ids, interval_ids,
                 beat_strength, step_resolved, resolving_lt, is_major, song_end,
                 root_y, quality_y, pad_mask) in val_loader:
                n_ids, pad_mask = n_ids.to(device), pad_mask.to(device)
                pitch_ids, duration_ids = pitch_ids.to(device), duration_ids.to(device)
                beat_type_ids, position_ids = beat_type_ids.to(device), position_ids.to(device)
                interval_ids, beat_strength = interval_ids.to(device), beat_strength.to(device)
                step_resolved = step_resolved.to(device)
                resolving_lt = resolving_lt.to(device)
                is_major = is_major.to(device)
                song_end = song_end.to(device)
                root_y, quality_y = root_y.to(device), quality_y.to(device)
                # ΠΑΝΤΑ πλήρως masked (v=0) - η επικύρωση προσομοιώνει ΠΡΑΓΜΑΤΙΚΟ
                # inference, όπου ΔΕΝ ξέρουμε τις αληθινές χορδές εκ των προτέρων.
                # Αυτό κρατάει το Val Acc ΑΜΕΣΑ συγκρίσιμο με το υπάρχον μοντέλο.
                val_chord_context_ids = torch.full_like(root_y, MASK_CHORD_ID)  # μόνο σχήμα/dtype χρειάζεται εδώ
                root_out, quality_out = model(pitch_ids, duration_ids, beat_type_ids, position_ids,
                                               interval_ids, beat_strength, val_chord_context_ids, pad_mask)
                loss, _ = criterion(n_ids, pad_mask, beat_type_ids, step_resolved, resolving_lt, is_major,
                                     song_end, root_out, root_y, quality_out, quality_y)
                val_loss += loss.item()

                root_preds = root_out.argmax(dim=-1)
                qual_preds = quality_out.argmax(dim=-1)
                valid = (~pad_mask)
                root_correct += ((root_preds == root_y) & valid).sum().item()
                qual_correct += ((qual_preds == quality_y) & valid).sum().item()
                correct += (((root_preds == root_y) & (qual_preds == quality_y)) & valid).sum().item()
                total += valid.sum().item()
                qual_pred_counts.update(qual_preds[valid].tolist())

                # ΝΕΟ: 'κοντινή' ακρίβεια - ΚΑΘΑΡΑ μέτρηση, όχι μέρος του loss.
                # root_preds/qual_preds είναι ΠΑΝΤΑ έγκυροι δείκτες (από argmax,
                # ποτέ -100) - ΜΟΝΟ τα root_y/quality_y χρειάζονται clamp λόγω
                # του ignore_index στα padded θέσεις.
                safe_root_y = root_y.clamp(min=0, max=tone_vector_table.shape[0] - 1)
                safe_qual_y = quality_y.clamp(min=0, max=tone_vector_table.shape[1] - 1)
                pred_tones = tone_vector_table[root_preds, qual_preds]      # [B,T,12]
                true_tones = tone_vector_table[safe_root_y, safe_qual_y]    # [B,T,12]
                pred_tones_valid = tone_vector_mask_table[root_preds, qual_preds]
                true_tones_valid = tone_vector_mask_table[safe_root_y, safe_qual_y]
                near_valid = pred_tones_valid & true_tones_valid & valid  # αγνοεί OTHER + padding
                distance = (pred_tones != true_tones).sum(dim=-1)  # Hamming, [B,T]
                near1_correct += ((distance <= 1) & near_valid).sum().item()
                near2_correct += ((distance <= 2) & near_valid).sum().item()
                near_total += near_valid.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = (correct / total * 100) if total > 0 else 0.0
        root_acc = (root_correct / total * 100) if total > 0 else 0.0
        qual_acc = (qual_correct / total * 100) if total > 0 else 0.0
        near1_acc = (near1_correct / near_total * 100) if near_total > 0 else 0.0
        near2_acc = (near2_correct / near_total * 100) if near_total > 0 else 0.0
        if val_acc > peak_val_acc:
            peak_val_acc, peak_val_acc_epoch = val_acc, epoch
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                'val_acc': val_acc}, best_val_acc_path)
        n_unique_pred = len(qual_pred_counts)
        top_pred = qual_pred_counts.most_common(3)
        pred_summary = ", ".join(f"cls{c}:{n/total*100:.1f}%" for c, n in top_pred)
        print(f"   [diag] quality: {n_unique_pred} διαφορετικές κλάσεις προβλέφθηκαν | top: {pred_summary}")

        # ΝΕΟ: ΠΑΝΤΑ σώζουμε, ΑΝΕΞΑΡΤΗΤΑ από το αν βελτιώθηκε το val loss -
        # ΑΚΡΙΒΩΣ όπως `(curriculum_type == 'f2f') or (best_val_loss > val_loss)`
        # στο train_utils.py τους, όπου αυτή η συνθήκη είναι ΠΑΝΤΑ True για f2f.
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch}, best_model_path)  # dict, όχι μόνο state_dict - χρειάζεται για resume
        if avg_val_loss < best_val_loss_seen:
                best_val_loss_seen = avg_val_loss
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch},
               os.path.join(OUT_DIR, "best_val_transformer_model.pth"))
        status_msg = "💾 αποθηκεύτηκε (τελευταία εποχή, χωρίς early stopping)"

        clamp_note = f" | .g clamped: {n_clamped_this_epoch}x" if n_clamped_this_epoch > 0 else ""
        # ΝΕΟ, ΠΑΝΤΑ ενεργό διαγνωστικό: τυπώνουμε την ΠΡΑΓΜΑΤΙΚΗ τιμή (όχι μόνο
        # μέτρημα ενεργοποίησης) της .g παραμέτρου ΑΠΕΥΘΕΙΑΣ από το live model object,
        # ΑΜΕΣΩΣ μετά το clamp. Αν αυτό δείξει σταθερά <=10 σε ΚΑΘΕ εποχή αλλά το
        # ΤΕΛΙΚΟ αποθηκευμένο checkpoint δείχνει εκατομμύρια (Διαγνωστικό 3), αυτό θα
        # αποδείξει ότι το πρόβλημα ΔΕΝ είναι στο ίδιο το clamp - θα είναι κάπου
        # αλλού (π.χ. πιθανή εσωτερική αναπαραμετροποίηση του RMSNorm στο
        # x-transformers, όπου το named_parameters() ΔΕΝ επιστρέφει το ΙΔΙΟ tensor
        # που χρησιμοποιείται στην πραγματικότητα στον υπολογισμό forward()).
        g_debug = ""
        for param_name, param in model.named_parameters():
            if param_name.endswith('.g'):
                g_debug = f" | live .g max|.|: {param.abs().max().item():.4f}"
                break
        print(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% (Root: {root_acc:.1f}%, Qual: {qual_acc:.1f}%) | "
              f"Near(≤1/≤2 νότες): {near1_acc:.1f}%/{near2_acc:.1f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | {status_msg}{clamp_note}{g_debug}")

    # ΝΕΟ: ΧΩΡΙΣ early stopping, το loop ΠΑΝΤΑ τρέχει ΟΛΕΣ τις max_epochs -
    # ΑΚΡΙΒΩΣ όπως αυτοί (200 σταθερές εποχές για f2f, καμία διακοπή). Το model
    # object, ΩΣ ΕΧΕΙ εδώ, ΕΙΝΑΙ ήδη η τελευταία εποχή - ΔΕΝ χρειάζεται
    # reload από δίσκο (already saved to best_model_path κάθε εποχή, ούτως ή
    # άλλως, για εξωτερική διαθεσιμότητα/backup).
    print(f"\n📦 Ολοκληρώθηκαν και οι {max_epochs} εποχές (χωρίς early stopping, ΑΚΡΙΒΩΣ όπως το paper).")
    print(f"   ℹ️ Η ΚΑΛΥΤΕΡΗ Val Acc που φάνηκε σε ΟΠΟΙΟΔΗΠΟΤΕ epoch ήταν {peak_val_acc:.2f}% "
          f"(εποχή {peak_val_acc_epoch}) - μόνο πληροφοριακά, ΔΕΝ είναι αυτό που επιστρέφεται.")

    # ΕΠΙΣΤΡΕΦΟΥΜΕ ΚΑΙ τα δύο αυτά - το harmonize_song_iterative τα χρειάζεται
    # ΩΣ globals (ίδια σύμβαση με το trained_model). Αν δεν τα επιστρέφαμε εδώ,
    # θα χρειαζόταν να τα ξαναχτίσεις χειροκίνητα αργότερα - ίδιο πρόβλημα με
    # το PAD_PITCH_ID νωρίτερα σε αυτό το project.
    return model, root_qual_to_flat_table, MASK_CHORD_ID



# ABLATION: True = Κανόνες 1&2 ενεργοί, False = μόνο το βασικό chord-tone rule
# (ίδια συμπεριφορά με πριν τους Κανόνες 1&2). Άλλαξε σε False για να ελέγξεις αν
# η διαφορά ακρίβειας οφείλεται στο vocabulary cleanup ή στους δύο νέους κανόνες.
ENABLE_RULES_1_2 = os.environ.get('ENABLE_RULES_1_2','0') == '1'
# ABLATION: Κανόνας 3 (σταθερότητα αρμονικού ρυθμού) ξεχωριστό switch από τους 1&2,
# ώστε να μπορεί να αξιολογηθεί ανεξάρτητα.
ENABLE_RULE_3 = os.environ.get('ENABLE_RULE_3','1') == '1'
# ABLATION: Κανόνας 4 (κίνηση ρίζας). ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟΣ εξ ορισμού - τρία ξεχωριστά
# πειράματα επιβεβαίωσαν ότι συγκρούεται με τον Κανόνα 6 (και οι δύο διαμορφώνουν
# τη ρίζα, ο Κανόνας 6 το κάνει καλύτερα/πιο γενικευμένα). Άλλαξε σε True μόνο για
# νέο ablation test.
ENABLE_RULE_4 = os.environ.get('ENABLE_RULE_4','0') == '1'
# ABLATION: Κανόνας 5 (επίλυση leading tone), επίσης ξεχωριστό switch.
ENABLE_RULE_5 = os.environ.get('ENABLE_RULE_5','0') == '1'
# ABLATION: Κανόνας 6 (λειτουργική αρμονία T/S/D), επίσης ξεχωριστό switch.
ENABLE_RULE_6 = os.environ.get('ENABLE_RULE_6','1') == '1'
# ABLATION: Κανόνας 7 (τονική στο τέλος), επίσης ξεχωριστό switch.
ENABLE_RULE_7 = os.environ.get('ENABLE_RULE_7','1') == '1'
# ABLATION: Κανόνας 8 (ποιότητα ανά λειτουργία), επίσης ξεχωριστό switch.
ENABLE_RULE_8 = os.environ.get('ENABLE_RULE_8','1') == '1'
if os.path.exists(os.path.join(DATA_DIR, 'best_transformer_model.pth')):
    shutil.copy(os.path.join(DATA_DIR, 'best_transformer_model.pth'),
                os.path.join(OUT_DIR, 'best_transformer_model.pth'))
    
trained_model_iterative, root_qual_to_flat_table, MASK_CHORD_ID = train_model_concat_encoder(
    raw_songs, chord_to_idx, note_to_idx, chord_root_idx, chord_quality_idx,
    compatibility_matrix, NUM_NOTE_CLASSES, NUM_CHORD_CLASSES, NUM_ROOT_CLASSES, NUM_QUALITY_CLASSES,
    NUM_CHORD_CLASSES, flat_key_to_idx,
    enable_rules_1_2=ENABLE_RULES_1_2, enable_rule_3=ENABLE_RULE_3, enable_rule_4=ENABLE_RULE_4,
    enable_rule_5=ENABLE_RULE_5, enable_rule_6=ENABLE_RULE_6, enable_rule_7=ENABLE_RULE_7,
    enable_rule_8=ENABLE_RULE_8, enable_tonal_distance_loss=False, curriculum_total_epochs=100000
)

# =============================================================================
# --- 5. ΔΟΚΙΜΗ ΣΕ ΠΡΑΓΜΑΤΙΚΑ ΤΡΑΓΟΥΔΙΑ (listening test) ---
# =============================================================================
# Μέχρι τώρα κάθε απόφαση κρίθηκε ΜΟΝΟ από αριθμούς ακρίβειας πάνω σε ΜΕΜΟΝΩΜΕΝΕΣ
# νότες. Εδώ βλέπουμε πραγματικά τραγούδια: τι ΠΡΟΒΛΕΠΕΙ το μοντέλο σαν ολόκληρη
# αρμονική πρόοδο, δίπλα στο τι ΠΡΑΓΜΑΤΙΚΑ υπήρχε.
#
# ΔΥΟ ΣΗΜΑΝΤΙΚΕΣ ΔΙΕΥΚΡΙΝΙΣΕΙΣ:
# 1. Το split train/val έγινε σε επίπεδο CHUNK, όχι ολόκληρου τραγουδιού - άρα ένα
#    τυχαίο τραγούδι εδώ πιθανότατα έχει ΚΑΠΟΙΑ κομμάτια που "είδε" το μοντέλο στην
#    εκπαίδευση. Αυτό εδώ δεν είναι τεστ γενίκευσης (αυτό το κάνει ήδη η ακρίβεια) -
#    είναι έλεγχος αν οι προβλέψεις βγάζουν ΜΟΥΣΙΚΟ νόημα σαν ακολουθία.
# 2. Οι χορδές εμφανίζονται στον ΚΑΝΟΝΙΚΟΠΟΙΗΜΕΝΟ χώρο (C major / A minor), όχι στο
#    αρχικό key του τραγουδιού - οι σχέσεις/κινήσεις είναι ίδιες, απλά όχι το
#    "πραγματικό" όνομα χορδής για χρήση σε πραγματικό κομμάτι.

PC_TO_DISPLAY = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
# Τα PAD_*_ID μέσα στο train_model() είναι ΤΟΠΙΚΕΣ μεταβλητές - χάνονται μόλις η
# συνάρτηση επιστρέψει. Εδώ τα ΞΑΝΑ-ορίζουμε σε επίπεδο module (ίδιος ακριβώς
# τύπος: "ένα δείκτη πέρα από την τελευταία πραγματική κλάση"), ώστε το
# harmonize_song_padded() παρακάτω να έχει πρόσβαση σε αυτά.
PAD_PITCH_ID = NUM_PITCH_OCT_CLASSES
PAD_DURATION_ID = NUM_DURATION_CLASSES
PAD_BEAT_TYPE_ID = NUM_BEAT_TYPE_CLASSES
PAD_POSITION_ID = NUM_POSITION_CLASSES
PAD_INTERVAL_ID = NUM_INTERVAL_CLASSES

def prettify_pitch_name(name):
    if not name or name == '__UNKNOWN__':
        return '?'
    m = PITCH_NAME_RE.match(name)
    if not m:
        return name
    letter, accidental, octave = m.group(1), m.group(2), m.group(3)
    suffix = '#' if accidental == '_sharp_' else ('b' if accidental == '_flat_' else '')
    return f"{letter}{suffix}{octave}"

def chord_display_name(root_idx, qual_idx):
    root_pc = idx_to_root_pc.get(root_idx, 'UNKNOWN_ROOT')
    qual = idx_to_quality.get(qual_idx, '?')
    if root_pc == 'UNKNOWN_ROOT' or not isinstance(root_pc, int):
        return qual if qual in ('NC', 'N.C.') else f"N.C.({qual})"
    return f"{PC_TO_DISPLAY[root_pc]}{'' if qual == 'maj' else qual}"

def harmonize_song(melody_iris, chord_iris):
    """Τρέχει το μοντέλο σε ΟΛΟΚΛΗΡΟ το τραγούδι (χωρίς chunking - η
    PositionalEncoding υποστηρίζει ως 20000 θέσεις, πολύ πάνω από κάθε πραγματικό
    τραγούδι), και επιστρέφει προβλέψεις + πραγματικές χορδές για σύγκριση."""
    pitch_ids = [pitch_oct_to_idx.get(n, NUM_PITCH_OCT_CLASSES) for n in melody_iris]
    duration_ids = [duration_to_idx.get(n, NUM_DURATION_CLASSES) for n in melody_iris]
    beat_type_ids = [beat_type_to_idx.get(n, NUM_BEAT_TYPE_CLASSES) for n in melody_iris]
    position_ids = [position_to_idx.get(n, NUM_POSITION_CLASSES) for n in melody_iris]
    interval_ids = [interval_to_idx.get(n, NUM_INTERVAL_CLASSES) for n in melody_iris]
    beat_strength_vals = [note_beat_strength.get(n, 0.0) for n in melody_iris]

    t_pitch = torch.tensor([pitch_ids], dtype=torch.long, device=device)
    t_duration = torch.tensor([duration_ids], dtype=torch.long, device=device)
    t_beat_type = torch.tensor([beat_type_ids], dtype=torch.long, device=device)
    t_position = torch.tensor([position_ids], dtype=torch.long, device=device)
    t_interval = torch.tensor([interval_ids], dtype=torch.long, device=device)
    t_beat_strength = torch.tensor([beat_strength_vals], dtype=torch.float32, device=device)
    t_pad_mask = torch.zeros((1, len(melody_iris)), dtype=torch.bool, device=device)

    trained_model.eval()
    with torch.no_grad():
        root_logits, quality_logits = trained_model(t_pitch, t_duration, t_beat_type, t_position,
                                                      t_interval, t_beat_strength, t_pad_mask)
    pred_root = root_logits.argmax(dim=-1)[0].tolist()
    pred_qual = quality_logits.argmax(dim=-1)[0].tolist()

    true_root = [chord_root_idx[c] for c in chord_iris]
    true_qual = [chord_quality_idx[c] for c in chord_iris]

    n = len(melody_iris)
    root_hits = sum(1 for i in range(n) if pred_root[i] == true_root[i])
    qual_hits = sum(1 for i in range(n) if pred_qual[i] == true_qual[i])
    joint_hits = sum(1 for i in range(n) if pred_root[i] == true_root[i] and pred_qual[i] == true_qual[i])

    return {
        'melody_names': [prettify_pitch_name(transposed_pitch_name.get(n_)) for n_ in melody_iris],
        'pred_root': pred_root, 'pred_qual': pred_qual,
        'true_root': true_root, 'true_qual': true_qual,
        'root_acc': root_hits / n * 100, 'qual_acc': qual_hits / n * 100, 'joint_acc': joint_hits / n * 100,
        'root_hits': root_hits, 'qual_hits': qual_hits, 'joint_hits': joint_hits, 'n_notes': n,
    }

# =============================================================================
# ΝΕΟ (#1 από την έρευνα): "certain"-first iterative unmasking. ΞΕΧΩΡΙΣΤΗ
# συνάρτηση από το harmonize_song - ΔΕΝ το αγγίζει. Χρειάζεται trained_model_
# iterative, root_qual_to_flat_table, MASK_CHORD_ID ως globals (ΙΔΙΑ σύμβαση με
# το trained_model που ήδη χρησιμοποιεί το harmonize_song).
#
# num_steps=1 ΕΚΦΥΛΙΖΕΤΑΙ αυτόματα σε "μία πρόβλεψη, όλα μαζί" (καμία επανάληψη)
# - ΙΔΑΝΙΚΟ για να συγκρίνουμε "iterative" έναντι "single-pass" με το ΙΔΙΟ
# μοντέλο/checkpoint, αλλάζοντας ΜΟΝΟ αυτή την παράμετρο (βλ.
# iterative_reveal_test.py για την επαλήθευση αυτής της λογικής).
def harmonize_song_iterative(melody_iris, chord_iris, num_steps=10):
    pitch_ids = [pitch_oct_to_idx.get(n, NUM_PITCH_OCT_CLASSES) for n in melody_iris]
    duration_ids = [duration_to_idx.get(n, NUM_DURATION_CLASSES) for n in melody_iris]
    beat_type_ids = [beat_type_to_idx.get(n, NUM_BEAT_TYPE_CLASSES) for n in melody_iris]
    position_ids = [position_to_idx.get(n, NUM_POSITION_CLASSES) for n in melody_iris]
    interval_ids = [interval_to_idx.get(n, NUM_INTERVAL_CLASSES) for n in melody_iris]
    beat_strength_vals = [note_beat_strength.get(n, 0.0) for n in melody_iris]
    n = len(melody_iris)

    t_pitch = torch.tensor([pitch_ids], dtype=torch.long, device=device)
    t_duration = torch.tensor([duration_ids], dtype=torch.long, device=device)
    t_beat_type = torch.tensor([beat_type_ids], dtype=torch.long, device=device)
    t_position = torch.tensor([position_ids], dtype=torch.long, device=device)
    t_interval = torch.tensor([interval_ids], dtype=torch.long, device=device)
    t_beat_strength = torch.tensor([beat_strength_vals], dtype=torch.float32, device=device)
    t_pad_mask = torch.zeros((1, n), dtype=torch.bool, device=device)

    chord_context_ids = torch.full((1, n), MASK_CHORD_ID, dtype=torch.long, device=device)
    is_revealed = [False] * n
    final_pred_root = [None] * n
    final_pred_qual = [None] * n

    trained_model_iterative.eval()
    with torch.no_grad():
        for step in range(num_steps):
            n_still_masked = sum(1 for r in is_revealed if not r)
            if n_still_masked == 0:
                break
            steps_left = num_steps - step
            n_to_reveal = max(1, math.ceil(n_still_masked / steps_left))

            root_logits, quality_logits = trained_model_iterative(
                t_pitch, t_duration, t_beat_type, t_position, t_interval,
                t_beat_strength, chord_context_ids, t_pad_mask
            )
            root_probs = torch.softmax(root_logits, dim=-1)[0]
            quality_probs = torch.softmax(quality_logits, dim=-1)[0]
            root_conf, root_pred = root_probs.max(dim=-1)
            qual_conf, qual_pred = quality_probs.max(dim=-1)
            joint_conf = (root_conf * qual_conf).tolist()

            candidates = [(joint_conf[i], i) for i in range(n) if not is_revealed[i]]
            candidates.sort(key=lambda x: -x[0])
            to_reveal_now = candidates[:n_to_reveal]

            for _, i in to_reveal_now:
                r_i, q_i = root_pred[i].item(), qual_pred[i].item()
                final_pred_root[i] = r_i
                final_pred_qual[i] = q_i
                is_revealed[i] = True
                chord_context_ids[0, i] = root_qual_to_flat_table[r_i, q_i].item()

    true_root = [chord_root_idx[c] for c in chord_iris]
    true_qual = [chord_quality_idx[c] for c in chord_iris]
    root_hits = sum(1 for i in range(n) if final_pred_root[i] == true_root[i])
    qual_hits = sum(1 for i in range(n) if final_pred_qual[i] == true_qual[i])
    joint_hits = sum(1 for i in range(n) if final_pred_root[i] == true_root[i]
                      and final_pred_qual[i] == true_qual[i])

    return {
        'melody_names': [prettify_pitch_name(transposed_pitch_name.get(n_)) for n_ in melody_iris],
        'pred_root': final_pred_root, 'pred_qual': final_pred_qual,
        'true_root': true_root, 'true_qual': true_qual,
        'root_acc': root_hits / n * 100, 'qual_acc': qual_hits / n * 100, 'joint_acc': joint_hits / n * 100,
        'root_hits': root_hits, 'qual_hits': qual_hits, 'joint_hits': joint_hits, 'n_notes': n,
    }

def condense_progression(root_seq, qual_seq):
    """Συμπυκνώνει διαδοχικές ΙΔΙΕΣ χορδές σε ένα στοιχείο με πλήθος νοτών, σαν
    πραγματικό chord chart, αντί για επανάληψη ανά νότα."""
    if not root_seq:
        return ""
    parts = []
    cur_r, cur_q, count = root_seq[0], qual_seq[0], 1
    for r, q in zip(root_seq[1:], qual_seq[1:]):
        if r == cur_r and q == cur_q:
            count += 1
        else:
            parts.append(f"{chord_display_name(cur_r, cur_q)}({count})")
            cur_r, cur_q, count = r, q, 1
    parts.append(f"{chord_display_name(cur_r, cur_q)}({count})")
    return " - ".join(parts)

def harmonize_song_padded(melody_iris, pad_amount=80):
    """ΔΙΑΓΝΩΣΤΙΚΟ: τρέχει το ΙΔΙΟ τραγούδι αλλά μέσα σε ένα ΤΕΧΝΗΤΑ padded batch
    (πραγματικό padding στο τέλος + pad_mask=True εκεί) - ακριβώς όπως τρέχει η
    ΚΑΝΟΝΙΚΗ επικύρωση μέσα στο train_model, σε αντίθεση με το harmonize_song() που
    δεν έχει ΚΑΘΟΛΟΥ padding. Αν οι προβλέψεις στις ΠΡΑΓΜΑΤΙΚΕΣ θέσεις διαφέρουν
    σημαντικά μεταξύ των δύο, το πρόβλημα είναι στο ΙΔΙΟ το inference pathway (χωρίς
    padding), όχι στην ποιότητα του εκπαιδευμένου μοντέλου."""
    n_real = len(melody_iris)
    pitch_ids = [pitch_oct_to_idx.get(n, NUM_PITCH_OCT_CLASSES) for n in melody_iris] + [PAD_PITCH_ID] * pad_amount
    duration_ids = [duration_to_idx.get(n, NUM_DURATION_CLASSES) for n in melody_iris] + [PAD_DURATION_ID] * pad_amount
    beat_type_ids = [beat_type_to_idx.get(n, NUM_BEAT_TYPE_CLASSES) for n in melody_iris] + [PAD_BEAT_TYPE_ID] * pad_amount
    position_ids = [position_to_idx.get(n, NUM_POSITION_CLASSES) for n in melody_iris] + [PAD_POSITION_ID] * pad_amount
    interval_ids = [interval_to_idx.get(n, NUM_INTERVAL_CLASSES) for n in melody_iris] + [PAD_INTERVAL_ID] * pad_amount
    beat_strength_vals = [note_beat_strength.get(n, 0.0) for n in melody_iris] + [0.0] * pad_amount

    t_pitch = torch.tensor([pitch_ids], dtype=torch.long, device=device)
    t_duration = torch.tensor([duration_ids], dtype=torch.long, device=device)
    t_beat_type = torch.tensor([beat_type_ids], dtype=torch.long, device=device)
    t_position = torch.tensor([position_ids], dtype=torch.long, device=device)
    t_interval = torch.tensor([interval_ids], dtype=torch.long, device=device)
    t_beat_strength = torch.tensor([beat_strength_vals], dtype=torch.float32, device=device)
    t_pad_mask = torch.zeros((1, n_real + pad_amount), dtype=torch.bool, device=device)
    t_pad_mask[0, n_real:] = True  # ΠΡΑΓΜΑΤΙΚΟ padding στο τέλος, όπως στην κανονική επικύρωση

    trained_model.eval()
    with torch.no_grad():
        root_logits, quality_logits = trained_model(t_pitch, t_duration, t_beat_type, t_position,
                                                      t_interval, t_beat_strength, t_pad_mask)
    pred_root = root_logits.argmax(dim=-1)[0, :n_real].tolist()  # ΜΟΝΟ οι πραγματικές θέσεις
    pred_qual = quality_logits.argmax(dim=-1)[0, :n_real].tolist()
    return pred_root, pred_qual


# =============================================================================
# --- LISTENING TEST για το iterative μοντέλο, ΙΔΙΟ μοτίβο με τα προηγούμενα
# πειράματα, αλλά ΔΙΠΛΟ: μία φορά single-pass (num_steps=1), μία φορά iterative
# (num_steps=ITERATIVE_STEPS) - ΙΔΙΟ μοντέλο/βάρη και στις δύο, ΜΟΝΟ η στρατηγική
# inference αλλάζει. ΤΡΕΞΕ αυτό ΩΣ ΝΕΟ, ΤΕΛΕΥΤΑΙΟ κελί, ΜΕΤΑ το Step 5
# (train_model_iterative) ΚΑΙ το Cell 4 (ορισμοί prettify_pitch_name,
# chord_display_name, harmonize_song_iterative, condense_progression).
# =============================================================================
# ΣΗΜΕΙΩΣΗ: το "unpadded vs padded-batch" diagnostic ΔΕΝ ξαναχτίστηκε εδώ - ήταν
# έλεγχος συγκεκριμένου, ήδη-λυμένου ζητήματος στο ΑΡΧΙΚΟ μοντέλο, όχι ενεργή
# ανησυχία για το iterative μοντέλο. Παραλείπεται σκόπιμα, όχι από παράλειψη.

ITERATIVE_STEPS = 10

def run_listening_test(num_steps, label):
    print("\n" + "#" * 78)
    print(f"###  {label}  ###")
    print("#" * 78)

    random.seed(7)
    candidate_songs = [s for s in raw_songs if 20 <= len(s[0]) <= 80]
    demo_songs = random.sample(candidate_songs, min(5, len(candidate_songs)))

    print("\n" + "=" * 78)
    print(f"ΔΟΚΙΜΗ ΣΕ ΠΡΑΓΜΑΤΙΚΑ ΤΡΑΓΟΥΔΙΑ ({label})")
    print("=" * 78)
    for song_idx, (melody_iris, chord_iris) in enumerate(demo_songs):
        r = harmonize_song_iterative(melody_iris, chord_iris, num_steps=num_steps)
        print(f"\n--- Τραγούδι #{song_idx+1} ({len(melody_iris)} νότες) "
              f"| Ακρίβεια εδώ: Root={r['root_acc']:.0f}% Qual={r['qual_acc']:.0f}% Joint={r['joint_acc']:.0f}% ---")
        print("   Νότα -> Πρόβλεψη        | Πραγματικό          |")
        for i in range(len(melody_iris)):
            pred_str = chord_display_name(r['pred_root'][i], r['pred_qual'][i])
            true_str = chord_display_name(r['true_root'][i], r['true_qual'][i])
            match = "✓" if (r['pred_root'][i] == r['true_root'][i] and r['pred_qual'][i] == r['true_qual'][i]) else " "
            print(f"   {r['melody_names'][i]:6} -> {pred_str:18} | {true_str:18} | {match}")
        print(f"   Συμπυκνωμένη ΠΡΟΒΛΕΨΗ:    {condense_progression(r['pred_root'], r['pred_qual'])}")
        print(f"   Συμπυκνωμένο ΠΡΑΓΜΑΤΙΚΟ: {condense_progression(r['true_root'], r['true_qual'])}")

    print("\n" + "=" * 78)
    print(f"ΣΥΓΚΕΝΤΡΩΤΙΚΗ ΑΚΡΙΒΕΙΑ ΣΕ ΜΕΓΑΛΥΤΕΡΟ ΔΕΙΓΜΑ ({label}, 100 τραγούδια)")
    print("=" * 78)
    random.seed(11)
    larger_sample = random.sample(candidate_songs, min(100, len(candidate_songs)))

    total_root_hits, total_qual_hits, total_joint_hits, total_notes = 0, 0, 0, 0
    per_song_joint = []
    for melody_iris, chord_iris in larger_sample:
        r = harmonize_song_iterative(melody_iris, chord_iris, num_steps=num_steps)
        total_root_hits += r['root_hits']
        total_qual_hits += r['qual_hits']
        total_joint_hits += r['joint_hits']
        total_notes += r['n_notes']
        per_song_joint.append(r['joint_acc'])

    pooled_root = total_root_hits / total_notes * 100 if total_notes > 0 else 0.0
    pooled_qual = total_qual_hits / total_notes * 100 if total_notes > 0 else 0.0
    pooled_joint = total_joint_hits / total_notes * 100 if total_notes > 0 else 0.0
    mean_joint = sum(per_song_joint) / len(per_song_joint) if per_song_joint else 0.0
    std_joint = (sum((x - mean_joint) ** 2 for x in per_song_joint) / len(per_song_joint)) ** 0.5 if per_song_joint else 0.0

    print(f"   Δείγμα: {len(larger_sample)} τραγούδια, {total_notes} νότες συνολικά")
    print(f"   Συγκεντρωτική ακρίβεια: Root={pooled_root:.1f}%  Qual={pooled_qual:.1f}%  Joint={pooled_joint:.1f}%")
    print(f"   Μέσος όρος JOINT ανά τραγούδι: {mean_joint:.1f}% (τυπική απόκλιση: {std_joint:.1f}%)")
    print(f"   Εύρος JOINT ανά τραγούδι: min={min(per_song_joint):.1f}%  max={max(per_song_joint):.1f}%")

    return {'root': pooled_root, 'qual': pooled_qual, 'joint': pooled_joint}


single_pass_results = run_listening_test(num_steps=1, label="SINGLE-PASS (num_steps=1)")
iterative_results = run_listening_test(num_steps=ITERATIVE_STEPS, label=f"ITERATIVE (num_steps={ITERATIVE_STEPS})")

print("\n" + "=" * 78)
print("ΤΕΛΙΚΗ ΣΥΓΚΡΙΣΗ: single-pass έναντι iterative, ΙΔΙΟ checkpoint")
print("=" * 78)
print(f"{'':22} {'Root':>10} {'Quality':>10} {'Joint':>10}")
print(f"{'single-pass':22} {single_pass_results['root']:>9.1f}% {single_pass_results['qual']:>9.1f}% "
      f"{single_pass_results['joint']:>9.1f}%")
print(f"{'iterative':22} {iterative_results['root']:>9.1f}% {iterative_results['qual']:>9.1f}% "
      f"{iterative_results['joint']:>9.1f}%")
print(f"\nΔιαφορά (iterative - single): Root={iterative_results['root']-single_pass_results['root']:+.1f}pp, "
      f"Qual={iterative_results['qual']-single_pass_results['qual']:+.1f}pp, "
      f"Joint={iterative_results['joint']-single_pass_results['joint']:+.1f}pp")