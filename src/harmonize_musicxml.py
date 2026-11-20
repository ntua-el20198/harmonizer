"""
harmonize_musicxml.py
======================
Παίρνει ένα MusicXML αρχείο με ΜΟΝΟ μελωδία (π.χ. εξαγωγή από MuseScore) και
επιστρέφει ένα νέο MusicXML με σύμβολα συγχορδιών προστιθέμενα, βασισμένα στο
εκπαιδευμένο νευρο-συμβολικό μοντέλο.

ΧΡΗΣΗ:
    python harmonize_musicxml.py --input melody.musicxml --output harmonized.musicxml \
        --checkpoint best_transformer_model.pth --export inference_export.pkl

ΑΠΑΙΤΕΙ: music21, torch (pip install music21 torch)
ΔΕΝ χρειάζεται πλέον x-transformers - το μοντέλο (HarmonizerConcatEncoder, concat-
encoder αρχιτεκτονική) χρησιμοποιεί ΚΑΝΟΝΙΚΟ nn.TransformerEncoder του PyTorch, όχι
x_transformers.Encoder (βλ. σημείωση #5 παρακάτω).
ΔΕΝ χρειάζεται το αρχικό corpus - μόνο το checkpoint + το inference_export.pkl
(βλ. export_for_inference.py - ΠΡΟΣΟΧΗ: το export script χρειάστηκε ενημέρωση για
τη νέα αρχιτεκτονική, βλ. σημείωση #5).

===============================================================================
ΣΗΜΑΝΤΙΚΕΣ ΠΑΡΑΔΟΧΕΣ (δεν έχουν επιβεβαιωθεί άμεσα σε πραγματικό αρχείο ακόμη -
ΔΟΚΙΜΑΣΕ και ΑΝΑΦΕΡΕ αν κάτι βγαίνει περίεργο):

1. "position" (Pos_beginning/Pos_middle/Pos_end): δεν έχουμε άμεση επιβεβαίωση
   τι ακριβώς σημαίνει στο αρχικό corpus. Η πιο λογική, φυσική ερμηνεία δεδομένου
   ότι το beat_type ήδη καλύπτει τη μετρική θέση ΜΕΣΑ στο μέτρο, είναι θέση ΜΕΣΑ
   ΣΤΟ ΜΕΤΡΟ σε επίπεδο "ολόκληρου μέτρου" (πρώτη νότα=beginning, τελευταία=end,
   ενδιάμεσες=middle). Αυτό υλοποιείται παρακάτω - αν τα αποτελέσματα δείχνουν
   συστηματικά περίεργα, αυτή είναι η πρώτη υποψία.

2. "interval": τα δείγματα τιμών (π.χ. "Interval_M3", ΟΧΙ "Interval_M-3") δείχνουν
   πιθανότατα ΜΗ-κατευθυντικά ονόματα διαστημάτων (music21 .name, όχι .directedName
   που θα έδειχνε πρόσημο για κατιούσα κίνηση). Υλοποιείται με .name παρακάτω.
   Το ίδιο το script ΕΛΕΓΧΕΙ αυτόματα το ποσοστό "άγνωστων" τιμών στο τέλος - αν
   είναι πολύ υψηλό, αυτή η παραδοχή πιθανότατα χρειάζεται αναθεώρηση σε .directedName.

3. Αντιστοίχιση quality -> chord symbol figure (π.χ. 'ø7' -> 'm7b5' για το music21):
   βασισμένη στις καθιερωμένες συμβάσεις chord symbol, ΔΕΝ έχει δοκιμαστεί άμεσα
   στον parser του music21. Δες QUALITY_TO_FIGURE_SUFFIX παρακάτω.

4. Ties/πολλαπλές φωνές ανά μέτρο: ΔΕΝ υποστηρίζονται ειδικά σε αυτή την πρώτη
   έκδοση - υποθέτουμε μονοφωνική μελωδία, μία φωνή ανά μέτρο. Παύλες (rests)
   παραλείπονται εντελώς από την ακολουθία που βλέπει το μοντέλο.

5. ΑΡΧΙΤΕΚΤΟΝΙΚΗ (ΕΝΗΜΕΡΩΣΗ): το checkpoint που δοκιμάζουμε τώρα προέρχεται από
   train_model_concat_encoder() / HarmonizerConcatEncoder - ΟΧΙ την παλιά
   HarmonizerTransformer (x_transformers-based) που αυτό το script υπέθετε αρχικά.
   Βασικές διαφορές που αντανακλώνται εδώ:
     - embed_dim=512 (όχι 256), num_layers=8 (όχι 4), plain nn.TransformerEncoder
       (όχι x_transformers.Encoder με σχετική θέση) + ΣΤΑΘΕΡΗ sinusoidal θέση.
     - Το forward χρειάζεται ΕΝΑ ΕΠΙΠΛΕΟΝ input: chord_context_ids. Το μοντέλο
       ΔΕΝ δουλεύει με μόνο μελωδία πλέον. Εδώ περνάμε ΠΛΗΡΩΣ MASKED context
       (mask_chord_id παντού) - ΑΚΡΙΒΩΣ το ίδιο pathway που χρησιμοποιεί το val
       loop στο train_harmonizer.py (val_chord_context_ids = πλήρως masked),
       άρα η ακρίβεια εδώ είναι άμεσα συγκρίσιμη με το αναφερόμενο Val Acc.
     - Checkpoint format: dict {'model_state_dict', 'optimizer_state_dict', 'epoch'},
       ΟΧΙ πλέον γυμνό state_dict.
     - Το inference_export.pkl ΧΡΕΙΑΖΕΤΑΙ ενημερωμένα πεδία (embedding_dim=512,
       num_layers=8, num_chord_classes, mask_chord_id, max_seq_len) - το ΠΑΛΙΟ
       export script ΔΕΝ τα περιέχει. Ξανά-τρέξε το ενημερωμένο export script
       (βλ. συνοδευτική σημείωση/μήνυμα) ΠΡΙΝ χρησιμοποιήσεις αυτό το αρχείο.
===============================================================================
"""

import argparse
import math
import pickle
import sys

import torch
import torch.nn as nn

try:
    from music21 import converter, note as m21note, harmony, interval as m21interval, stream
except ImportError:
    print("❌ Το music21 δεν είναι εγκατεστημένο. Τρέξε: pip install music21")
    sys.exit(1)


# =============================================================================
# --- Krumhansl-Schmuckler key detection (ΑΚΡΙΒΩΣ το ίδιο με το train_harmonizer.py,
#     ώστε η ανίχνευση key να είναι ΣΥΝΕΠΗΣ με αυτό που είδε το μοντέλο στην
#     εκπαίδευση - ΔΕΝ χρησιμοποιούμε το ενσωματωμένο key-finder του music21,
#     που μπορεί να διαφέρει σε λεπτομέρειες υλοποίησης). ---
# =============================================================================
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


def detect_key_verbose(pitch_classes, top_n=5):
    """ΙΔΙΟ αποτέλεσμα με detect_key, αλλά επιστρέφει ΚΑΙ το histogram ΚΑΙ τους
    top_n καλύτερους υποψήφιους (όχι μόνο τον νικητή) - για διάγνωση."""
    hist = [0.0] * 12
    for pc in pitch_classes:
        hist[pc] += 1.0
    candidates = []
    for tonic in range(12):
        rotated = [hist[(tonic + k) % 12] for k in range(12)]
        for mode, profile in (('major', KS_MAJOR_PROFILE), ('minor', KS_MINOR_PROFILE)):
            c = _pearson(rotated, profile)
            candidates.append((c, tonic, mode))
    candidates.sort(key=lambda x: -x[0])
    return hist, candidates[:top_n]


PC_TO_DISPLAY = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# =============================================================================
# --- Μοντέλο (ΑΚΡΙΒΩΣ ίδιο με το HarmonizerConcatEncoder του train_harmonizer.py -
#     concat-encoder αρχιτεκτονική, ΑΝΤΙΚΑΘΙΣΤΑ την παλιά HarmonizerTransformer
#     (x_transformers-based) που δεν είναι πλέον συμβατή με το τρέχον checkpoint). ---
# =============================================================================
def sinusoidal_positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


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
        pe = sinusoidal_positional_encoding(max_seq_len, embed_dim)
        self.register_buffer('shared_pos', pe)  # [1, max_seq_len, embed_dim] - ΙΔΙΑ σύμβαση με train script,
        # ΚΡΑΤΙΕΤΑΙ persistent=True (default) ώστε το κλειδί να υπάρχει στο state_dict -
        # ΑΠΑΡΑΙΤΗΤΟ για να ταιριάξει με τα κλειδιά του αποθηκευμένου checkpoint.
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

    def _positions_for(self, L, device):
        """Επιστρέφει [1, L, embed_dim] θεσιακή κωδικοποίηση. Αν L ταιριάζει μέσα
        στο ήδη-αποθηκευμένο buffer (max_seq_len - το όριο που είδε η εκπαίδευση),
        χρησιμοποιεί ΑΚΡΙΒΩΣ αυτό (ίδιες τιμές, ίδιο tensor object). Αν ένα πραγματικό
        MusicXML αρχείο έχει ΠΕΡΙΣΣΟΤΕΡΕΣ νότες από ό,τι είδε ΠΟΤΕ η εκπαίδευση (σπάνιο -
        max_seq_len ήταν βαθμονομημένο στο 99ο εκατοστημόριο*1.2 του corpus), υπολογίζει
        δυναμικά την επέκταση αντί να κόψει το τραγούδι ή να σκάσει με IndexError -
        ΙΔΙΟΣ ΑΚΡΙΒΩΣ τύπος, άρα οι τιμές στο κοινό εύρος ταυτίζονται με το buffer."""
        if L <= self.max_seq_len:
            return self.shared_pos[:, :L, :]
        return sinusoidal_positional_encoding(L, self.embed_dim).to(device)

    def forward(self, pitch_ids, duration_ids, beat_type_ids, position_ids, interval_ids,
                beat_strength, chord_context_ids, pad_mask=None):
        B, L = pitch_ids.shape
        melody_emb = (self.pitch_embedding(pitch_ids) + self.duration_embedding(duration_ids)
                      + self.beat_type_embedding(beat_type_ids) + self.position_embedding(position_ids)
                      + self.interval_embedding(interval_ids)
                      + self.beat_strength_proj(beat_strength.unsqueeze(-1)))
        harmony_emb = self.chord_context_embedding(chord_context_ids)
        full_seq = torch.cat([melody_emb, harmony_emb], dim=1)  # [B, 2L, D]
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


def build_root_qual_to_flat_table(flat_key_to_idx, num_root_classes, num_quality_classes, mask_chord_id):
    """Πίνακας [num_root, num_quality] -> flat chord index - ΙΔΙΟ με το ομώνυμο
    function στο train_harmonizer.py. Χρειάζεται ΜΟΝΟ αν χρησιμοποιήσεις το
    προαιρετικό --iterative_steps > 1 (βλ. run_inference_iterative παρακάτω)."""
    table = torch.full((num_root_classes, num_quality_classes), mask_chord_id, dtype=torch.long)
    for (r_idx, q_idx), flat_idx in flat_key_to_idx.items():
        table[r_idx, q_idx] = flat_idx
    return table


# =============================================================================
# --- Φόρτωση εξαγμένου vocab/config + checkpoint (ΧΩΡΙΣ το corpus) ---
# =============================================================================
def load_export(export_path):
    with open(export_path, 'rb') as f:
        return pickle.load(f)


def load_model(checkpoint_path, export, device):
    cfg = export['model_config']
    model = HarmonizerConcatEncoder(
        num_pitch_classes=cfg['num_pitch_classes'], pad_pitch_id=cfg['num_pitch_classes'],
        num_duration_classes=cfg['num_duration_classes'], pad_duration_id=cfg['num_duration_classes'],
        num_beat_type_classes=cfg['num_beat_type_classes'], pad_beat_type_id=cfg['num_beat_type_classes'],
        num_position_classes=cfg['num_position_classes'], pad_position_id=cfg['num_position_classes'],
        num_interval_classes=cfg['num_interval_classes'], pad_interval_id=cfg['num_interval_classes'],
        num_chord_classes=cfg['num_chord_classes'], mask_chord_id=cfg['mask_chord_id'],
        embed_dim=cfg['embedding_dim'], num_heads=cfg['num_heads'], num_layers=cfg['num_layers'],
        num_root_classes=cfg['num_root_classes'], num_quality_classes=cfg['num_quality_classes'],
        max_seq_len=cfg['max_seq_len'], dropout=cfg['dropout'],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_note = checkpoint.get('epoch')
        if epoch_note is not None:
            print(f"   (checkpoint από εποχή {epoch_note})")
    else:
        # Παλιάς-μορφής, γυμνό state_dict - πιθανότατα από ΔΙΑΦΟΡΕΤΙΚΗ αρχιτεκτονική
        # (HarmonizerTransformer/HarmonizerTransformerIterative), ΔΕΝ θα ταιριάξει με
        # τα κλειδιά της HarmonizerConcatEncoder. Το αφήνουμε να αποτύχει με το
        # φυσικό PyTorch key-mismatch error αντί να το κρύψουμε.
        model.load_state_dict(checkpoint)
    model.eval()

    root_qual_to_flat_table = None
    if 'flat_key_to_idx' in export:
        root_qual_to_flat_table = build_root_qual_to_flat_table(
            export['flat_key_to_idx'], cfg['num_root_classes'], cfg['num_quality_classes'],
            cfg['mask_chord_id']
        ).to(device)
    return model, cfg['mask_chord_id'], root_qual_to_flat_table


# =============================================================================
# --- Εξαγωγή χαρακτηριστικών από MusicXML ---
# =============================================================================
def extract_melody_notes(score):
    """Επιστρέφει μια επίπεδη λίστα music21 Note objects (ΧΩΡΙΣ rests), από το
    ΠΡΩΤΟ part της παρτιτούρας - υποθέτει μονοφωνική μελωδία."""
    parts = score.parts
    if len(parts) == 0:
        # Score χωρίς ρητά Parts (π.χ. απλό Stream) - χρησιμοποίησε όλο το score.
        flat = score.flatten()
    else:
        if len(parts) > 1:
            print(f"⚠️ Βρέθηκαν {len(parts)} parts στο αρχείο - χρησιμοποιείται ΜΟΝΟ το πρώτο "
                  f"(υποθέτουμε ότι είναι η μελωδία).")
        flat = parts[0].flatten()
    notes = [n for n in flat.notes if n.isNote]  # αγνοεί Chord objects και Rests
    return notes


def note_position_in_measure(note_obj, all_notes):
    """Pos_beginning / Pos_middle / Pos_end - ΠΑΡΑΔΟΧΗ: θέση μέσα στο ΜΕΤΡΟ.
    Βλ. σημείωση παραδοχών στην κορυφή του αρχείου."""
    measure_num = note_obj.measureNumber
    same_measure_notes = [n for n in all_notes if n.measureNumber == measure_num]
    idx = same_measure_notes.index(note_obj)
    if idx == 0:
        return 'Pos_beginning'
    elif idx == len(same_measure_notes) - 1:
        return 'Pos_end'
    else:
        return 'Pos_middle'


def note_beat_type(note_obj):
    """Beat_Strong / Beat_Weak / Beat_Offbeat.
    ΔΙΟΡΘΩΘΗΚΕ βάσει εμπειρικού ελέγχου στο πραγματικό corpus (τα δύο πρώτα
    ΔΕΝ διαχωρίζονται από το ΜΕΓΕΘΟΣ του beat_strength - επικαλύπτονται και τα
    δύο στην τιμή 0.25 - αλλά από το ΚΛΑΣΜΑΤΙΚΟ ΜΕΡΟΣ του note.beat):
      Strong  = ΟΠΟΙΟΔΗΠΟΤΕ ακέραιο beat (1, 2, 3, 4...) - ΟΧΙ μόνο το beat 1.
      Weak    = το μισό-του-beat ("and", π.χ. 1.5, 2.5).
      Offbeat = οτιδήποτε πιο λεπτομερές (π.χ. 1.25, 1.75, 16th-note θέσεις).
    """
    beat = note_obj.beat
    if beat is None:
        return 'Beat_Weak'
    frac = float(beat) % 1.0
    if frac == 0.0:
        return 'Beat_Strong'
    if frac == 0.5:
        return 'Beat_Weak'
    return 'Beat_Offbeat'


def extract_features(notes, export, manual_key=None):
    """Επιστρέφει: λίστα από dicts με raw values ΚΑΙ indices ανά νότα, μαζί με
    (tonic, mode, shift) από την ανίχνευση key."""
    raw_pcs = [n.pitch.pitchClass for n in notes]
    if manual_key is not None:
        tonic, mode = manual_key
        print(f"   Key ΔΟΘΗΚΕ χειροκίνητα: tonic pitch-class={tonic} ({PC_TO_DISPLAY[tonic]}), mode={mode}")
    else:
        hist, top_candidates = detect_key_verbose(raw_pcs)
        hist_str = ", ".join(f"{PC_TO_DISPLAY[i]}={hist[i]:.0f}" for i in range(12) if hist[i] > 0)
        print(f"   Ιστόγραμμα pitch-class (πόσες φορές εμφανίζεται κάθε φθόγγος): {hist_str}")
        print("   Κορυφαίοι υποψήφιοι key (συσχέτιση, tonic, mode):")
        for corr, cand_tonic, cand_mode in top_candidates:
            print(f"      {corr:.3f}  {PC_TO_DISPLAY[cand_tonic]:3} {cand_mode}")
        tonic, mode = top_candidates[0][1], top_candidates[0][2]
    shift = ((0 - tonic) % 12) if mode == 'major' else ((9 - tonic) % 12)
    print(f"   Ανιχνεύθηκε key: tonic pitch-class={tonic} ({PC_TO_DISPLAY[tonic]}), mode={mode} "
          f"-> μετάθεση={shift} ημιτόνια")

    pitch_vocab = export['pitch_value_vocab']
    duration_vocab = export['duration_value_vocab']
    beat_type_vocab = export['beat_type_value_vocab']
    position_vocab = export['position_value_vocab']
    interval_vocab = export['interval_value_vocab']

    features = []
    n_unknown_interval = 0
    n_unknown_pitch = 0
    for i, n in enumerate(notes):
        # --- pitch: μετατίθεται στο κανονικό key πριν το lookup ---
        # ΣΗΜΑΝΤΙΚΟ: η σύμβαση "απόλυτου semitone" στην εκπαίδευση είναι
        # pitch_class + 12*octave (ΟΧΙ το πρότυπο MIDI, όπου C4=60 -> εδώ C4
        # σημαίνει 0+12*4=48). Γι' αυτό χρησιμοποιούμε n.pitch.pitchClass +
        # 12*n.pitch.octave απευθείας, ΟΧΙ n.pitch.midi - το .midi θα έδινε
        # συστηματικά λάθος οκτάβα (κατά ένα, εξαιτίας της σύμβασης MIDI).
        absolute_training_convention = n.pitch.pitchClass + 12 * n.pitch.octave
        absolute_shifted = absolute_training_convention + shift
        transposed_octave, transposed_pc = divmod(absolute_shifted, 12)
        pitch_name_key = f"Pitch_{PC_TO_DISPLAY[transposed_pc].replace('#', '_sharp_')}{transposed_octave}"
        pitch_idx = pitch_vocab.get(pitch_name_key, len(pitch_vocab))
        if pitch_idx == len(pitch_vocab):
            n_unknown_pitch += 1

        # --- duration ---
        dur_key = f"Duration_{n.duration.type}"
        duration_idx = duration_vocab.get(dur_key, len(duration_vocab))

        # --- beat_type ---
        bt_key = note_beat_type(n)
        beat_type_idx = beat_type_vocab.get(bt_key, len(beat_type_vocab))

        # --- position ---
        pos_key = note_position_in_measure(n, notes)
        position_idx = position_vocab.get(pos_key, len(position_vocab))

        # --- interval (προς την ΕΠΟΜΕΝΗ νότα, όπως στην εκπαίδευση - βλ. σημείωση
        #     παραδοχών: μη-κατευθυντικό .name) ---
        if i < len(notes) - 1:
            try:
                iv = m21interval.Interval(n, notes[i + 1])
                iv_key = f"Interval_{iv.name}"
            except Exception:
                iv_key = None
        else:
            iv_key = None
        if iv_key is not None and iv_key in interval_vocab:
            interval_idx = interval_vocab[iv_key]
        else:
            interval_idx = len(interval_vocab)
            n_unknown_interval += 1

        # --- beat_strength: συνεχής τιμή, απευθείας από το music21 ---
        beat_strength_val = float(n.beatStrength) if n.beatStrength is not None else 0.0

        features.append({
            'pitch_idx': pitch_idx, 'duration_idx': duration_idx, 'beat_type_idx': beat_type_idx,
            'position_idx': position_idx, 'interval_idx': interval_idx,
            'beat_strength': beat_strength_val,
        })

    unknown_rate = n_unknown_interval / max(len(notes) - 1, 1) * 100
    print(f"   Διαστήματα εκτός λεξιλογίου: {unknown_rate:.1f}% "
          f"({'✅ φυσιολογικό' if unknown_rate < 15 else '⚠️ ΥΨΗΛΟ - πιθανή λάθος παραδοχή στο interval format, δες τις σημειώσεις στην κορυφή του αρχείου'})")
    pitch_unknown_rate = n_unknown_pitch / len(notes) * 100
    print(f"   Νότες εκτός λεξιλογίου pitch (πολύ ακραίες οκτάβες, σπάνιο): {pitch_unknown_rate:.1f}% "
          f"({'✅ φυσιολογικό' if pitch_unknown_rate < 5 else '⚠️ ΥΨΗΛΟ - πιθανό πρόβλημα στη σύμβαση οκτάβας'})")

    return features, tonic, mode, shift


# =============================================================================
# --- Inference ---
# =============================================================================
def run_inference(model, features, mask_chord_id, device):
    """Μία, πλήρως masked περνά - ΑΚΡΙΒΩΣ το ίδιο pathway με το validation loop
    στο train_harmonizer.py (val_chord_context_ids = πλήρως masked παντού), άρα
    η ακρίβεια εδώ είναι άμεσα συγκρίσιμη με το αναφερόμενο Val Acc των logs."""
    pitch_ids = torch.tensor([[f['pitch_idx'] for f in features]], dtype=torch.long, device=device)
    duration_ids = torch.tensor([[f['duration_idx'] for f in features]], dtype=torch.long, device=device)
    beat_type_ids = torch.tensor([[f['beat_type_idx'] for f in features]], dtype=torch.long, device=device)
    position_ids = torch.tensor([[f['position_idx'] for f in features]], dtype=torch.long, device=device)
    interval_ids = torch.tensor([[f['interval_idx'] for f in features]], dtype=torch.long, device=device)
    beat_strength = torch.tensor([[f['beat_strength'] for f in features]], dtype=torch.float32, device=device)
    pad_mask = torch.zeros((1, len(features)), dtype=torch.bool, device=device)
    chord_context_ids = torch.full((1, len(features)), mask_chord_id, dtype=torch.long, device=device)

    with torch.no_grad():
        root_logits, quality_logits = model(pitch_ids, duration_ids, beat_type_ids, position_ids,
                                             interval_ids, beat_strength, chord_context_ids, pad_mask)
    return root_logits[0], quality_logits[0]  # [T, num_classes] - ΩΜΑ logits, όχι ακόμη argmax


def run_inference_iterative(model, features, mask_chord_id, root_qual_to_flat_table, device, num_steps=10):
    """ΠΡΟΑΙΡΕΤΙΚΗ εναλλακτική - ΙΔΙΑ λογική με το harmonize_song_iterative στο
    train_harmonizer.py: αποκαλύπτει σταδιακά τις πιο σίγουρες προβλέψεις σαν
    context για τις υπόλοιπες, αντί για μία, πλήρως masked πρόβλεψη όλων μαζί.
    Χρειάζεται root_qual_to_flat_table (άρα flat_key_to_idx στο export). Επιστρέφει
    ΚΑΤΕΥΘΕΙΑΝ class indices (όχι logits) - ΔΕΝ συνδυάζεται με Viterbi smoothing
    παρακάτω, αφού η ίδια η διαδικασία ήδη ενσωματώνει context μεταξύ θέσεων."""
    if root_qual_to_flat_table is None:
        raise ValueError("Το export δεν περιέχει 'flat_key_to_idx' - χρειάζεται για iterative "
                          "inference. Ξανά-έξαγε με το ενημερωμένο export script.")
    n = len(features)
    pitch_ids = torch.tensor([[f['pitch_idx'] for f in features]], dtype=torch.long, device=device)
    duration_ids = torch.tensor([[f['duration_idx'] for f in features]], dtype=torch.long, device=device)
    beat_type_ids = torch.tensor([[f['beat_type_idx'] for f in features]], dtype=torch.long, device=device)
    position_ids = torch.tensor([[f['position_idx'] for f in features]], dtype=torch.long, device=device)
    interval_ids = torch.tensor([[f['interval_idx'] for f in features]], dtype=torch.long, device=device)
    beat_strength = torch.tensor([[f['beat_strength'] for f in features]], dtype=torch.float32, device=device)
    pad_mask = torch.zeros((1, n), dtype=torch.bool, device=device)
    chord_context_ids = torch.full((1, n), mask_chord_id, dtype=torch.long, device=device)

    is_revealed = [False] * n
    final_pred_root = [0] * n
    final_pred_qual = [0] * n
    with torch.no_grad():
        for step in range(num_steps):
            n_still_masked = sum(1 for r in is_revealed if not r)
            if n_still_masked == 0:
                break
            steps_left = num_steps - step
            n_to_reveal = max(1, math.ceil(n_still_masked / steps_left))
            root_logits, quality_logits = model(pitch_ids, duration_ids, beat_type_ids, position_ids,
                                                 interval_ids, beat_strength, chord_context_ids, pad_mask)
            root_probs = torch.softmax(root_logits, dim=-1)[0]
            quality_probs = torch.softmax(quality_logits, dim=-1)[0]
            root_conf, root_pred = root_probs.max(dim=-1)
            qual_conf, qual_pred = quality_probs.max(dim=-1)
            joint_conf = (root_conf * qual_conf).tolist()
            candidates = [(joint_conf[i], i) for i in range(n) if not is_revealed[i]]
            candidates.sort(key=lambda x: -x[0])
            for _, i in candidates[:n_to_reveal]:
                r_i, q_i = root_pred[i].item(), qual_pred[i].item()
                final_pred_root[i] = r_i
                final_pred_qual[i] = q_i
                is_revealed[i] = True
                chord_context_ids[0, i] = root_qual_to_flat_table[r_i, q_i].item()
    return final_pred_root, final_pred_qual


# =============================================================================
# --- Viterbi εξομάλυνση (αντί για ανεξάρτητο argmax ανά θέση) ---
# =============================================================================
# ΙΔΕΑ: το raw argmax ανά θέση αγνοεί ΕΝΤΕΛΩΣ ό,τι ήδη ξέρουμε για το πώς
# συμπεριφέρονται πραγματικές χορδές στο χρόνο (Κανόνας 3 - ΗΔΗ βαθμονομημένο,
# ΗΔΗ επιβεβαιωμένο ότι το μοντέλο το ακολουθεί σε επίπεδο ΣΥΝΟΛΙΚΟΥ ποσοστού).
# Εδώ ΕΠΑΝΑΧΡΗΣΙΜΟΠΟΙΟΥΜΕ το ΙΔΙΟ βαθμονομημένο P(ίδια) ανά beat_type σαν
# transition prior σε ΚΑΝΟΝΙΚΗ Viterbi αποκωδικοποίηση πάνω στα logits του
# μοντέλου - συνδυάζει την εμπιστοσύνη του μοντέλου ΑΝΑ ΘΕΣΗ με τη γνωστή τάση
# σταθερότητας σε ΟΛΟΚΛΗΡΗ την ακολουθία, αντί για ανεξάρτητη απόφαση ανά νότα.
# ΔΕΝ αλλάζει το εκπαιδευμένο μοντέλο - είναι ΜΕΤΑ-επεξεργασία στην έξοδο του.
#
# Οι τιμές είναι οι ΗΔΗ πολλαπλά επιβεβαιωμένες, σταθερές στατιστικές του corpus
# (Κανόνας 3, ίδιες σε κάθε training run/diagnostic σε αυτό το project).
REAL_P_SAME_BY_BEAT_TYPE = {'Beat_Strong': 0.628, 'Beat_Offbeat': 0.979, 'Beat_Weak': 0.971}


def viterbi_smooth(logits, beat_type_labels, default_p_same=0.8):
    """logits: [T, num_classes] tensor. beat_type_labels: λίστα μήκους T από
    strings ("Beat_Strong" κλπ). Επιστρέφει λίστα μήκους T από class indices."""
    T, num_classes = logits.shape
    log_probs = torch.log_softmax(logits, dim=-1).tolist()

    V = [log_probs[0][:]]
    backptr = [[0] * num_classes]
    for t in range(1, T):
        p_same = REAL_P_SAME_BY_BEAT_TYPE.get(beat_type_labels[t], default_p_same)
        log_p_same = math.log(max(p_same, 1e-6))
        log_p_diff = math.log(max((1.0 - p_same) / max(num_classes - 1, 1), 1e-6))
        prev_row = V[t - 1]
        row_scores, row_backptr = [0.0] * num_classes, [0] * num_classes
        for s in range(num_classes):
            best_score, best_prev = -1e18, 0
            for prev in range(num_classes):
                trans = log_p_same if prev == s else log_p_diff
                score = prev_row[prev] + trans
                if score > best_score:
                    best_score, best_prev = score, prev
            row_scores[s] = best_score + log_probs[t][s]
            row_backptr[s] = best_prev
        V.append(row_scores)
        backptr.append(row_backptr)

    best_last = max(range(num_classes), key=lambda s: V[T - 1][s])
    path = [0] * T
    path[T - 1] = best_last
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1][path[t + 1]]
    return path


def smooth_predictions(root_logits, quality_logits, features, export):
    """Εφαρμόζει Viterbi ΞΕΧΩΡΙΣΤΑ σε root και quality (απλοποίηση - ΟΧΙ πλήρες
    κοινό μοντέλο root+quality μαζί, αλλά συλλαμβάνει το μεγαλύτερο μέρος του
    οφέλους με πολύ λιγότερη πολυπλοκότητα)."""
    idx_to_beat_type = {v: k for k, v in export['beat_type_value_vocab'].items()}
    beat_type_labels = [idx_to_beat_type.get(f['beat_type_idx'], '__UNKNOWN__') for f in features]
    pred_root = viterbi_smooth(root_logits, beat_type_labels)
    pred_qual = viterbi_smooth(quality_logits, beat_type_labels)
    return pred_root, pred_qual


# =============================================================================
# --- Μετατροπή προβλέψεων σε chord symbols (ΣΤΟ ΑΡΧΙΚΟ key) ---
# =============================================================================
# ΠΑΡΑΔΟΧΗ (βλ. σημείωση στην κορυφή): αντιστοίχιση quality -> figure suffix
# ΣΥΜΒΑΤΟ με τον parser chord symbol του music21. Δεν έχει δοκιμαστεί άμεσα.
QUALITY_TO_FIGURE_SUFFIX = {
    'maj': '', 'm': 'm', '7': '7', 'm7': 'm7', 'maj7': 'maj7', 'dim': 'dim',
    '6': '6', '9': '9', 'ø7': 'm7b5', 'sus': 'sus4', 'm6': 'm6', 'o7': 'dim7',
    'aug': 'aug', '7aug': 'aug7', 'm9': 'm9', 'power': '5', '13': '13',
    'sus2': 'sus2', 'M9': 'maj9', 'm11': 'm11', 'mM7': 'mMaj7', '11': '11',
    'OTHER': '',  # άγνωστη ποιότητα -> εμφανίζεται σκέτη τριάδα σαν καλύτερη διαθέσιμη εικασία
}


def build_chord_symbol(root_pc_original_key, quality):
    if quality == 'NC' or root_pc_original_key is None:
        return harmony.NoChord()
    root_name = PC_TO_DISPLAY[root_pc_original_key]
    suffix = QUALITY_TO_FIGURE_SUFFIX.get(quality, '')
    figure = f"{root_name}{suffix}"
    try:
        return harmony.ChordSymbol(figure)
    except Exception:
        # Αν το music21 δεν αναγνωρίζει το figure, γύρνα τουλάχιστον τη σκέτη ρίζα
        # αντί να σκάσει ολόκληρο το script.
        return harmony.ChordSymbol(root_name)


def predictions_to_grouped_chords(pred_root, pred_qual, export, shift):
    """Μετατρέπει τις ανά-νότα προβλέψεις σε ομαδοποιημένες (συμπυκνωμένες) χορδές -
    ΜΙΑ chord symbol ανά αλλαγή, όχι μία ανά νότα, ΜΕΤΑΤΕΘΕΙΜΕΝΕΣ πίσω στο ΑΡΧΙΚΟ key."""
    idx_to_root_pc = export['idx_to_root_pc']
    idx_to_quality = export['idx_to_quality']

    groups = []  # [(start_note_idx, root_pc_original_key ή None, quality), ...]
    prev = None
    for i, (r_idx, q_idx) in enumerate(zip(pred_root, pred_qual)):
        root_pc_canonical = idx_to_root_pc.get(r_idx, 'UNKNOWN_ROOT')
        quality = idx_to_quality.get(q_idx, 'OTHER')
        if isinstance(root_pc_canonical, int):
            root_pc_original = (root_pc_canonical - shift) % 12
        else:
            root_pc_original = None
        current = (root_pc_original, quality)
        if current != prev:
            groups.append([i, root_pc_original, quality])
            prev = current
    return groups


# =============================================================================
# --- Εγγραφή στο τελικό MusicXML ---
# =============================================================================
def write_harmonized_musicxml(score, notes, groups, output_path):
    part = score.parts[0] if len(score.parts) > 0 else score
    for start_idx, root_pc_original, quality in groups:
        anchor_note = notes[start_idx]
        cs = build_chord_symbol(root_pc_original, quality)
        cs.offset = anchor_note.offset
        measure = anchor_note.getContextByClass(stream.Measure)
        if measure is not None:
            measure.insert(anchor_note.offset - measure.offset, cs)
        else:
            part.insert(anchor_note.offset, cs)
    score.write('musicxml', fp=output_path)
    print(f"✅ Αποθηκεύτηκε {output_path} ({len(groups)} σύμβολα συγχορδιών)")


# =============================================================================
# --- Main ---
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Προσθέτει συγχορδίες σε MusicXML μελωδία.")
    parser.add_argument('--input', required=True, help="Είσοδος: MusicXML με μόνο μελωδία")
    parser.add_argument('--output', required=True, help="Έξοδος: MusicXML με συγχορδίες")
    parser.add_argument('--checkpoint', required=True, help="best_transformer_model.pth")
    parser.add_argument('--export', required=True, help="inference_export.pkl")
    parser.add_argument('--key', default=None,
                         help="Προαιρετικό: παρακάμπτει την αυτόματη ανίχνευση key. "
                              "Μορφή 'C:major' ή 'F#:minor' (χρησιμοποίησε # για δίεση).")
    parser.add_argument('--debug', action='store_true',
                         help="Τυπώνει αναλυτικά τα χαρακτηριστικά και τις ωμές προβλέψεις ανά νότα.")
    parser.add_argument('--no_smoothing', action='store_true',
                         help="Απενεργοποιεί την Viterbi εξομάλυνση - επιστρέφει στο απλό ανά-θέση "
                              "argmax (χρήσιμο για ΣΥΓΚΡΙΣΗ πριν/μετά). Αγνοείται αν --iterative_steps > 1.")
    parser.add_argument('--iterative_steps', type=int, default=1,
                         help="1 (προεπιλογή) = μία πλήρως masked πρόβλεψη, ΙΔΙΟ pathway με το Val Acc "
                              "στα training logs. >1 = προοδευτική αποκάλυψη (βλ. harmonize_song_iterative "
                              "στο train_harmonizer.py) - χρειάζεται 'flat_key_to_idx' στο export.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Συσκευή: {device}")

    print("⏳ Φόρτωση εξαγμένου vocab/config...")
    export = load_export(args.export)

    print("⏳ Φόρτωση μοντέλου...")
    model, mask_chord_id, root_qual_to_flat_table = load_model(args.checkpoint, export, device)

    print(f"⏳ Ανάγνωση {args.input}...")
    score = converter.parse(args.input)
    notes = extract_melody_notes(score)
    print(f"   Βρέθηκαν {len(notes)} νότες (rests παραλείφθηκαν).")
    if len(notes) < 8:
        print("❌ Πολύ λίγες νότες για αξιόπιστη εναρμόνιση (χρειάζονται τουλάχιστον 8).")
        sys.exit(1)

    manual_key = None
    if args.key is not None:
        try:
            tonic_str, mode_str = args.key.split(':')
            tonic_idx = PC_TO_DISPLAY.index(tonic_str)
            manual_key = (tonic_idx, mode_str.strip().lower())
        except (ValueError, IndexError):
            print(f"❌ Δεν αναγνωρίστηκε το --key '{args.key}'. Χρησιμοποίησε μορφή όπως 'C:major' ή 'F#:minor'.")
            sys.exit(1)

    print("⏳ Εξαγωγή χαρακτηριστικών...")
    features, tonic, mode, shift = extract_features(notes, export, manual_key=manual_key)

    if args.debug:
        print("\n" + "=" * 78)
        print("DEBUG: ανά-νότα εξαγμένα χαρακτηριστικά (πριν το inference)")
        print("=" * 78)
        idx_to_duration = {v: k for k, v in export['duration_value_vocab'].items()}
        idx_to_beat_type = {v: k for k, v in export['beat_type_value_vocab'].items()}
        idx_to_position = {v: k for k, v in export['position_value_vocab'].items()}
        idx_to_pitch = {v: k for k, v in export['pitch_value_vocab'].items()}
        print(f"{'#':>3} {'γραμμένη νότα':14} {'lookup pitch':16} {'duration':16} "
              f"{'beat_type':14} {'position':14} {'beat_str':8}")
        for i, (n, f) in enumerate(zip(notes, features)):
            written = n.nameWithOctave
            lookup_pitch = idx_to_pitch.get(f['pitch_idx'], '??UNKNOWN??')
            dur = idx_to_duration.get(f['duration_idx'], '??UNKNOWN??')
            bt = idx_to_beat_type.get(f['beat_type_idx'], '??UNKNOWN??')
            pos = idx_to_position.get(f['position_idx'], '??UNKNOWN??')
            print(f"{i:>3} {written:14} {lookup_pitch:16} {dur:16} {bt:14} {pos:14} "
                  f"{f['beat_strength']:.3f}")

    print("⏳ Εκτέλεση μοντέλου...")
    if args.iterative_steps > 1:
        print(f"   (iterative, {args.iterative_steps} βήματα αποκάλυψης)")
        pred_root, pred_qual = run_inference_iterative(
            model, features, mask_chord_id, root_qual_to_flat_table, device,
            num_steps=args.iterative_steps
        )
        if args.no_smoothing is False:
            print("   ℹ️ Viterbi εξομάλυνση παραλείπεται - η iterative αποκάλυψη ήδη ενσωματώνει "
                  "context μεταξύ θέσεων.")
    else:
        root_logits, quality_logits = run_inference(model, features, mask_chord_id, device)
        if args.no_smoothing:
            pred_root = root_logits.argmax(dim=-1).tolist()
            pred_qual = quality_logits.argmax(dim=-1).tolist()
        else:
            print("⏳ Εξομάλυνση Viterbi (χρησιμοποιεί το ήδη βαθμονομημένο P(ίδια) ανά beat_type)...")
            pred_root, pred_qual = smooth_predictions(root_logits, quality_logits, features, export)

    if args.debug:
        idx_to_root_pc = export['idx_to_root_pc']
        idx_to_quality = export['idx_to_quality']
        print("\n" + "=" * 78)
        print("DEBUG: ανά-νότα ΩΜΕΣ προβλέψεις μοντέλου (πριν το group/μετάθεση πίσω)")
        print("=" * 78)
        for i, (r_idx, q_idx) in enumerate(zip(pred_root, pred_qual)):
            root_pc = idx_to_root_pc.get(r_idx, 'UNKNOWN_ROOT')
            root_name = PC_TO_DISPLAY[root_pc] if isinstance(root_pc, int) else str(root_pc)
            quality = idx_to_quality.get(q_idx, 'OTHER')
            print(f"{i:>3} {notes[i].nameWithOctave:8} -> canonical root={root_name:4} quality={quality}")

    print("⏳ Δημιουργία συμβόλων συγχορδιών...")
    groups = predictions_to_grouped_chords(pred_root, pred_qual, export, shift)

    print("⏳ Εγγραφή τελικού αρχείου...")
    write_harmonized_musicxml(score, notes, groups, args.output)


if __name__ == '__main__':
    main()
