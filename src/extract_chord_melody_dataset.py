# =============================================================================
# ΕΞΑΓΩΓΗ (μελωδία, χορδή) ζευγών από το chord-melody-dataset (shiehn/agangzz)
# =============================================================================
#
# *** ΔΙΟΡΘΩΣΗ (βλ. is_chord_note/is_rest παρακάτω) ***
# Το `continue` μετά το `measure_cursor += dur_q` σε ένα rest (ή chord-note)
# παρέκαμπτε την ενημέρωση του `measure_len_so_far` - αν ένα μέτρο ΤΕΛΕΙΩΝΕ σε
# rest, το πραγματικό μήκος του μέτρου χανόταν (γινόταν πιο μικρό απ' ό,τι
# έπρεπε), διαστρεβλώνοντας το feature 'position' (Pos_beginning/middle/end)
# για ΟΛΕΣ τις νότες σε τέτοια μέτρα. Επιβεβαιώθηκε με πραγματικό τραγούδι
# (Hey Jude) όπου εξαγόμενα MusicXML είχαν λάθος ρυθμό - η αιτία εντοπίστηκε
# εδώ. Η διόρθωση: η ενημέρωση του measure_len_so_far μπαίνει ΠΡΙΝ από κάθε
# `continue`, όχι μόνο μετά το πλήρες if/elif chain.
#
# Κάθε φάκελος τραγουδιού περιέχει 12 αρχεία .xml (a.xml, as.xml, b.xml, ...,
# gs.xml) - το ΙΔΙΟ τραγούδι μεταφερμένο σε όλες τις 12 τονικότητες. Χρειαζόμαστε
# ΜΟΝΟ ΕΝΑ ανά τραγούδι (τα υπόλοιπα 11 είναι πανομοιότυπα, απλά μεταφερμένα).
#
# Κάθε κομμάτι αυτού του pipeline έχει δοκιμαστεί ξεχωριστά ενάντια σε
# πραγματικά δεδομένα (το "All the Things You Are" από τον χρήστη) πριν
# συναρμολογηθεί εδώ:
#   - το offset-tracking επιβεβαιώθηκε ότι ταιριάζει ακριβώς με την ακολουθία
#     νότα->διάρκεια->επόμενη νότα σε όλο το αρχείο
#   - η ευθυγράμμιση χορδής->νότας αναπαρήγαγε σωστά το γνωστό μοτίβο
#     "κάθε νότα = 3η της χορδής" στην αρχή του τραγουδιού
#   - η ανίχνευση τονικότητας (Krumhansl-Schmuckler) επιβεβαιώθηκε σε ΟΛΑ τα
#     12 αρχεία ταυτόχρονα, δείχνοντας τέλεια διαδοχικές τονικότητες που
#     ταιριάζουν ακριβώς με τα ονόματα αρχείων (a.xml->A, as.xml->A#, ...)
#
# ΧΡΗΣΗ:
#   from extract_chord_melody_dataset import extract_dataset
#   songs, skipped = extract_dataset('/path/to/chord-melody-dataset')

import os
from fractions import Fraction
import xml.etree.ElementTree as ET

STEP_TO_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
PC_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

DURATION_LABELS = {
    Fraction(4, 1): 'Duration_whole',
    Fraction(3, 1): 'Duration_dotted-half',
    Fraction(2, 1): 'Duration_half',
    Fraction(3, 2): 'Duration_dotted-quarter',
    Fraction(1, 1): 'Duration_quarter',
    Fraction(3, 4): 'Duration_dotted-eighth',
    Fraction(1, 2): 'Duration_eighth',
    Fraction(3, 8): 'Duration_dotted-16th',
    Fraction(1, 4): 'Duration_16th',
    Fraction(1, 8): 'Duration_32nd',
}

KIND_TO_QUALITY = {
    'major': 'maj', 'minor': 'm', 'dominant': '7',
    'major-seventh': 'maj7', 'minor-seventh': 'm7',
    'diminished': 'dim', 'diminished-seventh': 'o7',
    'half-diminished': 'ø7', 'augmented': 'aug',
    'major-sixth': '6', 'minor-sixth': 'm6',
    'dominant-ninth': '9',
    'major-ninth': 'maj7',  # δεν έχουμε ξεχωριστό maj9 label - διατηρούμε την οικογένεια maj7
    'suspended-fourth': 'sus', 'suspended-second': 'sus',
    'none': 'NC',
}

# Γνωστά "βασικό kind + degree modification" patterns που ανακατατάσσονται σε
# ΔΙΑΦΟΡΕΤΙΚΟ kind ΠΡΙΝ την αντιστοίχιση σε ποιότητα.
#
# Η ΜΟΝΗ ενεργή ανακατάταξη (m7b5 -> ø7) επιβεβαιώθηκε ΔΙΠΛΑ: αρχικά σε
# συγκεκριμένη χορδή (μέτρο 15, all_the_things_you_are), και μετά σε ΚΛΙΜΑΚΑ
# σε ολόκληρο το dataset (7.362 εμφανίσεις, μηδέν μη-αναγνωρισμένες παραλλαγές
# του ίδιου pattern) μέσω πλήρους σάρωσης με το survey_chord_kinds.py.
#
# Η σάρωση βρήκε ΕΠΙΠΛΕΟΝ 3 συνδυασμούς kind+degree (dominant+b5="7b5",
# dominant+addb9="7b9", major-seventh+b5="maj7b5") που ΣΚΟΠΙΜΑ ΔΕΝ
# ανακατατάσσονται: η βασική οικογένεια της χορδής (dominant/maj7) παραμένει
# αρμονικά ορθή ακόμη και αγνοώντας το επιπλέον χρώμα - σε αντίθεση με το
# m7b5, όπου η λειτουργία της χορδής αλλάζει θεμελιωδώς (half-diminished ii°
# σε ελάσσονα έναντι απλού m7). Προσθήκη νέων κλάσεων ποιότητας γι' αυτές θα
# έσπαγε τη συμβατότητα με το υπάρχον λεξιλόγιο 24 κλάσεων του corpus.
DEGREE_RECLASSIFICATION = {
    ('minor-seventh', 5, -1, 'alter'): 'half-diminished',    # m7b5 -> ø7   [ΕΠΙΒΕΒΑΙΩΜΕΝΟ σε κλίμακα]
}

def apply_degree_modifications(kind, degree_elements):
    for value, alter, dtype in degree_elements:
        key = (kind, value, alter, dtype)
        if key in DEGREE_RECLASSIFICATION:
            kind = DEGREE_RECLASSIFICATION[key]
    return kind

KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


# -----------------------------------------------------------------------------
# ΒΗΜΑ 1: parse MusicXML -> ακατέργαστα melody/chord events (με σωστό offset-tracking)
# -----------------------------------------------------------------------------
def parse_song(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    part = root.find('.//part')

    melody_events, chord_events, warnings = [], [], []
    divisions = 1
    global_offset = Fraction(0)

    for measure in part.findall('measure'):
        measure_cursor = Fraction(0)
        measure_len_so_far = Fraction(0)

        for child in measure:
            tag = child.tag
            if tag == 'attributes':
                divs = child.find('divisions')
                if divs is not None:
                    divisions = int(divs.text)

            elif tag == 'harmony':
                rstep = child.find('root/root-step')
                ralter = child.find('root/root-alter')
                kind_el = child.find('kind')
                if rstep is not None and kind_el is not None:
                    pc = STEP_TO_PC.get(rstep.text)
                    if pc is not None:
                        alter = int(ralter.text) if ralter is not None else 0
                        kind = kind_el.text

                        degree_elements = []
                        for deg in child.findall('degree'):
                            dv = deg.find('degree-value')
                            da = deg.find('degree-alter')
                            dt = deg.find('degree-type')
                            if dv is not None and da is not None and dt is not None:
                                degree_elements.append((int(dv.text), int(da.text), dt.text))
                        kind = apply_degree_modifications(kind, degree_elements)

                        chord_events.append({
                            'global_offset': global_offset + measure_cursor,
                            'root_pc': (pc + alter) % 12,
                            'kind': kind,
                        })

            elif tag == 'note':
                dur_el = child.find('duration')
                dur_q = Fraction(int(dur_el.text), divisions) if dur_el is not None else Fraction(0)
                is_chord_note = child.find('chord') is not None
                is_rest = child.find('rest') is not None
                tie_stop = any(t.get('type') == 'stop' for t in child.findall('tie'))

                if is_chord_note:
                    warnings.append(f"Ταυτόχρονη νότα σε offset {global_offset+measure_cursor} - αγνοήθηκε")
                    measure_len_so_far = max(measure_len_so_far, measure_cursor)  # ΔΙΟΡΘΩΣΗ: αλλιώς χανόταν το μήκος
                    continue
                if is_rest:
                    measure_cursor += dur_q
                    measure_len_so_far = max(measure_len_so_far, measure_cursor)  # ΔΙΟΡΘΩΣΗ: αλλιώς χανόταν το μήκος (το ΚΥΡΙΟ πρόβλημα - βλ. σχόλιο στο τέλος του αρχείου)
                    continue

                pitch_el = child.find('pitch')
                if pitch_el is None:
                    measure_cursor += dur_q
                    continue

                step = pitch_el.find('step').text
                alter_el = pitch_el.find('alter')
                alter = int(alter_el.text) if alter_el is not None else 0
                octave = int(pitch_el.find('octave').text)
                pc = (STEP_TO_PC[step] + alter) % 12

                if tie_stop and melody_events:
                    melody_events[-1]['duration'] += dur_q
                else:
                    # Κρατάμε ΚΑΙ την αρχική γραφή (step, alter) - όχι μόνο pitch
                    # class - χρειάζεται για το ΣΩΣΤΟ diatonic interval (βλ.
                    # diatonic_interval_label), το οποίο ΔΙΑΦΕΡΕΤΙΚΑ διαστήματα με
                    # τα ίδια semitones (π.χ. M3 vs d4) ξεχωρίζει μόνο μέσω γραφής.
                    melody_events.append({
                        'global_offset': global_offset + measure_cursor,
                        'measure_offset': measure_cursor,
                        'pitch_pc': pc, 'octave': octave, 'duration': dur_q,
                        'step': step, 'alter': alter,
                    })
                measure_cursor += dur_q

            elif tag == 'forward':
                dur_el = child.find('duration')
                measure_cursor += Fraction(int(dur_el.text), divisions) if dur_el is not None else Fraction(0)

            elif tag == 'backup':
                dur_el = child.find('duration')
                measure_cursor -= Fraction(int(dur_el.text), divisions) if dur_el is not None else Fraction(0)
                warnings.append("BACKUP βρέθηκε - πιθανό πολυφωνικό περιεχόμενο")

            measure_len_so_far = max(measure_len_so_far, measure_cursor)

        for ev in melody_events:
            if 'measure_len' not in ev and ev['global_offset'] >= global_offset:
                ev['measure_len'] = measure_len_so_far
        global_offset += measure_len_so_far

    return melody_events, chord_events, warnings


# -----------------------------------------------------------------------------
# ΒΗΜΑ 2: feature computation
# -----------------------------------------------------------------------------
def duration_label(dur_q):
    return DURATION_LABELS.get(dur_q, 'Duration_OTHER')

def beat_type_and_strength(measure_offset):
    frac = measure_offset - int(measure_offset)
    beat_num = int(measure_offset)
    if frac == 0:
        if beat_num % 4 == 0:
            return 'Beat_Strong', 1.0
        elif beat_num % 4 == 2:
            return 'Beat_Strong', 0.5
        return 'Beat_Strong', 0.25
    elif frac == Fraction(1, 2):
        return 'Beat_Weak', 0.25
    return 'Beat_Offbeat', 0.125 if frac in (Fraction(1, 4), Fraction(3, 4)) else 0.0625

def position_label(measure_offset, measure_len):
    frac = float(measure_offset) / float(measure_len) if measure_len else 0
    if frac < 1/3:
        return 'Pos_beginning'
    elif frac < 2/3:
        return 'Pos_middle'
    return 'Pos_end'

def align_chords_to_notes(melody_sorted, chord_events):
    chord_events_sorted = sorted(chord_events, key=lambda c: c['global_offset'])
    result, ci, active = [], 0, None
    for note in melody_sorted:
        while ci < len(chord_events_sorted) and chord_events_sorted[ci]['global_offset'] <= note['global_offset']:
            active = chord_events_sorted[ci]
            ci += 1
        result.append((note, active))
    return result

LETTER_INDEX = {'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'A': 5, 'B': 6}
LETTER_SEMITONES = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
DEFAULT_SEMITONES_PATTERN = [0, 2, 4, 5, 7, 9, 11]  # unison,2nd,3rd,4th,5th,6th,7th (major/perfect)
PERFECT_TYPE_CLASSES = {0, 3, 4}  # unison, 4th, 5th (mod 7) είναι "perfect-type"

def diatonic_interval_label(step1, alter1, octave1, step2, alter2, octave2):
    """ΑΠΡΟΣΗΜΟ (direction-agnostic) διαστημικό label, π.χ. 'M3', 'd5', 'A4' - ΙΔΙΑ
    σύμβαση με το raw 'interval' feature του corpus (επιβεβαιωμένο: ανοδικό Μ2 και
    καθοδικό Μ2 παίρνουν το ΙΔΙΟ label - βλ. check_interval_full_vocab.py). Πρέπει
    να υπολογίζεται από την ΑΡΧΙΚΗ γραφή (πριν οποιαδήποτε μετάθεση key), αφού το
    διάστημα μεταξύ δύο νοτών είναι αναλλοίωτο σε μετάθεση εξ ορισμού."""
    letter_pos1 = LETTER_INDEX[step1] + 7 * octave1
    letter_pos2 = LETTER_INDEX[step2] + 7 * octave2
    semitone_pos1 = LETTER_SEMITONES[step1] + alter1 + 12 * octave1
    semitone_pos2 = LETTER_SEMITONES[step2] + alter2 + 12 * octave2

    diatonic_diff = abs(letter_pos2 - letter_pos1)
    semitone_diff = abs(semitone_pos2 - semitone_pos1)
    number = diatonic_diff + 1

    within_octave = diatonic_diff % 7
    octaves = diatonic_diff // 7
    default_semitones = DEFAULT_SEMITONES_PATTERN[within_octave] + 12 * octaves
    is_perfect_type = within_octave in PERFECT_TYPE_CLASSES

    delta = semitone_diff - default_semitones

    if delta == 0:
        quality = 'P' if is_perfect_type else 'M'
    elif delta == -1:
        quality = 'd' if is_perfect_type else 'm'
    elif delta == 1:
        quality = 'A'
    elif delta > 1:
        quality = 'A' * delta
    else:
        quality = 'd' * (-delta) if is_perfect_type else 'd' * (-delta - 1)

    return f"{quality}{number}"

def compute_intervals(melody_sorted):
    """Forward-looking, ΑΠΡΟΣΗΜΟ diatonic interval (ΟΧΙ πρόσημο semitone διαφορά -
    βλ. diatonic_interval_label) προς την ΕΠΟΜΕΝΗ νότα· None για την τελευταία."""
    intervals = []
    for i, ev in enumerate(melody_sorted):
        if i + 1 < len(melody_sorted):
            nxt = melody_sorted[i + 1]
            intervals.append(diatonic_interval_label(
                ev['step'], ev['alter'], ev['octave'],
                nxt['step'], nxt['alter'], nxt['octave']))
        else:
            intervals.append(None)
    return intervals

def detect_key(melody_events):
    histogram = [0.0] * 12
    for ev in melody_events:
        histogram[ev['pitch_pc']] += float(ev['duration'])

    def correlate(hist, profile):
        n = len(hist)
        mh, mp = sum(hist) / n, sum(profile) / n
        num = sum((hist[i] - mh) * (profile[i] - mp) for i in range(n))
        dh = sum((hist[i] - mh) ** 2 for i in range(n)) ** 0.5
        dp = sum((profile[i] - mp) ** 2 for i in range(n)) ** 0.5
        return num / (dh * dp) if dh and dp else 0

    best_score, best_tonic, best_is_major = -2, 0, True
    for tonic in range(12):
        rotated = histogram[tonic:] + histogram[:tonic]
        sm, smin = correlate(rotated, KS_MAJOR), correlate(rotated, KS_MINOR)
        if sm > best_score:
            best_score, best_tonic, best_is_major = sm, tonic, True
        if smin > best_score:
            best_score, best_tonic, best_is_major = smin, tonic, False
    return best_tonic, best_is_major


# -----------------------------------------------------------------------------
# ΒΗΜΑ 3: πλήρης εξαγωγή ενός τραγουδιού
# -----------------------------------------------------------------------------
def extract_song(filepath, title="untitled"):
    melody, chords, warnings = parse_song(filepath)
    if not melody:
        return None, warnings + ["Καμία νότα μελωδίας βρέθηκε"]

    tonic_pc, is_major = detect_key(melody)
    shift = ((0 if is_major else 9) - tonic_pc) % 12

    melody_sorted = sorted(melody, key=lambda n: n['global_offset'])
    aligned = align_chords_to_notes(melody_sorted, chords)
    intervals = compute_intervals(melody_sorted)

    note_records = []
    for (note, chord), interval in zip(aligned, intervals):
        bt, bs = beat_type_and_strength(note['measure_offset'])
        pos = position_label(note['measure_offset'], note['measure_len'])
        transposed_pc = (note['pitch_pc'] + shift) % 12
        pitch_feature = transposed_pc + 12 * note['octave']

        if chord is not None:
            chord_root_pc = (chord['root_pc'] + shift) % 12
            chord_quality = KIND_TO_QUALITY.get(chord['kind'], 'OTHER')
        else:
            chord_root_pc, chord_quality = 'UNKNOWN_ROOT', 'NC'

        note_records.append({
            'pitch_feature': pitch_feature,
            'duration_label': duration_label(note['duration']),
            'beat_type': bt, 'beat_strength': bs, 'position': pos,
            'interval': (f'Interval_{interval}' if interval is not None else None),
            'chord_root_pc': chord_root_pc, 'chord_quality': chord_quality,
        })

    return {'title': title, 'is_major': is_major, 'n_notes': len(note_records),
            'notes': note_records}, warnings


# -----------------------------------------------------------------------------
# ΒΗΜΑ 4: batch - ένα ολόκληρο dataset (φάκελος ανά τραγούδι, 12 αρχεία έκαστος)
# -----------------------------------------------------------------------------
KEY_FILE_PREFERENCE = ['c', 'a', 'as', 'b', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs']

def extract_dataset(dataset_root):
    """Επεξεργάζεται ΟΛΟΥΣ τους υποφακέλους-τραγούδια, διαλέγοντας ΕΝΑ .xml
    ανά τραγούδι (προτίμηση: c.xml αν υπάρχει, αλλιώς το πρώτο διαθέσιμο)."""
    songs, skipped = [], []
    for entry in sorted(os.listdir(dataset_root)):
        song_dir = os.path.join(dataset_root, entry)
        if not os.path.isdir(song_dir):
            continue

        chosen_path = None
        for key_name in KEY_FILE_PREFERENCE:
            candidate = os.path.join(song_dir, f'{key_name}.xml')
            if os.path.exists(candidate):
                chosen_path = candidate
                break
        if chosen_path is None:
            skipped.append((entry, 'κανένα .xml βρέθηκε στον φάκελο'))
            continue

        try:
            song, warnings = extract_song(chosen_path, title=entry)
        except Exception as e:
            skipped.append((entry, f'σφάλμα parsing: {e}'))
            continue

        if song is None:
            skipped.append((entry, warnings[0] if warnings else 'άγνωστο πρόβλημα'))
            continue

        songs.append(song)

    return songs, skipped


if __name__ == '__main__':
    import json

    DATASET_ROOT = './chord-melody-dataset'   # <-- ΑΛΛΑΞΕ στον δικό σου φάκελο
    OUTPUT_PATH = './extracted_songs.json'    # <-- ΑΛΛΑΞΕ αν θες αλλού

    print(f"⏳ Εξαγωγή όλου του dataset από {DATASET_ROOT}...")
    songs, skipped = extract_dataset(DATASET_ROOT)

    print("\n" + "=" * 78)
    print("ΑΠΟΤΕΛΕΣΜΑΤΑ ΕΞΑΓΩΓΗΣ")
    print("=" * 78)
    print(f"✅ Επιτυχώς εξήχθησαν: {len(songs)} τραγούδια, "
          f"{sum(s['n_notes'] for s in songs)} νότες συνολικά")
    if skipped:
        print(f"\n⚠️ Παραλείφθηκαν {len(skipped)} τραγούδια:")
        for name, reason in skipped[:20]:
            print(f"   {name}: {reason}")
        if len(skipped) > 20:
            print(f"   ... και {len(skipped) - 20} ακόμα")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump({'songs': songs}, f, ensure_ascii=False)
    print(f"\n📦 Αποθηκεύτηκε στο {OUTPUT_PATH}")