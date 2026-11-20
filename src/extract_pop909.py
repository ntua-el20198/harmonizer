import os
import struct
import glob
from fractions import Fraction
import json

try:
    import openpyxl
except ImportError:
    openpyxl = None

# =============================================================================
# extract_pop909.py
# =============================================================================
# Εξάγει melody+chord δεδομένα από το POP909 dataset (909 πλήρη pop τραγούδια,
# MIDI format, επαγγελματικά arranged - ΟΧΙ αποσπάσματα, σε αντίθεση με το
# HookTheory) στο ΙΔΙΟ σχήμα εξόδου με το extract_hooktheory.py, ώστε να μπει
# στο pipeline με τον ΙΔΙΟ ακριβώς μηχανισμό merge.
#
# ΓΙΑΤΙ δικός μας MIDI parser αντί για mido/pretty_midi: καμία εξωτερική
# βιβλιοθήκη δεν χρειάζεται (ίδιο πνεύμα με το υπόλοιπο έργο - βλ.
# extract_hooktheory.py που χρησιμοποιεί μόνο stdlib + δικά μας modules).
#
# ΣΗΜΑΝΤΙΚΗ ΠΑΡΑΤΗΡΗΣΗ ΓΙΑ ΤΗΝ ΠΟΙΟΤΗΤΑ ΧΟΡΔΩΝ: ελέγξαμε ΟΛΕΣ τις 124,805
# χορδές σε ολόκληρο το dataset - 15 από τις 16 διαφορετικές "ποιότητες" που
# εμφανίζονται αντιστοιχούν ΑΚΡΙΒΩΣ σε ήδη υπάρχουσες κλάσεις του project. Η
# ΜΟΝΗ εξαίρεση είναι 'sus4(b7)' (1,239 εμφανίσεις, ~1% του συνόλου) - ένα
# dominant 7sus4, το ΙΔΙΟ ακριβώς chord type που βρήκαμε να είναι η 2η
# μεγαλύτερη πηγή 'OTHER' στο HookTheory. Πέφτει σε 'OTHER' εδώ επίσης, με
# τον ΙΔΙΟ μηχανισμό (η canonicalize_quality() του κύριου pipeline θα το
# ταξινομήσει ήδη έτσι αυτόματα).
# =============================================================================

# --- Ίδιες συμβάσεις duration/beat_type/position/interval με το
# extract_hooktheory.py, για συνέπεια μεταξύ πηγών ---
DURATION_LABELS = {
    Fraction(4, 1): 'Duration_whole', Fraction(3, 1): 'Duration_dotted-half',
    Fraction(2, 1): 'Duration_half', Fraction(3, 2): 'Duration_dotted-quarter',
    Fraction(1, 1): 'Duration_quarter', Fraction(3, 4): 'Duration_dotted-eighth',
    Fraction(1, 2): 'Duration_eighth', Fraction(3, 8): 'Duration_dotted-16th',
    Fraction(1, 4): 'Duration_16th', Fraction(1, 8): 'Duration_32nd',
}
# ΣΗΜΕΙΩΣΗ: δοκιμάστηκαν επιπλέον κατηγορίες τριπλέτας εδώ, αλλά αφαιρέθηκαν -
# η προσέγγιση "στρογγυλοποίηση ΚΑΘΕ νότας ανεξάρτητα στην πλησιέστερη κατηγορία
# (δυαδική Ή τριπλέτας)" παράγει ΜΕΜΟΝΩΜΕΝΕΣ, ΑΣΥΝΔΕΤΕΣ "τριπλέτες" σε νότες
# που δεν σχηματίζουν πραγματική ομάδα-3 με τις γειτονικές τους - απλά θόρυβος
# από την ατελή χρονική ακρίβεια πραγματικής εκτέλεσης (MIDI performance, όχι
# quantized παρτιτούρα), ΟΧΙ πραγματικό swing/triplet feel. Σωστή ανίχνευση
# τριπλέτας απαιτεί συλλογιστική σε ΕΠΙΠΕΔΟ ΟΜΑΔΑΣ νοτών (ελέγχοντας αν αρκετές
# διαδοχικές νότες ΜΑΖΙ αθροίζουν σε καθαρό μήκος beat) - πολύ πιο σύνθετο
# πρόβλημα, δεν έχει υλοποιηθεί ακόμα.
def duration_label(dur_q):
    return DURATION_LABELS.get(dur_q, 'Duration_OTHER')

# ΝΕΟ: το POP909 MIDI είναι ΕΚΤΕΛΕΣΜΕΝΟ (performed arrangement) - ΟΧΙ αυστηρά
# quantized παρτιτούρα σαν το HookTheory/chord-melody-dataset. Οι πραγματικές
# διάρκειες έχουν φυσική μικρο-χρονική απόκλιση (π.χ. staccato/legato παίξιμο)
# που ΣΠΑΝΙΑ πέφτει ΑΚΡΙΒΩΣ σε μια από τις 10 γνωστές τιμές - το exact-match
# duration_label() παραπάνω έδινε 'Duration_OTHER' στο 83.6% των νοτών! Αντ'
# αυτού, στρογγυλοποιούμε στην ΠΛΗΣΙΕΣΤΕΡΗ γνωστή κατηγορία (κατά απόσταση),
# ΙΔΙΑ λογική με το πώς θα το έκανε ένας άνθρωπος διαβάζοντας το tempo.
def duration_label_nearest(dur_q):
    best_label, best_dist = 'Duration_OTHER', None
    for frac, label in DURATION_LABELS.items():
        dist = abs(dur_q - frac)
        if best_dist is None or dist < best_dist:
            best_dist, best_label = dist, label
    return best_label

def beat_type_and_strength(measure_offset):
    frac = measure_offset - int(measure_offset)
    beat_num = int(measure_offset)
    if frac == 0:
        if beat_num % 4 == 0: return 'Beat_Strong', 1.0
        elif beat_num % 4 == 2: return 'Beat_Strong', 0.5
        return 'Beat_Strong', 0.25
    elif frac == Fraction(1, 2):
        return 'Beat_Weak', 0.25
    return 'Beat_Offbeat', (0.125 if frac in (Fraction(1, 4), Fraction(3, 4)) else 0.0625)

def position_label(measure_offset, measure_len):
    frac = float(measure_offset) / float(measure_len) if measure_len else 0
    if frac < 1/3: return 'Pos_beginning'
    elif frac < 2/3: return 'Pos_middle'
    return 'Pos_end'

LETTER_INDEX = {'C':0,'D':1,'E':2,'F':3,'G':4,'A':5,'B':6}
LETTER_SEMITONES = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
DEFAULT_SEMITONES_PATTERN = [0,2,4,5,7,9,11]
PERFECT_TYPE_CLASSES = {0,3,4}
def diatonic_interval_label(step1, alter1, octave1, step2, alter2, octave2):
    letter_pos1 = LETTER_INDEX[step1] + 7*octave1
    letter_pos2 = LETTER_INDEX[step2] + 7*octave2
    semitone_pos1 = LETTER_SEMITONES[step1] + alter1 + 12*octave1
    semitone_pos2 = LETTER_SEMITONES[step2] + alter2 + 12*octave2
    diatonic_diff = abs(letter_pos2 - letter_pos1)
    semitone_diff = abs(semitone_pos2 - semitone_pos1)
    number = diatonic_diff + 1
    within_octave = diatonic_diff % 7
    octaves = diatonic_diff // 7
    default_semitones = DEFAULT_SEMITONES_PATTERN[within_octave] + 12*octaves
    is_perfect = within_octave in PERFECT_TYPE_CLASSES
    delta = semitone_diff - default_semitones
    if delta == 0: quality = 'P' if is_perfect else 'M'
    elif delta == -1: quality = 'd' if is_perfect else 'm'
    elif delta == 1: quality = 'A'
    elif delta > 1: quality = 'A' * delta
    else: quality = 'd'*(-delta) if is_perfect else 'd'*(-delta-1)
    return f"{quality}{number}"

# --- Diatonic scale builder + speller, ΙΔΙΟ με pitch_speller.py ---
LETTERS = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
LETTER_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
PC_TO_SHARP_LETTER_ALTER = [
    ('C', 0), ('C', 1), ('D', 0), ('D', 1), ('E', 0), ('F', 0),
    ('F', 1), ('G', 0), ('G', 1), ('A', 0), ('A', 1), ('B', 0),
]
def build_diatonic_scale(tonic_pitch_class, scale_degree_intervals):
    tonic_letter, tonic_alter = PC_TO_SHARP_LETTER_ALTER[tonic_pitch_class % 12]
    tonic_letter_idx = LETTERS.index(tonic_letter)
    degrees = [(tonic_letter, tonic_pitch_class % 12)]
    pc = tonic_pitch_class
    for i, interval in enumerate(scale_degree_intervals):
        pc = (pc + interval) % 12
        letter = LETTERS[(tonic_letter_idx + i + 1) % 7]
        degrees.append((letter, pc))
    return degrees
def spell_pitch_class(query_pc, diatonic_scale):
    candidates = []
    for letter, deg_pc in diatonic_scale:
        diff_to_degree = (query_pc - deg_pc + 6) % 12 - 6
        alter = (query_pc - LETTER_PC[letter] + 6) % 12 - 6
        candidates.append((abs(diff_to_degree), abs(alter), letter, alter))
    candidates.sort(key=lambda c: (c[0], c[1]))
    return candidates[0][2], candidates[0][3]

MAJOR_SCALE_INTERVALS = [2, 2, 1, 2, 2, 2, 1]
MINOR_SCALE_INTERVALS = [2, 1, 2, 2, 1, 2, 2]  # φυσική ελάσσονα

# --- Χορδική ποιότητα: μετάφραση από το mir_eval-style λεξιλόγιο του POP909
# ΑΠΕΥΘΕΙΑΣ στις 24 κλάσεις του project (ήδη κανονικά strings, ΟΧΙ raw) ---
QUALITY_TRANSLATION = {
    'maj': 'maj', 'min': 'm', 'sus2': 'sus2', 'min7': 'm7', 'maj7': 'maj7',
    'sus4': 'sus', '7': '7', 'maj6': '6', 'dim': 'dim', 'hdim7': 'ø7',
    'aug': 'aug', 'min6': 'm6', 'dim7': 'o7', 'minmaj7': 'mM7',
    'sus4(b7)': 'OTHER',  # dominant 7sus4 - δεν υπάρχει ακόμα στο λεξιλόγιο (βλ. σχόλιο στην κορυφή)
}

ROOT_LETTER_RE_CHARS = set('ABCDEFG')
def parse_root_pc(root_str):
    """'F#' -> 6, 'Bb' -> 10, 'C' -> 0, κλπ."""
    letter = root_str[0]
    pc = LETTER_SEMITONES[letter]
    for ch in root_str[1:]:
        if ch == '#':
            pc += 1
        elif ch == 'b':
            pc -= 1
    return pc % 12

# --- Ελάχιστος, χωρίς εξαρτήσεις, MIDI parser (μόνο ό,τι χρειαζόμαστε) ---
def read_var_len(data, i):
    value = 0
    while True:
        byte = data[i]; i += 1
        value = (value << 7) | (byte & 0x7F)
        if not (byte & 0x80):
            break
    return value, i

def parse_midi(path):
    """Επιστρέφει (division, tempo_map, tracks) όπου tracks είναι λίστα από
    (track_name, [(pitch, onset_tick, offset_tick), ...]) - ΜΟΝΟ note events."""
    with open(path, 'rb') as f:
        data = f.read()
    header_len = struct.unpack('>I', data[4:8])[0]
    fmt, ntracks, division = struct.unpack('>HHH', data[8:8+6])
    i = 8 + header_len
    tempo_map = []  # [(abs_tick, microsec_per_beat)]
    tracks = []
    for t in range(ntracks):
        track_len = struct.unpack('>I', data[i+4:i+8])[0]
        track_end = i + 8 + track_len
        j = i + 8
        abs_tick = 0
        track_name = None
        notes = []
        active_notes = {}  # pitch -> onset_tick (υποθέτουμε non-overlapping ίδιου pitch)
        running_status = None
        while j < track_end:
            delta, j = read_var_len(data, j)
            abs_tick += delta
            status = data[j]
            if status == 0xFF:
                meta_type = data[j+1]
                length, j2 = read_var_len(data, j+2)
                content = data[j2:j2+length]
                if meta_type == 0x03:
                    track_name = content.decode('latin-1')
                elif meta_type == 0x51:
                    tempo_map.append((abs_tick, int.from_bytes(content, 'big')))
                j = j2 + length
            elif status in (0xF0, 0xF7):
                length, j2 = read_var_len(data, j+1)
                j = j2 + length
            else:
                if status & 0x80:
                    running_status = status
                    j += 1
                else:
                    status = running_status
                event_type = status & 0xF0
                if event_type in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                    d1, d2 = data[j], data[j+1]
                    j += 2
                    if event_type == 0x90 and d2 > 0:
                        active_notes[d1] = abs_tick
                    elif (event_type == 0x80) or (event_type == 0x90 and d2 == 0):
                        if d1 in active_notes:
                            notes.append((d1, active_notes.pop(d1), abs_tick))
                elif event_type in (0xC0, 0xD0):
                    j += 1
                else:
                    j += 2
        notes.sort(key=lambda n: n[1])
        tracks.append((track_name, notes))
        i = track_end
    tempo_map.sort()
    if not tempo_map:
        tempo_map = [(0, 500000)]  # default 120bpm αν λείπει τελείως
    return division, tempo_map, tracks

def make_tick_to_sec(division, tempo_map):
    """Χτίζει μια συνάρτηση tick->δευτερόλεπτα, σεβόμενη ΟΛΕΣ τις αλλαγές
    tempo μέσα στο κομμάτι (ορισμένα τραγούδια έχουν πολλαπλές)."""
    breakpoints = []  # [(tick, sec_at_tick, microsec_per_beat)]
    sec_acc = 0.0
    prev_tick = 0
    prev_tempo = tempo_map[0][1]
    for tick, tempo in tempo_map:
        sec_acc += (tick - prev_tick) * prev_tempo / 1e6 / division
        breakpoints.append((tick, sec_acc, tempo))
        prev_tick, prev_tempo = tick, tempo
    def tick_to_sec(tick):
        seg = breakpoints[0]
        for bp in breakpoints:
            if bp[0] <= tick:
                seg = bp
            else:
                break
        bp_tick, bp_sec, bp_tempo = seg
        return bp_sec + (tick - bp_tick) * bp_tempo / 1e6 / division
    return tick_to_sec

def parse_chord_file(path):
    chords = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            start, end, label = float(parts[0]), float(parts[1]), parts[2]
            if label in ('N', 'X'):
                chords.append((start, end, 'UNKNOWN_ROOT', 'NC'))
                continue
            no_bass = label.split('/')[0]
            if ':' not in no_bass:
                continue
            root_str, quality_str = no_bass.split(':', 1)
            root_pc = parse_root_pc(root_str)
            quality = QUALITY_TRANSLATION.get(quality_str, 'OTHER')
            chords.append((start, end, root_pc, quality))
    return chords

def parse_beat_file(path):
    """Επιστρέφει (all_times, beat_times, downbeat_times), όλα ταξινομημένα."""
    all_times, beat_times, downbeat_times = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            t, is_beat, is_downbeat = float(parts[0]), float(parts[1]), float(parts[2])
            all_times.append(t)
            if is_beat >= 0.5:
                beat_times.append(t)
            if is_downbeat >= 0.5:
                downbeat_times.append(t)
    return sorted(all_times), sorted(beat_times), sorted(downbeat_times)

def parse_key_file(path):
    """Επιστρέφει (tonic_pc, mode) από το ΠΡΩΤΟ segment - απλοποίηση για
    τραγούδια με modulation (σπάνιο, βλ. σχόλιο)."""
    with open(path) as f:
        first_line = f.readline().strip()
    parts = first_line.split('\t')
    label = parts[2]
    root_str, mode_str = label.split(':')
    tonic_pc = parse_root_pc(root_str)
    mode = 'major' if mode_str == 'maj' else 'minor'
    return tonic_pc, mode

def compute_steps_per_beat(all_times, beat_times):
    """ΝΕΟ: μετράει ΕΜΠΕΙΡΙΚΑ πόσα raw grid steps αντιστοιχούν σε ΕΝΑ beat,
    ΑΝΤΙ να υποθέτουμε πάντα 2 (όγδοο). Επιβεβαιώσαμε ότι 29/909 τραγούδια
    (~3.2%) έχουν στην πραγματικότητα 3 (πιθανώς τριπλέτας-βασισμένο πλέγμα),
    όχι 2 - για εκείνα τα τραγούδια, η σταθερή υπόθεση "2" έδινε λάθος
    beat_type/position σε ΚΑΘΕ νότα του τραγουδιού."""
    if not beat_times:
        return 2
    ratio = len(all_times) / len(beat_times)
    return max(1, round(ratio))

def measure_offset_for_time(t, all_times, downbeat_times, steps_per_beat=2):
    """Υπολογίζει (measure_offset σε τεταρτημόρια, measure_len σε τεταρτημόρια)
    για τη χρονική στιγμή t, χρησιμοποιώντας το πλέγμα beat_midi.txt. Το
    steps_per_beat υπολογίζεται ΤΩΡΑ εμπειρικά ανά τραγούδι (βλ.
    compute_steps_per_beat) αντί να υποτίθεται σταθερό."""
    if not all_times or not downbeat_times:
        return Fraction(0), Fraction(4)
    # βρες το πιο πρόσφατο downbeat πριν (ή στο) t
    prev_downbeat_idx = 0
    for i, dt in enumerate(downbeat_times):
        if dt <= t:
            prev_downbeat_idx = i
        else:
            break
    bar_start = downbeat_times[prev_downbeat_idx]
    bar_end = downbeat_times[prev_downbeat_idx + 1] if prev_downbeat_idx + 1 < len(downbeat_times) else None
    # μέτρα πόσα βήματα πλέγματος (grid steps) από το bar_start μέχρι το t
    grid_in_bar = [gt for gt in all_times if gt >= bar_start and (bar_end is None or gt < bar_end)]
    if len(grid_in_bar) < 2:
        return Fraction(0), Fraction(4)
    steps_from_bar_start = 0
    for i in range(len(grid_in_bar) - 1):
        if grid_in_bar[i] <= t < grid_in_bar[i+1]:
            steps_from_bar_start = i
            break
    else:
        steps_from_bar_start = len(grid_in_bar) - 1
    measure_offset = Fraction(steps_from_bar_start, steps_per_beat)
    measure_len = Fraction(len(grid_in_bar), steps_per_beat)
    return measure_offset, measure_len


def extract_song(song_dir, song_id, title, artist):
    midi_path = os.path.join(song_dir, f"{song_id}.mid")
    chord_path = os.path.join(song_dir, "chord_midi.txt")
    beat_path = os.path.join(song_dir, "beat_midi.txt")
    key_path = os.path.join(song_dir, "key_audio.txt")
    if not all(os.path.exists(p) for p in (midi_path, chord_path, beat_path, key_path)):
        return None, 'missing_files'

    division, tempo_map, tracks = parse_midi(midi_path)
    melody_track = next((notes for name, notes in tracks if name == 'MELODY'), None)
    if not melody_track:
        return None, 'no_melody_track'
    tick_to_sec = make_tick_to_sec(division, tempo_map)

    tonic_pc, mode = parse_key_file(key_path)
    scale_intervals = MAJOR_SCALE_INTERVALS if mode == 'major' else MINOR_SCALE_INTERVALS
    diatonic_scale = build_diatonic_scale(tonic_pc, scale_intervals)

    all_times, beat_times, downbeat_times = parse_beat_file(beat_path)
    steps_per_beat = compute_steps_per_beat(all_times, beat_times)
    chords_sorted = sorted(parse_chord_file(chord_path), key=lambda c: c[0])

    notes = []
    for idx, (pitch, onset_tick, offset_tick) in enumerate(melody_track):
        onset_sec = tick_to_sec(onset_tick)
        pc = pitch % 12
        octave = pitch // 12
        letter, alter = spell_pitch_class(pc, diatonic_scale)

        dur_quarters = Fraction(offset_tick - onset_tick, division)
        dur_q_label = duration_label_nearest(dur_quarters)

        measure_offset, measure_len = measure_offset_for_time(onset_sec, all_times, downbeat_times, steps_per_beat)
        beat_type, beat_strength = beat_type_and_strength(measure_offset)
        position = position_label(measure_offset, measure_len)

        if idx + 1 < len(melody_track):
            n_pitch = melody_track[idx + 1][0]
            n_letter, n_alter = spell_pitch_class(n_pitch % 12, diatonic_scale)
            interval = diatonic_interval_label(letter, alter, octave, n_letter, n_alter, n_pitch // 12)
        else:
            interval = None

        active = None
        for c in chords_sorted:
            if c[0] <= onset_sec:
                if c[1] > onset_sec:
                    active = c
                    break
                active = c
            else:
                break
        if active is None:
            root_pc_out, quality_out = 'UNKNOWN_ROOT', 'NC'
        else:
            root_pc_out, quality_out = active[2], active[3]

        notes.append({
            'onset': onset_sec, 'letter': letter, 'alter': alter, 'octave': octave,
            'duration_label': dur_q_label, 'beat_type': beat_type,
            'position': position, 'beat_strength': beat_strength,
            'interval': f'Interval_{interval}' if interval else None,
            'chord_root_pc': root_pc_out, 'chord_quality': quality_out,
        })

    return {'notes': notes, 'n_notes': len(notes)}, 'ok'


if __name__ == '__main__':
    POP909_ROOT = './POP909-Dataset-master/POP909'   # <-- ΑΛΛΑΞΕ ΑΝ ΧΡΕΙΑΖΕΤΑΙ
    OUTPUT_PATH = './extracted_pop909.json'           # <-- ΑΛΛΑΞΕ ΑΝ ΧΡΕΙΑΖΕΤΑΙ

    titles, artists = {}, {}
    index_path = os.path.join(POP909_ROOT, 'index.xlsx')
    if openpyxl and os.path.exists(index_path):
        wb = openpyxl.load_workbook(index_path)
        ws = wb.active
        headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
        for row in ws.iter_rows(min_row=2, values_only=True):
            rec = dict(zip(headers, row))
            sid = str(int(rec['song_id'])).zfill(3)
            titles[sid] = rec.get('name', '?')
            artists[sid] = rec.get('artist', '?')
        print(f"✅ Φορτώθηκαν τίτλοι/καλλιτέχνες για {len(titles)} τραγούδια από index.xlsx")
    else:
        print("⚠️ Δεν βρέθηκε index.xlsx (ή λείπει openpyxl) - θα χρησιμοποιηθούν γενικά ονόματα.")

    song_dirs = sorted(glob.glob(os.path.join(POP909_ROOT, '[0-9][0-9][0-9]')))
    print(f"⏳ Βρέθηκαν {len(song_dirs)} φάκελοι τραγουδιών.")

    status_counts = {}
    extracted_songs = []
    for i, song_dir in enumerate(song_dirs):
        song_id = os.path.basename(song_dir)
        if i % 100 == 0:
            print(f"   ... {i}/{len(song_dirs)}")
        result, status = extract_song(song_dir, song_id,
                                        titles.get(song_id, f"POP909_{song_id}"),
                                        artists.get(song_id, '?'))
        status_counts[status] = status_counts.get(status, 0) + 1
        if result is None:
            continue
        extracted_songs.append({
            'id': song_id,
            'artist': artists.get(song_id, '?'),
            'title': titles.get(song_id, f"POP909_{song_id}"),
            'notes': result['notes'],
            'n_notes': result['n_notes'],
        })

    print("\n" + "=" * 78)
    print("ΑΠΟΤΕΛΕΣΜΑΤΑ ΕΞΑΓΩΓΗΣ")
    print("=" * 78)
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"   {status:25} {count:>5}")
    print(f"\n✅ Επιτυχώς εξήχθησαν: {len(extracted_songs)} τραγούδια, "
          f"{sum(s['n_notes'] for s in extracted_songs)} νότες συνολικά")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump({'songs': extracted_songs}, f, ensure_ascii=False)
    print(f"\n📦 Αποθηκεύτηκε στο {OUTPUT_PATH}")
