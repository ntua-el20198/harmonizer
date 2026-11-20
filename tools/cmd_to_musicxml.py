import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction
from xml.etree import ElementTree as ET
from xml.dom import minidom

from extract_chord_melody_dataset import (
    parse_song, detect_key, align_chords_to_notes, KIND_TO_QUALITY,
    KEY_FILE_PREFERENCE,
)

# =============================================================================
# cmd_to_musicxml.py
# =============================================================================
# Εξάγει ΕΝΑ τραγούδι του chord-melody-dataset σε πραγματικό MusicXML - ΜΕΤΑ
# την ΙΔΙΑ μετάθεση τονικότητας (shift) που εφαρμόζει το extract_chord_
# melody_dataset.py, ώστε αυτό που βλέπεις να είναι ΑΚΡΙΒΩΣ ό,τι «βλέπει» το
# μοντέλο στο extracted_songs.json - ΟΧΙ το ωμό, αμετάθετο περιεχόμενο του
# πρωτότυπου .xml αρχείου.
#
# ΞΑΝΑΧΡΗΣΙΜΟΠΟΙΕΙ ΑΠΕΥΘΕΙΑΣ (import, όχι re-implementation) τις ήδη
# επιβεβαιωμένες συναρτήσεις parse_song/detect_key/align_chords_to_notes -
# ώστε η μετάθεση/ευθυγράμμιση εδώ να ταιριάζει ΑΚΡΙΒΩΣ με αυτή που παρήγαγε
# το πραγματικό training corpus, χωρίς κίνδυνο απόκλισης σε δεύτερη υλοποίηση.
#
# ΓΝΩΣΤΗ ΙΔΙΟΡΡΥΘΜΙΑ (ΚΛΗΡΟΝΟΜΗΜΕΝΗ από το ίδιο το training pipeline, ΟΧΙ
# bug εδώ): η μετάθεση αλλάζει μόνο το pitch class (0-11) - το octave ΔΕΝ
# διορθώνεται για wraparound. Μια νότα κοντά σε όριο οκτάβας μπορεί να
# εμφανιστεί με ασυνήθιστο άλμα μετά τη μετάθεση - αυτό είναι ΑΚΡΙΒΩΣ ό,τι
# βλέπει και το μοντέλο, όχι κάτι που «διορθώθηκε» εδώ.
#
# ⚠️ ΔΕΝ έχω πραγματικό αρχείο .xml του dataset για να το δοκιμάσω άμεσα σε
# αυτό το session - γραμμένο με προσοχή, ΞΑΝΑΧΡΗΣΙΜΟΠΟΙΩΝΤΑΣ τις ήδη
# επιβεβαιωμένες συναρτήσεις, αλλά δοκίμασέ το σε ένα πραγματικό τραγούδι
# πρώτα και πες μου αν βγει κάτι αναπάντεχο.
# =============================================================================

PC_TO_STEP_ALTER = [
    ('C', 0), ('C', 1), ('D', 0), ('D', 1), ('E', 0), ('F', 0),
    ('F', 1), ('G', 0), ('G', 1), ('A', 0), ('A', 1), ('B', 0),
]

def duration_to_type_and_dots(dur_quarters):
    candidates = [
        (Fraction(4,1), 'whole', 0), (Fraction(3,1), 'half', 1),
        (Fraction(2,1), 'half', 0), (Fraction(3,2), 'quarter', 1),
        (Fraction(1,1), 'quarter', 0), (Fraction(3,4), 'eighth', 1),
        (Fraction(1,2), 'eighth', 0), (Fraction(3,8), '16th', 1),
        (Fraction(1,4), '16th', 0), (Fraction(1,8), '32nd', 0),
    ]
    best = min(candidates, key=lambda c: abs(c[0] - dur_quarters))
    return best[1], best[2]

def export_song_to_musicxml(xml_path, title, output_path, divisions=24, verbose=True):
    melody_events, chord_events, warnings = parse_song(xml_path)
    if not melody_events:
        raise ValueError(f"Καμία νότα μελωδίας βρέθηκε στο {xml_path}")

    tonic_pc, is_major = detect_key(melody_events)
    shift = ((0 if is_major else 9) - tonic_pc) % 12

    melody_sorted = sorted(melody_events, key=lambda n: n['global_offset'])
    aligned = align_chords_to_notes(melody_sorted, chord_events)

    # --- Ομαδοποίηση σε μέτρα: το (global_offset - measure_offset) ενός
    # γεγονότος είναι ΤΟ ΣΤΑΘΕΡΟ αρχικό offset του δικού του μέτρου - όλα τα
    # γεγονότα με το ΙΔΙΟ αυτό αποτέλεσμα ανήκουν στο ΙΔΙΟ μέτρο. ---
    measures_dict = {}
    for note, chord in aligned:
        measure_start = note['global_offset'] - note['measure_offset']
        measures_dict.setdefault(measure_start, {'len': note['measure_len'], 'events': []})
        measures_dict[measure_start]['events'].append((note, chord))
    measure_starts = sorted(measures_dict.keys())

    root = ET.Element('score-partwise', version="3.1")
    work = ET.SubElement(root, 'work')
    ET.SubElement(work, 'work-title').text = title
    part_list = ET.SubElement(root, 'part-list')
    score_part = ET.SubElement(part_list, 'score-part', id="P1")
    ET.SubElement(score_part, 'part-name').text = "Melody"
    part = ET.SubElement(root, 'part', id="P1")

    def emit_note(measure_el, step, alter, octave, dur_quarters, tie_start=False, tie_stop=False):
        note_el = ET.SubElement(measure_el, 'note')
        pitch_el = ET.SubElement(note_el, 'pitch')
        ET.SubElement(pitch_el, 'step').text = step
        if alter != 0:
            ET.SubElement(pitch_el, 'alter').text = str(alter)
        ET.SubElement(pitch_el, 'octave').text = str(octave)
        ET.SubElement(note_el, 'duration').text = str(int(round(float(dur_quarters) * divisions)))
        if tie_stop:
            ET.SubElement(note_el, 'tie', type="stop")
        if tie_start:
            ET.SubElement(note_el, 'tie', type="start")
        type_str, dots = duration_to_type_and_dots(dur_quarters)
        ET.SubElement(note_el, 'type').text = type_str
        for _ in range(dots):
            ET.SubElement(note_el, 'dot')
        if tie_start or tie_stop:
            notations = ET.SubElement(note_el, 'notations')
            if tie_stop:
                ET.SubElement(notations, 'tied', type="stop")
            if tie_start:
                ET.SubElement(notations, 'tied', type="start")

    def emit_rest(measure_el, dur_quarters):
        rest_el = ET.SubElement(measure_el, 'note')
        ET.SubElement(rest_el, 'rest')
        ET.SubElement(rest_el, 'duration').text = str(int(round(float(dur_quarters) * divisions)))

    last_beats = None
    last_chord_symbol = None
    carry_over = None  # (remaining_duration, step, alter, octave) - νότα δεμένη (tie) από προηγούμενο μέτρο

    for m_i, m_start in enumerate(measure_starts):
        m_data = measures_dict[m_start]
        measure_len = m_data['len']
        beats = max(1, round(float(measure_len)))
        measure_el = ET.SubElement(part, 'measure', number=str(m_i + 1))

        if m_i == 0:
            attributes = ET.SubElement(measure_el, 'attributes')
            ET.SubElement(attributes, 'divisions').text = str(divisions)
            key_el = ET.SubElement(attributes, 'key')
            ET.SubElement(key_el, 'fifths').text = "0"  # ήδη κανονικοποιημένο σε C major/A minor
            time_el = ET.SubElement(attributes, 'time')
            ET.SubElement(time_el, 'beats').text = str(beats)
            ET.SubElement(time_el, 'beat-type').text = "4"
            clef = ET.SubElement(attributes, 'clef')
            ET.SubElement(clef, 'sign').text = "G"
            ET.SubElement(clef, 'line').text = "2"
            last_beats = beats
        elif beats != last_beats:
            attributes = ET.SubElement(measure_el, 'attributes')
            time_el = ET.SubElement(attributes, 'time')
            ET.SubElement(time_el, 'beats').text = str(beats)
            ET.SubElement(time_el, 'beat-type').text = "4"
            last_beats = beats

        cursor = Fraction(0)

        # --- Νότα δεμένη (tie) από το προηγούμενο μέτρο - μπαίνει ΠΡΩΤΗ ---
        if carry_over is not None:
            dur, step, alter, octave = carry_over
            emit_note(measure_el, step, alter, octave, dur, tie_stop=True)
            cursor = dur
            carry_over = None

        for note, chord in m_data['events']:
            # --- ΚΕΝΟ (rest) ΠΡΙΝ από αυτή τη νότα, αν υπάρχει - αυτό ήταν το
            # ΚΥΡΙΟ πρόβλημα: τέτοια κενά ΔΕΝ εμφανίζονταν καθόλου πριν. ---
            if note['measure_offset'] > cursor:
                emit_rest(measure_el, note['measure_offset'] - cursor)
                cursor = note['measure_offset']

            transposed_pc = (note['pitch_pc'] + shift) % 12  # βλ. σχόλιο πάνω για octave wraparound
            octave = note['octave']
            step, alter = PC_TO_STEP_ALTER[transposed_pc]

            if chord is not None:
                chord_root_pc = (chord['root_pc'] + shift) % 12
                chord_quality = KIND_TO_QUALITY.get(chord['kind'], 'OTHER')
                chord_symbol = (chord_root_pc, chord_quality)
            else:
                chord_symbol = ('UNKNOWN_ROOT', 'NC')

            if chord_symbol != last_chord_symbol:
                harmony_el = ET.SubElement(measure_el, 'harmony')
                root_pc, quality = chord_symbol
                if quality == 'NC' or root_pc == 'UNKNOWN_ROOT':
                    kind_el = ET.SubElement(harmony_el, 'kind', text="N.C.")
                    kind_el.text = 'none'
                else:
                    root_el = ET.SubElement(harmony_el, 'root')
                    r_step, r_alter = PC_TO_STEP_ALTER[root_pc]
                    ET.SubElement(root_el, 'root-step').text = r_step
                    if r_alter != 0:
                        ET.SubElement(root_el, 'root-alter').text = str(r_alter)
                    kind_el = ET.SubElement(harmony_el, 'kind', text=quality)
                    kind_el.text = 'other'  # display-only text carries the real quality (βλ. πάνω)
                last_chord_symbol = chord_symbol

            note_end = note['measure_offset'] + note['duration']
            if note_end > measure_len:
                # --- Η νότα ξεπερνάει το μέτρο (π.χ. δεμένη πέρα από τη
                # γραμμή μέτρου) - ΣΠΑΜΕ σε δύο δεμένα (tied) κομμάτια, αντί
                # να αφήνουμε το άθροισμα του μέτρου να ΞΕΠΕΡΝΑ το δηλωμένο
                # μήκος (αυτό ήταν το ΔΕΥΤΕΡΟ πρόβλημα - τα "αρνητικά κενά"). ---
                first_part = measure_len - note['measure_offset']
                second_part = note_end - measure_len
                emit_note(measure_el, step, alter, octave, first_part, tie_start=True)
                cursor = measure_len
                carry_over = (second_part, step, alter, octave)
            else:
                emit_note(measure_el, step, alter, octave, note['duration'])
                cursor = note_end

        # --- Τελικό κενό (trailing rest) αν το μέτρο δεν γέμισε πλήρως ---
        if carry_over is None and cursor < measure_len:
            emit_rest(measure_el, measure_len - cursor)

    xml_str = ET.tostring(root, encoding='unicode')
    pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")
    doctype = '<?xml version="1.0" encoding="UTF-8"?>\n'
    pretty_body = pretty.split('\n', 1)[1]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(doctype + pretty_body)
    if verbose:
        print(f"✅ Αποθηκεύτηκε: {output_path} ({len(measure_starts)} μέτρα)")
        if warnings:
            print(f"   ⚠️ {len(warnings)} προειδοποιήσεις από το parsing: {warnings[:3]}{'...' if len(warnings) > 3 else ''}")


if __name__ == '__main__':
    DATASET_ROOT = './chord-melody-dataset'  # <-- ΑΛΛΑΞΕ
    OUTPUT_DIR = './musicxml_output'          # <-- ΑΛΛΑΞΕ ΑΝ ΘΕΣ ΑΛΛΟΝ ΦΑΚΕΛΟ

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    exported, skipped = [], []
    entries = sorted(os.listdir(DATASET_ROOT))
    print(f"⏳ Βρέθηκαν {len(entries)} στοιχεία στο {DATASET_ROOT} (φάκελοι + πιθανά άλλα αρχεία)...")

    for i, entry in enumerate(entries):
        song_dir = os.path.join(DATASET_ROOT, entry)
        if not os.path.isdir(song_dir):
            continue
        if i % 50 == 0:
            print(f"   ... {i}/{len(entries)}")

        chosen_path = None
        for key_name in KEY_FILE_PREFERENCE:
            candidate = os.path.join(song_dir, f'{key_name}.xml')
            if os.path.exists(candidate):
                chosen_path = candidate
                break
        if chosen_path is None:
            skipped.append((entry, 'κανένα .xml βρέθηκε στον φάκελο'))
            continue

        output_path = os.path.join(OUTPUT_DIR, f"{entry}.musicxml")
        try:
            export_song_to_musicxml(chosen_path, entry, output_path, verbose=False)
            exported.append(entry)
        except Exception as e:
            skipped.append((entry, f'σφάλμα: {e}'))

    print("\n" + "=" * 78)
    print(f"✅ Εξήχθησαν {len(exported)} τραγούδια στο {OUTPUT_DIR}/")
    if skipped:
        print(f"⚠️ Παραλείφθηκαν {len(skipped)} τραγούδια:")
        for name, reason in skipped[:20]:
            print(f"   {name}: {reason}")
        if len(skipped) > 20:
            print(f"   ... και {len(skipped) - 20} ακόμα")
