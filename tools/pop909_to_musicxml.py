import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction
from xml.etree import ElementTree as ET
from xml.dom import minidom

from extract_pop909 import (parse_midi, make_tick_to_sec, parse_chord_file,
                             parse_beat_file, parse_key_file, LETTER_SEMITONES)

# =============================================================================
# pop909_to_musicxml.py
# =============================================================================
# Εξάγει ΕΝΑ τραγούδι POP909 σε πραγματικό, έγκυρο MusicXML - μελωδία + χορδικά
# σύμβολα (<harmony>) - ώστε να ανοίγει κανονικά σε πρόγραμμα παρτιτούρας
# (MuseScore, Finale, κλπ) και να φαίνεται ΑΚΡΙΒΩΣ αυτό που εξήχθη στο
# extracted_pop909.json (ίδια μέτρα, ίδιες χορδές) - ΟΧΙ ξαναϋπολογισμένα από
# την αρχή, ΓΙΑ ΝΑ ΕΊΝΑΙ αξιόπιστο σαν "οπτικός έλεγχος" των δεδομένων.
#
# ΓΝΩΣΤΕΣ ΑΠΛΟΠΟΙΗΣΕΙΣ (για ένα εργαλείο spot-check, ΟΧΙ έκδοση παρτιτούρας):
# - Οι νότες που θα περνούσαν πέρα από ένα μέτρο ΚΟΒΟΝΤΑΙ στο τέλος του μέτρου
#   (όχι tie σε επόμενο μέτρο) - σπάνιο σε μια μελωδική γραμμή, αλλά υπάρχει.
# - Ο αριθμός beats/μέτρο υπολογίζεται εμπειρικά ανά μέτρο από το ΙΔΙΟ πλέγμα
#   beat_midi.txt που χρησιμοποιεί ΚΑΙ το extract_pop909.py - ώστε τα «μέτρα»
#   εδώ να ταιριάζουν ΑΚΡΙΒΩΣ με το πώς υπολογίστηκαν positio/beat_type εκεί.
# =============================================================================

PC_TO_STEP_ALTER = [
    ('C', 0), ('C', 1), ('D', 0), ('D', 1), ('E', 0), ('F', 0),
    ('F', 1), ('G', 0), ('G', 1), ('A', 0), ('A', 1), ('B', 0),
]

QUALITY_TO_MUSICXML_KIND = {
    'maj': 'major', 'm': 'minor', '7': 'dominant', 'm7': 'minor-seventh',
    'maj7': 'major-seventh', 'dim': 'diminished', 'aug': 'augmented',
    'sus': 'suspended-fourth', 'sus2': 'suspended-second', '6': 'major-sixth',
    'm6': 'minor-sixth', '9': 'dominant-ninth', 'M9': 'major-ninth',
    'm9': 'minor-ninth', 'o7': 'diminished-seventh', 'ø7': 'half-diminished',
    '7aug': 'augmented-seventh', 'mM7': 'major-minor', '11': 'dominant-11th',
    'm11': 'minor-11th', '13': 'dominant-13th', 'power': 'power',
}

STEP_LETTERS = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
LETTER_TO_STEP_IDX = {l: i for i, l in enumerate(STEP_LETTERS)}

def midi_pitch_to_step_octave(pitch, prefer_sharp=True):
    pc = pitch % 12
    octave = pitch // 12 - 1  # σύμβαση MIDI: 60 = C4 (πρότυπο για notation, ΟΧΙ το εσωτερικό octave=pitch//12 του project)
    step, alter = PC_TO_STEP_ALTER[pc]
    return step, alter, octave

def duration_to_type_and_dots(dur_quarters):
    """Επιλέγει ΕΝΑ εμφανιζόμενο type/dots ΚΑΙ το ΑΚΡΙΒΕΣ διάστημα σε
    τεταρτημόρια που του αντιστοιχεί - ώστε το δηλωμένο <duration> να
    ΣΥΜΦΩΝΕΙ ΠΑΝΤΑ με το type/dots.
    ΣΗΜΕΙΩΣΗ: δοκιμάστηκαν κατηγορίες τριπλέτας εδώ, αλλά αφαιρέθηκαν - η
    στρογγυλοποίηση ΚΑΘΕ νότας ανεξάρτητα παρήγαγε μεμονωμένες, ασύνδετες
    "τριπλέτες" σε νότες που δεν σχηματίζουν πραγματική ομάδα-3 (βλ. σχόλιο
    στο extract_pop909.py). Χρειάζεται σωστή ανίχνευση σε επίπεδο ομάδας
    νοτών, όχι per-note nearest-match - δεν έχει υλοποιηθεί ακόμα."""
    candidates = [
        (Fraction(4,1), 'whole', 0), (Fraction(3,1), 'half', 1),
        (Fraction(2,1), 'half', 0), (Fraction(3,2), 'quarter', 1),
        (Fraction(1,1), 'quarter', 0), (Fraction(3,4), 'eighth', 1),
        (Fraction(1,2), 'eighth', 0), (Fraction(3,8), '16th', 1),
        (Fraction(1,4), '16th', 0), (Fraction(1,8), '32nd', 0),
    ]
    best = min(candidates, key=lambda c: abs(c[0] - dur_quarters))
    return best[1], best[2], best[0]

def export_song_to_musicxml(song_dir, song_id, title, artist, output_path):
    midi_path = os.path.join(song_dir, f"{song_id}.mid")
    chord_path = os.path.join(song_dir, "chord_midi.txt")
    beat_path = os.path.join(song_dir, "beat_midi.txt")

    division, tempo_map, tracks = parse_midi(midi_path)
    melody_track = next((notes for name, notes in tracks if name == 'MELODY'), None)
    if not melody_track:
        raise ValueError("Δεν βρέθηκε MELODY track")
    tick_to_sec = make_tick_to_sec(division, tempo_map)
    all_times, beat_times, downbeat_times = parse_beat_file(beat_path)
    chords_sorted = sorted(parse_chord_file(chord_path), key=lambda c: c[0])

    def chord_at(t):
        active = None
        for c in chords_sorted:
            if c[0] <= t:
                active = c
            else:
                break
        return active

    # --- ΧΤΙΣΙΜΟ ΜΕΤΡΩΝ: ένα μέτρο ανά ζεύγος διαδοχικών downbeats, ΙΔΙΟ
    # σκεπτικό με το measure_offset_for_time() στο extract_pop909.py ---
    measures = []  # [(start_tick, end_tick, beats_per_measure)]
    def sec_to_tick(t_sec):
        # tempo_map έχει (tick, microsec_per_beat) breakpoints - ίδια λογική με
        # το make_tick_to_sec αλλά αντίστροφα
        sec_acc, prev_tick, prev_tempo = 0.0, 0, tempo_map[0][1]
        breakpoints = []
        for tick, tempo in tempo_map:
            sec_acc += (tick - prev_tick) * prev_tempo / 1e6 / division
            breakpoints.append((tick, sec_acc, tempo))
            prev_tick, prev_tempo = tick, tempo
        seg = breakpoints[0]
        for bp in breakpoints:
            if bp[1] <= t_sec:
                seg = bp
            else:
                break
        bp_tick, bp_sec, bp_tempo = seg
        return bp_tick + (t_sec - bp_sec) * 1e6 / bp_tempo * division

    downbeat_ticks = sorted(set(int(round(sec_to_tick(t))) for t in downbeat_times))
    song_end_tick = max(n[2] for n in melody_track)
    if not downbeat_ticks or downbeat_ticks[0] > 0:
        downbeat_ticks = [0] + downbeat_ticks
    if downbeat_ticks[-1] < song_end_tick:
        downbeat_ticks.append(song_end_tick)

    for i in range(len(downbeat_ticks) - 1):
        start_tick, end_tick = downbeat_ticks[i], downbeat_ticks[i + 1]
        quarters_in_measure = Fraction(end_tick - start_tick, division)
        beats = max(1, round(float(quarters_in_measure)))
        measures.append((start_tick, end_tick, beats))

    # --- Χτίσιμο MusicXML ---
    root = ET.Element('score-partwise', version="3.1")
    work = ET.SubElement(root, 'work')
    ET.SubElement(work, 'work-title').text = f"{title} ({artist}) [POP909 #{song_id}]"
    part_list = ET.SubElement(root, 'part-list')
    score_part = ET.SubElement(part_list, 'score-part', id="P1")
    ET.SubElement(score_part, 'part-name').text = "Melody"
    part = ET.SubElement(root, 'part', id="P1")

    last_beats = None
    last_chord_symbol = None
    note_idx = 0

    for m_i, (m_start, m_end, beats) in enumerate(measures):
        measure_el = ET.SubElement(part, 'measure', number=str(m_i + 1))
        if m_i == 0:
            attributes = ET.SubElement(measure_el, 'attributes')
            ET.SubElement(attributes, 'divisions').text = str(division)
            key_el = ET.SubElement(attributes, 'key')
            ET.SubElement(key_el, 'fifths').text = "0"  # C major/A minor - ήδη κανονικοποιημένο
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

        filled = 0
        measure_capacity = (m_end - m_start)
        while note_idx < len(melody_track) and melody_track[note_idx][1] < m_end:
            pitch, onset_tick, offset_tick = melody_track[note_idx]
            if onset_tick < m_start:
                note_idx += 1
                continue
            dur_ticks = min(offset_tick, m_end) - onset_tick  # ΚΟΨΙΜΟ στο τέλος του μέτρου (βλ. σχόλιο πάνω)
            if dur_ticks <= 0:
                note_idx += 1
                continue

            onset_sec = tick_to_sec(onset_tick)
            active = chord_at(onset_sec)
            if active is None:
                chord_symbol = None
            else:
                chord_symbol = (active[2], active[3])  # (root_pc ή 'UNKNOWN_ROOT', quality)

            if chord_symbol != last_chord_symbol:
                harmony_el = ET.SubElement(measure_el, 'harmony')
                root_pc, quality = chord_symbol if chord_symbol else ('UNKNOWN_ROOT', 'NC')
                if quality == 'NC' or root_pc == 'UNKNOWN_ROOT':
                    kind_el = ET.SubElement(harmony_el, 'kind', text="N.C.")
                    kind_el.text = 'none'
                else:
                    root_el = ET.SubElement(harmony_el, 'root')
                    step, alter = PC_TO_STEP_ALTER[root_pc]
                    ET.SubElement(root_el, 'root-step').text = step
                    if alter != 0:
                        ET.SubElement(root_el, 'root-alter').text = str(alter)
                    kind_val = QUALITY_TO_MUSICXML_KIND.get(quality, 'other')
                    kind_el = ET.SubElement(harmony_el, 'kind', text=quality)
                    kind_el.text = kind_val
                last_chord_symbol = chord_symbol

            note_el = ET.SubElement(measure_el, 'note')
            step, alter, octave = midi_pitch_to_step_octave(pitch)
            pitch_el = ET.SubElement(note_el, 'pitch')
            ET.SubElement(pitch_el, 'step').text = step
            if alter != 0:
                ET.SubElement(pitch_el, 'alter').text = str(alter)
            ET.SubElement(pitch_el, 'octave').text = str(octave)
            type_str, dots, snapped_quarters = duration_to_type_and_dots(Fraction(dur_ticks, division))
            snapped_ticks = int(round(float(snapped_quarters) * division))
            ET.SubElement(note_el, 'duration').text = str(snapped_ticks)
            ET.SubElement(note_el, 'type').text = type_str
            for _ in range(dots):
                ET.SubElement(note_el, 'dot')

            filled += snapped_ticks
            if offset_tick <= m_end:
                note_idx += 1
            else:
                # η νότα συνέχιζε πέρα από το μέτρο - κόπηκε (βλ. σχόλιο), προχωράμε
                note_idx += 1

        if filled < measure_capacity:
            rest_el = ET.SubElement(measure_el, 'note')
            ET.SubElement(rest_el, 'rest')
            ET.SubElement(rest_el, 'duration').text = str(measure_capacity - filled)

    xml_str = ET.tostring(root, encoding='unicode')
    pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")
    doctype = '<?xml version="1.0" encoding="UTF-8"?>\n'
    pretty_body = pretty.split('\n', 1)[1]  # αφαίρεση της <?xml...?> γραμμής του minidom
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(doctype + pretty_body)
    print(f"✅ Αποθηκεύτηκε: {output_path} ({len(measures)} μέτρα, {note_idx} νότες)")


if __name__ == '__main__':
    POP909_ROOT = './POP909'  # <-- ΑΛΛΑΞΕ ΑΝ ΧΡΕΙΑΖΕΤΑΙ
    import openpyxl
    wb = openpyxl.load_workbook(os.path.join(POP909_ROOT, 'index.xlsx'))
    ws = wb.active
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    titles, artists = {}, {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        rec = dict(zip(headers, row))
        sid = str(int(rec['song_id'])).zfill(3)
        titles[sid] = rec.get('name', '?')
        artists[sid] = rec.get('artist', '?')

    SONG_IDS_TO_EXPORT = ['001', '005', '050']  # <-- ΑΛΛΑΞΕ ΓΙΑ ΑΛΛΑ ΤΡΑΓΟΥΔΙΑ
    for sid in SONG_IDS_TO_EXPORT:
        export_song_to_musicxml(
            os.path.join(POP909_ROOT, sid), sid,
            titles.get(sid, f"POP909_{sid}"), artists.get(sid, '?'),
            f"pop909_{sid}.musicxml"
        )
