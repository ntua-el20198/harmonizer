import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xml.etree import ElementTree as ET
from xml.dom import minidom
from fractions import Fraction

from extract_pop909 import parse_chord_file, parse_midi, make_tick_to_sec

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
STEP_TO_PC = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}

def add_chords_to_converted_musicxml(converted_path, chord_midi_path, midi_path, part_id, output_path):
    """ΔΙΟΡΘΩΜΕΝΗ έκδοση: ΔΕΝ εμπιστευόμαστε το tempo που δήλωσε ο converter
    (βρέθηκε ΛΑΘΟΣ - 120 αντί για το πραγματικό 102 BPM στο τραγούδι 034,
    προκαλώντας συστηματική λάθος ευθυγράμμιση χορδών). Αντ' αυτού,
    ΤΑΙΡΙΑΖΟΥΜΕ κάθε νότα του converted αρχείου (με τη ΣΕΙΡΑ/ΤΟΝΙΚΟ ύψος) με
    την αντίστοιχη νότα από το ΔΙΚΟ ΜΑΣ, ήδη επιβεβαιωμένο raw MIDI parse, και
    παίρνουμε το ΠΡΑΓΜΑΤΙΚΟ onset_sec από ΕΚΕΙ - χειριζόμενοι τη διάσπαση
    (tie-split) μακρών νοτών σε δύο κομμάτια από τον converter."""
    tree = ET.parse(converted_path)
    root = tree.getroot()

    part_list = root.find('part-list')
    for score_part in list(part_list.findall('score-part')):
        if score_part.get('id') != part_id:
            part_list.remove(score_part)
    for part in list(root.findall('part')):
        if part.get('id') != part_id:
            root.remove(part)
    target_part = root.find(f"part[@id='{part_id}']")
    divisions = int(target_part.find('.//divisions').text)

    # --- Το ΔΙΚΟ ΜΑΣ, ήδη επιβεβαιωμένο raw MIDI parse - πηγή ΑΛΗΘΕΙΑΣ για timing ---
    midi_division, tempo_map, tracks = parse_midi(midi_path)
    tick_to_sec = make_tick_to_sec(midi_division, tempo_map)
    track_name = 'MELODY' if part_id == 'P1' else ('BRIDGE' if part_id == 'P2' else 'PIANO')
    raw_melody = sorted(next(notes for name, notes in tracks if name == track_name), key=lambda n: n[1])
    raw_idx = 0  # δείκτης στο raw_melody

    chords_sorted = sorted(parse_chord_file(chord_midi_path), key=lambda c: c[0])
    def chord_at(t_sec):
        active = None
        for c in chords_sorted:
            if c[0] <= t_sec:
                active = c
            else:
                break
        return active

    last_chord_symbol = None

    for measure in target_part.findall('measure'):
        to_insert = []
        for note in measure.findall('note'):
            is_chord_note = note.find('chord') is not None
            is_rest = note.find('rest') is not None
            if is_rest or is_chord_note:
                continue

            # ΝΕΟ, ΣΤΙΒΑΡΟ κριτήριο: μια νότα ΧΩΡΙΣ εισερχόμενο tie-stop είναι
            # ΠΑΝΤΑ η αρχή ενός ΝΕΟΥ raw note - μια νότα ΜΕ tie-stop είναι
            # ΠΑΝΤΑ συνέχεια της προηγούμενης (χρησιμοποιούμε το ΙΔΙΟ, ήδη
            # καταγεγραμμένο onset_sec/χορδή, δεν ελέγχουμε ξανά).
            has_incoming_tie = any(t.get('type') == 'stop' for t in note.findall('tie'))

            if not has_incoming_tie:
                if raw_idx >= len(raw_melody):
                    continue
                pitch, onset, offset = raw_melody[raw_idx]
                onset_sec = tick_to_sec(onset)
                raw_idx += 1
                active = chord_at(onset_sec)
                chord_symbol = ('UNKNOWN_ROOT', 'NC') if active is None else (active[2], active[3])
                if chord_symbol != last_chord_symbol:
                    to_insert.append((note, chord_symbol))
                    last_chord_symbol = chord_symbol
            # αν ΕΧΕΙ εισερχόμενο tie-stop: συνέχεια της προηγούμενης νότας - τίποτα να κάνουμε

        for note, chord_symbol in to_insert:
            harmony_el = ET.Element('harmony')
            root_pc, quality = chord_symbol
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
            note_index = list(measure).index(note)
            measure.insert(note_index, harmony_el)

    xml_str = ET.tostring(root, encoding='unicode')
    pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")
    pretty_body = pretty.split('\n', 1)[1]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n' + pretty_body)
    print(f"✅ Αποθηκεύτηκε: {output_path} (ταιριάχτηκαν {raw_idx}/{len(raw_melody)} raw νότες)")


if __name__ == '__main__':
    add_chords_to_converted_musicxml(
        converted_path='converted-score.musicxml',
        chord_midi_path='POP909/034/chord_midi.txt',
        midi_path='POP909/034/034.mid',
        part_id='P1',
        output_path='pop909_034_with_chords_v2.musicxml',
    )

