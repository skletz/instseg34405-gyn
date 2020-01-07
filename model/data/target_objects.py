INSTRUMENT_MAPPING = {
    "BG": ["", "#000000", (0, 0, 0), 'Background'],
    "morcellator": ["", "#FFFFFF", (70, 240, 240), 'Morcellator'],
    "needle": ["", "#FFFFFF", (0, 130, 200), 'Needle'],
    "trocar": ["", "#FFFFFF", (0, 128, 128), 'Trocar'],
    "instrument": ["", "#FFFFFF", (0, 0, 128), 'Instrument'],
    "flexible-manipulator": ["", "#FFFFFF", (255, 102, 102), 'Flexible Tip'],
    "rigid-manipulator": ["", "#FFFFFF", (102, 255, 153), 'Rigid Tip']
}

LABEL_MAPPING = {
  0: 'BG',
  1: 'morcellator',
  2: 'instrument',
  3: 'needle',
  4: 'trocar',
  5: 'rigid-manipulator',
  6: 'flexible-manipulator'
}


class Instrument:
    rgb = (255, 255, 255)

    def __init__(self, label_name, label_id, mask, bbox, score):
        self.label = label_name
        self.label_id = label_id
        self.mask = mask
        self.bbox = bbox
        self.score = score

    def set_rgb(self, rgb):
        self.rgb = rgb
