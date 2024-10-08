from _parser import ParserBrain, ParserDebugger
import _parser
# from _parser import *
from collections import namedtuple
from collections import defaultdict
from enum import Enum
import pptree

LEX_SIZE = 100
LEX = "LEX"
NUMW = "NUMW"
SUBJ = "SUBJ"
OBJ = "OBJ"
VERB = "VERB"
PREP = "PREP"
PREP_P = "PREP_P"
ADJ = "ADJ"
ADVERB = "ADVERB"

DISINHIBIT = "DISINHIBIT"
INHIBIT = "INHIBIT"

# AREAS = [LEX, NUMW, SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P]
AREAS = [LEX, SUBJ, OBJ, VERB]
EXPLICIT_AREAS = [LEX]
# RECURRENT_AREAS = [SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P]
RECURRENT_AREAS = [SUBJ, OBJ, VERB]

# AreaRule = namedtuple('AreaRule', ['action', 'area', 'index'])
# FiberRule = namedtuple('FiberRule', ['action', 'area1', 'area2', 'index'])
# FiringRule = namedtuple('FiringRule', ['action'])
# OtherRule = namedtuple('OtherRule', ['action'])
AreaRule = _parser.AreaRule
FiberRule = _parser.FiberRule
FiringRule = _parser.FiringRule
OtherRule = _parser.OtherRule


def generic_noun(index):
    return {
        "index": index,
        "PRE_RULES": [
            FiberRule(DISINHIBIT, LEX, SUBJ, 0),
            FiberRule(DISINHIBIT, LEX, OBJ, 0),
            FiberRule(DISINHIBIT, VERB, OBJ, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, SUBJ, 0),
            FiberRule(INHIBIT, LEX, OBJ, 0),
            FiberRule(INHIBIT, VERB, OBJ, 0),
            # FiberRule(DISINHIBIT, LEX, SUBJ, 1),
            # FiberRule(DISINHIBIT, LEX, OBJ, 1),
        ]
    }


def generic_trans_verb(index):
    return {
        "index": index,
        "PRE_RULES": [
            FiberRule(DISINHIBIT, LEX, VERB, 0),
            FiberRule(DISINHIBIT, VERB, SUBJ, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, VERB, 0),
            AreaRule(DISINHIBIT, OBJ, 0),
            AreaRule(INHIBIT, SUBJ, 0),
        ]
    }


LEXEME_DICT = {
    'gou': generic_noun(0),
    'haozi': generic_noun(1),
    'na': generic_trans_verb(2),
}

CN_READOUT_RULES = {
    VERB: [LEX, SUBJ, OBJ],
    SUBJ: [LEX],
    OBJ: [LEX],
    LEX: []
}


class CNParserBrain(ParserBrain):
    def __init__(self, p, non_LEX_n=10000, non_LEX_k=100, LEX_k=20,
                 default_beta=0.2, LEX_beta=1.0, recurrent_beta=0.05, interarea_beta=0.5, verbose=False):
        ParserBrain.__init__(self, p,
                             lexeme_dict=LEXEME_DICT,
                             all_areas=AREAS,
                             recurrent_areas=RECURRENT_AREAS,
                             initial_areas=[LEX, SUBJ, VERB],
                             readout_rules=CN_READOUT_RULES)
        self.verbose = verbose

        LEX_n = LEX_SIZE * LEX_k
        self.add_explicit_area(LEX, LEX_n, LEX_k, default_beta)

        # NUMW_k = LEX_k
        self.add_area(SUBJ, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(OBJ, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(VERB, non_LEX_n, non_LEX_k, default_beta)
        # self.add_area(ADJ, non_LEX_n, non_LEX_k, default_beta)
        # self.add_area(PREP, non_LEX_n, non_LEX_k, default_beta)
        # self.add_area(PREP_P, non_LEX_n, non_LEX_k, default_beta)
        # self.add_area(NUMW, non_LEX_n, NUMW_k, default_beta)
        # self.add_area(ADVERB, non_LEX_n, non_LEX_k, default_beta)

        # LEX: all areas -> * strong, * -> * can be strong
        # non LEX: other areas -> * (?), LEX -> * strong, * -> * weak
        # DET? Should it be different?
        custom_plasticities = defaultdict(list)
        for area in RECURRENT_AREAS:
            custom_plasticities[LEX].append((area, LEX_beta))
            custom_plasticities[area].append((LEX, LEX_beta))
            custom_plasticities[area].append((area, recurrent_beta))
            for other_area in RECURRENT_AREAS:
                if other_area == area:
                    continue
                custom_plasticities[area].append((other_area, interarea_beta))

        self.update_plasticities(area_update_map=custom_plasticities)

    def getProjectMap(self):
        proj_map = ParserBrain.getProjectMap(self)
        # "War of fibers"
        if LEX in proj_map and len(proj_map[LEX]) > 2:  # because LEX->LEX
            raise Exception(
                "Got that LEX projecting into many areas: " + str(proj_map[LEX]))
        return proj_map

    def getWord(self, area_name, min_overlap=0.7):
        word = ParserBrain.getWord(self, area_name, min_overlap)
        if word:
            return word
        if not word and area_name == NUMW:
            winners = set(self.area_by_name[area_name].winners)
            area_k = self.area_by_name[area_name].k
            threshold = min_overlap * area_k
            nodet_index = NUMW_SIZE - 1
            nodet_assembly_start = nodet_index * area_k
            nodet_assembly = set(
                range(nodet_assembly_start, nodet_assembly_start + area_k))
            if len((winners & nodet_assembly)) > threshold:
                return "<null-det>"
        # If nothing matched, at least we can see that in the parse output.
        return "<NON-WORD>"


class ReadoutMethod(Enum):
    FIXED_MAP_READOUT = 1
    FIBER_READOUT = 2
    NATURAL_READOUT = 3


def parse(sentence="cats chase mice", language="Chinese", p=0.1, LEX_k=20,
          project_rounds=20, verbose=True, debug=False, readout_method=ReadoutMethod.FIBER_READOUT):

    if language == "Chinese":
        b = CNParserBrain(p, LEX_k=LEX_k, verbose=verbose)
        lexeme_dict = LEXEME_DICT
        all_areas = AREAS
        explicit_areas = EXPLICIT_AREAS
        readout_rules = CN_READOUT_RULES

    parseHelper(b, sentence, p, LEX_k, project_rounds, verbose, debug,
                lexeme_dict, all_areas, explicit_areas, readout_method, readout_rules)


def parseHelper(b, sentence, p, LEX_k, project_rounds, verbose, debug,
                lexeme_dict, all_areas, explicit_areas, readout_method, readout_rules):
    debugger = ParserDebugger(b, all_areas, explicit_areas)

    sentence = sentence.split(" ")

    extreme_debug = False

    for word in sentence:
        lexeme = lexeme_dict[word]
        b.activateWord(LEX, word)
        if verbose:
            print("Activated word: " + word)
            print(b.area_by_name[LEX].winners)

        for rule in lexeme["PRE_RULES"]:
            b.applyRule(rule)

        proj_map = b.getProjectMap()
        for area in proj_map:
            if area not in proj_map[LEX]:
                b.area_by_name[area].fix_assembly()
                if verbose:
                    print("FIXED assembly bc not LEX->this area in: " + area)
            elif area != LEX:
                b.area_by_name[area].unfix_assembly()
                b.area_by_name[area].winners = []
                if verbose:
                    print("ERASED assembly because LEX->this area in " + area)

        proj_map = b.getProjectMap()
        if verbose:
            print("Got proj_map = ")
            print(proj_map)

        for i in range(project_rounds):
            b.parse_project()
            if verbose:
                proj_map = b.getProjectMap()
                print("Got proj_map = ")
                print(proj_map)
            if extreme_debug and word == "a":
                print("Starting debugger after round " +
                      str(i) + "for word" + word)
                debugger.run()

        # if verbose:
        # print("Done projecting for this round")
        # for area_name in all_areas:
        # print("Post proj stats for " + area_name)
        # print("w=" + str(b.area_by_name[area_name].w))
        # print("num_first_winners=" + str(b.area_by_name[area_name].num_first_winners))

        for rule in lexeme["POST_RULES"]:
            b.applyRule(rule)

        if debug:
            print("Starting debugger after the word " + word)
            debugger.run()

    # Readout
    # For all readout methods, unfix assemblies and remove plasticity.
    b.disable_plasticity = True
    for area in all_areas:
        b.area_by_name[area].unfix_assembly()

    dependencies = []

    def read_out(area, mapping):
        to_areas = mapping[area]
        b.project({}, {area: to_areas})
        this_word = b.getWord(LEX)

        for to_area in to_areas:
            if to_area == LEX:
                continue
            b.project({}, {to_area: [LEX]})
            other_word = b.getWord(LEX)
            dependencies.append([this_word, other_word, to_area])

        for to_area in to_areas:
            if to_area != LEX:
                read_out(to_area, mapping)

    def treeify(parsed_dict, parent):
        for key, values in parsed_dict.items():
            key_node = pptree.Node(key, parent)
            if isinstance(values, str):
                _ = pptree.Node(values, key_node)
            else:
                treeify(values, key_node)

    if readout_method == ReadoutMethod.FIXED_MAP_READOUT:
        # Try "reading out" the parse.
        # To do so, start with final assembly in VERB
        # project VERB->SUBJ,OBJ,LEX

        parsed = {VERB: read_out(VERB, readout_rules)}

        print("Final parse dict: ")
        print(parsed)

        root = pptree.Node(VERB)
        treeify(parsed[VERB], root)

    if readout_method == ReadoutMethod.FIBER_READOUT:
        activated_fibers = b.getActivatedFibers()
        if verbose:
            print("Got activated fibers for readout:")
            print(activated_fibers)

        read_out(VERB, activated_fibers)
        print("Got dependencies: ")
        print(dependencies)

        # root = pptree.Node(VERB)
        # treeify(parsed[VERB], root)

    # pptree.print_tree(root)


def main():
    parse(sentence='gou na haozi')


if __name__ == "__main__":
    main()
