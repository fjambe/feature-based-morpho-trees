#!/usr/bin/env python3

"""
Script to produce morphological trees of segmented words and insert them in UD trees for sentences (CoNLLU format).
Exploiting UniMorph and UniSegments data (+ Sigmorphon22 for Czech).
Supported languages: Catalan, Czech, English, Finnish, French, German, Hungarian, Italian, Latin, Portuguese.
"""

import udapi, json, argparse
from addmwt import *
from itertools import zip_longest
from collections import defaultdict, Counter
from handle_sigm_data_ces import *

parser = argparse.ArgumentParser()
parser.add_argument("unimorph", type=str, help="UniMorph file path, e.g. ces/ces.segmentations")
parser.add_argument("disclosure", type=str, help="UniSegments data: either public or private.")
parser.add_argument("unisegments", type=str, help="UniSegments file path, e.g. ces-DeriNet/UniSegments-1.0-ces-DeriNet.useg")
parser.add_argument("treebank", type=str, help="Treebank file path, e.g. UD_Czech-PUD/cs_pud-ud-test.conllu")
parser.add_argument("--directory", type=str, help="Treebank directory, UD (default) or other. If other, provide complete path.")
args = parser.parse_args()

args.unimorph = '/net/work/people/zabokrtsky/git_clones/universal-segmentations/data/original/UniMorph/' + args.unimorph
args.unisegments = f'/net/data/universal-segmentations/UniSegments-1.0-{args.disclosure}/data/' + args.unisegments
segm = args.unisegments.split('/')[-1].split('-')[-1].removesuffix('.useg')
if args.directory == 'other':
    split = None
    lang = None
    tb = None
else:
    tb = args.treebank.split('/')[0].removeprefix('UD_')
    args.treebank = '/net/data/universal-dependencies-2.12/' + args.treebank
    lang = tb.split('-')[0]
    split = args.treebank.split('/')[-1].split('-')[2].removesuffix('.conllu')

splitmerge = AddMwt()


# defining functions
def ready_to_store():
    prefs, roots, sufs = [], [], []
    prefs_lemma, roots_lemma, sufs_lemma = [], [], []
    prefs_deprel, roots_deprel, sufs_deprel = [], [], []
    mwt_root, new_root = '_', '_'  # temp variables
    um_deprel = [] # da verificare
    return prefs, roots, sufs, prefs_lemma, roots_lemma, sufs_lemma, prefs_deprel, roots_deprel, sufs_deprel, mwt_root, new_root, um_deprel


def get_info_from_uniseg(lemma):
    safety_check = []
    for seg in segmentations[lemma.lower()]['segmentation']:
        morpheme = seg.get('morpheme')
        if len(seg['span']) > 0:
            span_start, span_end = seg['span'][0], seg['span'][-1]
            span_end += 1  # because here the last index is included, in slicing not
            # TODO: occhio che potrebbe essere un morfema non consecutivo
        else:
            continue
        safety_check.append(tuple(seg['span']))
        if seg['type'] == 'root':
            control = Counter(safety_check)
            if not span_end or node.lemma[span_start:span_end] == '' or control[tuple(seg['span'])] > 1:
                break
            else:
                roots.append(lemma[span_start:span_end])
            if morpheme:
                roots_lemma.append(seg['morpheme'])
            else:
                if lemma[span_start:span_end] != '':
                    roots_lemma.append(lemma[span_start:span_end])
            if not roots_deprel:
                roots_deprel.append(node.deprel)
            else:
                roots_deprel.append('conj:morph')
        if seg['type'] == 'prefix':  # and span_start and span_end:
            prefs.append(lemma[span_start:span_end])
            if morpheme:
                prefs_lemma.append(seg['morpheme'])
            else:
                prefs_lemma.append(node.lemma[span_start:span_end])
            if node.upos in ['NOUN', 'PROPN']:
                prefs_deprel.append('nmod:morph')
            else:
                prefs_deprel.append('advmod:morph')
        if seg['type'] == 'suffix':
            sufs.append(lemma[span_start:span_end])
            if morpheme:
                sufs_lemma.append(seg['morpheme'])
            else:
                sufs_lemma.append(lemma[span_start:span_end])
            if node.upos in ['NOUN', 'ADJ', 'DET', 'NUM', 'PRON', 'PROPN']:
                sufs_deprel.append('case:morph')
            elif node.upos in ['VERB', 'AUX']:
                sufs_deprel.append('aux:morph')
            else:
                sufs_deprel.append('dep:morph')
    return prefs, prefs_lemma, prefs_deprel, roots, roots_lemma, roots_deprel, sufs, sufs_lemma, sufs_deprel


# building MWT
def define_multiword(node, form, lemma, governing_node, prefs_deprel, roots_deprel, sufs_deprel, multiplier, um_deprel=[]):
    features = [f'{node.feats}'] * multiplier
    nth_roots = len(roots_deprel) - 1  # number of roots excluding the main one (valid e.g. for compounds)
    MWTS[node.form.lower()] = {'form': form,
                               'main': governing_node,
                               'upos': f"{' '.join(['X'] * len(prefs_deprel))} {node.upos} {' '.join(['X'] * (len(sufs_deprel) + nth_roots))} {' '.join(['X'] * (len(um_deprel)))}",
                               'feats': f"{' '.join(features)}",
                               'deprel': f"{' '.join(prefs_deprel)} {' '.join(roots_deprel)} {' '.join(sufs_deprel)} {' '.join(um_deprel)}",
                               'lemma': lemma,
                               'shape': 'subtree'
                               }
    return MWTS


# obtaining morph-feature alignments
def get_alignments():
    if lang.lower() == 'english':
        alignment_file = f'../abishek/{lang.lower()}_morphs_MorphoLex.tsv'
    else:
        alignment_file = f'../abishek/{lang.lower()}_morphs.tsv'
    alignments = defaultdict(list)
    with open(alignment_file, 'r', encoding='utf8') as table:
        content = table.readlines()
        for line in content:
            morph, feat = line.split('\t')[0], line.split('\t')[1]
            morph = morph.strip('+')
            alignments[morph].append(feat)
        return alignments


# loading data from UniSegments
segmentations = {}
with open(args.unisegments, 'r', encoding='utf8') as infile:
    segmented = infile.readlines()
    for line in segmented:
        lemma, info = line.split('\t')[0], line.split('\t')[4]
        info = json.loads(info)
        segmentations[lemma] = info


# loading data from UniMorph
unimorph = defaultdict(dict)
with open(args.unimorph, 'r', encoding='utf8') as um:
    umorphs = um.readlines()
    for line in umorphs:
        if lang == 'Latin':  # data normalisation needed (vowel length)
            from unidecode import unidecode
            form, lemma, segmentation = unidecode(line.split('\t')[1]), unidecode(line.split('\t')[0]), unidecode(line.split('\t')[3]).strip()
        else:
            form, lemma, segmentation = line.split('\t')[1], line.split('\t')[0], line.split('\t')[3].strip()
        if segmentation != '-' and '|' in segmentation:
            unimorph[form][lemma] = segmentation.split('|')
unimorph = {k: v for k, v in unimorph.items() if v is not None}  # to avoid issues

# loading data from SIGMORPHON 2022 (Czech only)
words = None
if lang == 'Czech':
    words = read_file('/lnet/work/people/gamba/morphological-segmentation/ces.sentence.total.tsv')
    frequency = compute_frequency(words)


# TREEBANK
doc = udapi.Document(args.treebank)
for node in doc.nodes:

    # editing of spaces in MISC column in en_GUM and de_GSD
    for feat in ['XML', 'CorrectForm', 'Gloss']:
        if ' ' in node.misc[feat]:
            node.misc[feat] = node.misc[feat].replace(' ', '-')

    # for Czech, first check manually segmented data from Sigmorphon 2022
    if words and node.form.lower() in words:
        root = predict_root(words[node.form.lower()], frequency)
        main_index = words[node.form.lower()].index(root)
        prefs_deprel, sufs_deprel, roots_deprel = [], [], [node.deprel]
        tracking = 0
        for seg in words[node.form.lower()]:
            if words[node.form.lower()].index(seg, tracking) < main_index:
                if node.upos in ['NOUN', 'PROPN']:
                    prefs_deprel.append('nmod:morph')
                else:
                    prefs_deprel.append('advmod:morph')
                tracking += 1
            elif words[node.form.lower()].index(seg, tracking) == main_index:
                tracking += 1
            elif words[node.form.lower()].index(seg, tracking) > main_index:
                if node.upos in ['VERB', 'AUX']:
                    sufs_deprel.append('aux:morph')
                elif node.upos in ['NOUN', 'ADJ', 'ADP', 'ADV', 'DET', 'NUM', 'PRON', 'PROPN']:
                    sufs_deprel.append('case:morph')
                else:
                    sufs_deprel.append('dep:morph')
                tracking += 1
        multiplier = len(prefs_deprel) + len(roots_deprel) + len(sufs_deprel)
        features = [f'{node.feats}'] * multiplier
        form = f"{' '.join(words[node.form.lower()])}"  # lemma equal to form
        define_multiword(node, form, form, main_index, prefs_deprel, roots_deprel, sufs_deprel, multiplier)

    # checking UniSegments
    elif node.lemma.lower() in segmentations and segmentations[node.lemma.lower()].get('segmentation') and len(segmentations[node.lemma.lower()]['segmentation']) != 1: # i.e., if word (lemma) is segmented
            # TODO: fix next line
            prefs, roots, sufs, prefs_lemma, roots_lemma, sufs_lemma, prefs_deprel, roots_deprel, sufs_deprel, mwt_root, new_root, um_deprel = ready_to_store()
            prefs, prefs_lemma, prefs_deprel, roots, roots_lemma, roots_deprel, sufs, sufs_lemma, sufs_deprel = get_info_from_uniseg(node.lemma)

            if len(roots) > 0:
                # enumeration needed to face the issue of duplicates
                prefs_enum = [p for p in enumerate(prefs)]
                tracking = len(prefs_enum)
                roots_enum = [r for r in enumerate(roots, start=tracking)]
                tracking += len(prefs_enum)
                for i, el in roots_enum:
                    if el == roots[0]:
                        selected_root = (i, el)
                sufs_enum = [s for s in enumerate(sufs, start=tracking)]
                indexes = [el for el in [prefs_enum, roots_enum, sufs_enum] if len(el) != 0]
                flat_indexes = [el for sublist in indexes for el in (sublist if isinstance(sublist, list) else [sublist])]
                main_index = flat_indexes.index(selected_root)
                if node.form.lower() == node.lemma.lower():
                    multiplier = len(prefs_deprel) + len(roots_deprel) + len(sufs_deprel)
                    form = f"{' '.join(prefs)} {' '.join(roots)} {' '.join(sufs)}"
                    lemma = f"{' '.join(prefs_lemma)} {' '.join(roots_lemma)} {' '.join(sufs_lemma)}"
                    define_multiword(node, form, lemma, main_index, prefs_deprel, roots_deprel, sufs_deprel, multiplier)
                else:  # node.form.lower() != node.lemma.lower():

                    # checking UniMorph first
                    if node.form.lower() in unimorph and node.lemma.lower() in unimorph[node.form.lower()]:
                        affixes = len(unimorph[node.form.lower()].get(node.lemma.lower())) - 1
                        if not roots_deprel:
                            roots_deprel.append(node.deprel)
                        else:
                            roots_deprel.append('conj:morph')
                        if node.upos in ['VERB', 'AUX']:
                            um_deprel = ['aux:morph'] * affixes
                        elif node.upos in ['NOUN', 'ADJ', 'ADP', 'ADV', 'DET', 'NUM', 'PRON', 'PROPN']:
                            um_deprel = ['case:morph'] * affixes
                        else:
                            um_deprel = ['dep:morph'] * affixes
                        if f"{' '.join(prefs)} {' '.join(roots)} {' '.join(sufs)}" != ' _ ':
                            multiplier = len(prefs_deprel) + len(roots_deprel) + len(sufs_deprel) + len(um_deprel)
                            form = f"{' '.join(prefs)} {' '.join(roots)} {' '.join(sufs)} {' '.join(unimorph[node.form.lower()].get(node.lemma.lower())[1:])}"
                            lemma = f"{' '.join(prefs_lemma)} {' '.join(roots_lemma)} {' '.join(sufs_lemma)} {' '.join(unimorph[node.form.lower()].get(node.lemma.lower())[1:])}"
                            define_multiword(node, form, lemma, main_index, prefs_deprel, roots_deprel, sufs_deprel, multiplier, um_deprel)

                    # if not in UniMorph
                    else:
                        ending, ending_deprel = '', []
                        for i, (char_f, char_l) in enumerate(zip_longest(node.form.lower(), node.lemma.lower())):
                            if char_f == char_l:
                                continue
                            else:
                                if node.form.lower()[0] == node.lemma.lower()[0]:  # at least same initial, otherwise issue of plurimus-multus
                                    ending = node.form.lower()[i:] # approximation
                                    if node.upos in ['NOUN', 'ADJ', 'ADP', 'ADV', 'DET', 'NUM', 'PRON', 'PROPN']:
                                        ending_deprel.append('case:morph')
                                    elif node.upos in ['VERB', 'AUX']:
                                        ending_deprel.append('aux:morph')
                                    else:
                                        ending_deprel.append('dep:morph')
                                    unsegmented = node.lemma.lower()[i:]
                                    if len(roots) == 1:
                                        mwt_root = roots[0]
                                    elif len(roots) > 1:
                                        mwt_root = roots[-1]
                                    if len(mwt_root.removesuffix(unsegmented)) != 0:
                                        new_root = mwt_root.removesuffix(unsegmented)
                                    else:
                                        new_root = mwt_root
                                    if len(roots) == 1:
                                        roots[0] = new_root
                                    elif len(roots) == 2:
                                        roots[-1] = new_root
                                    if not roots_deprel:
                                        roots_deprel.append(node.deprel)
                                    else:
                                        roots_deprel.append('conj:morph')
                                    break  # avoid nesting
                        multiplier = len(prefs_deprel) + len(roots_deprel) + len(sufs_deprel) + 1
                        form = f"{' '.join(prefs)} {' '.join(roots)} {' '.join(sufs)} {ending}"
                        lemma = f"{' '.join(prefs)} {' '.join(roots)} {' '.join(sufs)} {ending}"
                        define_multiword(node, form, lemma, main_index, prefs_deprel, roots_deprel, sufs_deprel,
                                         multiplier, ending_deprel)

    # forms in UniMorph only (absent or not segmented in UniSegments)
    else:
        if node.form.lower() in unimorph:
            prefs, roots, sufs, prefs_lemma, roots_lemma, sufs_lemma, prefs_deprel, roots_deprel, sufs_deprel, mwt_root, new_root, um_deprel = ready_to_store()
            if node.lemma.lower() in unimorph[node.form.lower()]:  # attention: maybe more than one um candidate!
                matching = {k: v for k, v in unimorph[node.form.lower()].items() if k == node.lemma.lower()}
                affixes = len(matching[node.lemma.lower()]) - 1
                segmented = unimorph[node.form.lower()][node.lemma.lower()][1:]  # suffixes (UM inflectional endings)
                unimorph[node.form.lower()][node.lemma.lower()][0] = node.form.removesuffix(''.join(segmented))
                if not roots_deprel:
                    roots_deprel.append(node.deprel)
                else:
                    roots_deprel.append('conj:morph')
                if node.upos in ['VERB', 'AUX']:
                    um_deprel = ['aux:morph'] * affixes
                elif node.upos in ['NOUN', 'ADJ', 'ADP', 'ADV', 'DET', 'NUM', 'PRON', 'PROPN']:
                    um_deprel = ['case:morph'] * affixes
                else:
                    um_deprel = ['dep:morph'] * affixes
                multiplier = 1 + len(um_deprel)
                form = ' '.join(unimorph[node.form.lower()][node.lemma.lower()])
                lemma = ' '.join(unimorph[node.form.lower()][node.lemma.lower()])
                define_multiword(node, form, lemma, 0, [], roots_deprel, [], multiplier, um_deprel)
            else:
                continue  # lemma does not match, so do not split

    # if no segmentation is available, move on
    if (node.lemma.lower() in segmentations and segmentations[node.lemma.lower()].get('segmentation') and len(segmentations[node.lemma.lower()]['segmentation']) != 1) or (node.form.lower() in unimorph and node.lemma.lower() in unimorph[node.form.lower()]): # if in UniSegments or in UniMorph
        splitmerge.process_node(node)


# second round: updating features based on feature-alignment
alignments = get_alignments()
for node in doc.nodes:
    if node.form in alignments and node.multiword_token:
        for f, v in node.feats.items():
            pair = f'{f}={v}'
            if pair in alignments[node.form] and node.parent.multiword_token:  # it probably is a workaround, but so far it works
                node.parent.feats[f] = ''

# storing final CoNLL-U file
with open(f'./outputs/{split}/{tb}-{split}-{segm}.conllu', 'w', encoding='utf8') as final:
    final.write(doc.to_conllu_string())
