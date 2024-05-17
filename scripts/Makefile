all: catalan czech english finnish french german hungarian italian portuguese latin

catalan:
	python3 retreeing.py cat/cat.segmentations cat-MorphyNet/UniSegments-1.0-cat-MorphyNet.useg UD_Catalan-AnCora/ca_ancora-ud-train.conllu

czech:
	python3 retreeing.py ces/ces.segmentations ces-DeriNet/UniSegments-1.0-ces-DeriNet.useg UD_Czech-PUD/cs_pud-ud-test.conllu
	python3 retreeing.py ces/ces.segmentations ces-MorphyNet/UniSegments-1.0-ces-MorphyNet.useg UD_Czech-PUD/cs_pud-ud-test.conllu

english:
	python3 retreeing.py eng/eng.segmentations eng-MorphyNet/UniSegments-1.0-eng-MorphyNet.useg UD_English-PUD/en_pud-ud-test.conllu
	python3 retreeing.py eng/eng.segmentations eng-MorphoLex/UniSegments-1.0-eng-MorphoLex.useg UD_English-PUD/en_pud-ud-test.conllu

finnish:
	python3 retreeing.py fin/fin.segmentations fin-MorphyNet/UniSegments-1.0-fin-MorphyNet.useg UD_Finnish-PUD/fi_pud-ud-test.conllu

french:
	python3 retreeing.py fra/fra.segmentations fra-Demonette/UniSegments-1.0-fra-Demonette.useg UD_French-PUD/fr_pud-ud-test.conllu
	python3 retreeing.py fra/fra.segmentations fra-Echantinom/UniSegments-1.0-fra-Echantinom.useg UD_French-PUD/fr_pud-ud-test.conllu
	python3 retreeing.py fra/fra.segmentations fra-MorphoLex/UniSegments-1.0-fra-MorphoLex.useg UD_French-PUD/fr_pud-ud-test.conllu

german:
	python3 retreeing.py deu/deu.segmentations deu-MorphyNet/UniSegments-1.0-deu-MorphyNet.useg UD_German-PUD/de_pud-ud-test.conllu

hungarian:
	python3 retreeing.py hun/hun.segmentations hun-MorphyNet/UniSegments-1.0-hun-MorphyNet.useg UD_Hungarian-Szeged/hu_szeged-ud-train.conllu

italian:
	python3 retreeing.py ita/ita.segmentations ita-DerIvaTario/UniSegments-1.0-ita-DerIvaTario.useg UD_Italian-PUD/it_pud-ud-test.conllu
	python3 retreeing.py ita/ita.segmentations ita-MorphyNet/UniSegments-1.0-ita-MorphyNet.useg UD_Italian-PUD/it_pud-ud-test.conllu

latin:
	python3 retreeing.py lat/lat.segmentations lat-WordFormationLatin/UniSegments-1.0-lat-WordFormationLatin.useg UD_Latin-ITTB/la_ittb-ud-train.conllu

portuguese:
	python3 retreeing.py por/por.segmentations por-MorphyNet/UniSegments-1.0-por-MorphyNet.useg UD_Portuguese-PUD/pt_pud-ud-test.conllu

pipe-prepare:
	python3 paralleling.py

pipe-parse:
	curl -F data=@english-translated-sigm.txt -F input=horizontal -F model=english-gum-ud-2.10-220711 -F tokenizer= -F tagger= -F parser= http://lindat.mff.cuni.cz/services/udpipe/api/process | PYTHONIOENCODING=utf-8 python3 -c "import sys,json; sys.stdout.write(json.load(sys.stdin)['result'])" > eng-parsed-sigm.conllu

pipe-segment:
	python3 retreeing.py eng/eng.segmentations eng-MorphoLex/UniSegments-1.0-eng-MorphoLex.useg eng-parsed-sigm.conllu --directory other

pipe-all: pipe-prepare pipe-parse pipe-segment
