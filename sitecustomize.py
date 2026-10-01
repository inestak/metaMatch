"""English-only runtime messages for the OAEI release package.

Python imports sitecustomize automatically.  This translates legacy progress
messages emitted by pipeline modules without changing scientific values.
"""
import sys


REPLACEMENTS = (
    ("Configuration locale", "Local configuration"),
    ("Configuration Occidata", "Occidata configuration"),
    ("concurrence", "concurrency"),
    ("mem/tâche", "memory/task"),
    ("aucun test.tsv caché n'a été lu", "no hidden test.tsv was read"),
    ("Aucun test.tsv caché n'a été lu", "No hidden test.tsv was read"),
    ("Réentraînement final sur tout train.tsv", "Final retraining on all train.tsv"),
    ("Sélection train-only BioGITOM", "BioGITOM train-only selection"),
    ("Chargement du modèle transformers", "Loading transformer model"),
    ("Chargement de ", "Loading "),
    ("Extraction labels", "Label extraction"),
    ("classes_and_properties uniquement", "classes_and_properties only"),
    ("propriétés matchables", "matchable properties"),
    ("propriétés ignorées", "properties ignored"),
    ("classes trouvées", "classes found"),
    ("triplets chargés", "triples loaded"),
    ("classes chargées", "classes loaded"),
    ("Entités src à encoder une seule fois", "Source entities to encode once"),
    ("Entités tgt à encoder une seule fois", "Target entities to encode once"),
    ("texte label+synonyms", "label+synonyms text"),
    ("exactement 79 features", "exactly 79 features"),
    ("tâches sorties", "tasks completed"),
    ("tâches échouées/sans reçu", "failed tasks/tasks without receipt"),
    ("tâches", "tasks"),
    ("succès vérifiés", "verified successful"),
    ("checkpoints terminés", "checkpoints completed"),
    ("checkpoints réutilisés", "checkpoints reused"),
    ("candidats lexicaux", "lexical candidates"),
    ("fusionné", "merged"),
    ("fusionnée", "merged"),
    ("exporté", "exported"),
    ("réutilisé", "reused"),
    ("réutilisée", "reused"),
    ("PREFLIGHT OK: entrées valides; aucun calcul lancé.",
     "PREFLIGHT OK: valid inputs; no computation started."),
    ("SOUMISSION FINALE", "FINAL SUBMISSION"),
    ("MATCHES OAEI", "OAEI MATCHES"),
    ("PAIRE TERMINEE", "PAIR COMPLETE"),
    ("packaging séparé après les trois succès", "separate packaging after all three succeed"),
    ("positifs", "positives"),
    ("seulement", "only"),
    ("libres dans", "free in"),
    ("Définir", "Set"),
    ("vers un disque ayant davantage d'espace", "to a disk with more free space"),
    ("Dossier absent", "Missing directory"),
    ("Deux ontologies attendues", "Two ontologies expected"),
    ("trouvé", "found"),
    ("Colonnes de soumission invalides", "Invalid submission columns"),
    ("TSV final introuvable", "Final TSV not found"),
    ("paire inconnue", "unknown pair"),
    ("fichier introuvable", "file not found"),
    ("colonnes manquantes", "missing columns"),
    ("IRI non absolue", "non-absolute IRI"),
    ("relation attendue", "expected relation"),
    ("reçue", "received"),
    ("correspondance dupliquée", "duplicate correspondence"),
    ("Score invalide", "invalid Score"),
    ("Score hors de", "Score outside"),
    ("soumission vide", "empty submission"),
    ("échec du validateur officiel", "official validator failure"),
    ("ZIP prêt", "ZIP ready"),
    ("Archive code prête", "Source archive ready"),
    ("Fichiers", "Files"),
)


def translate(value):
    for source, target in REPLACEMENTS:
        value = value.replace(source, target)
    return value


class EnglishStream:
    def __init__(self, stream):
        self._stream = stream

    def write(self, value):
        return self._stream.write(translate(value))

    def __getattr__(self, name):
        return getattr(self._stream, name)


sys.stdout = EnglishStream(sys.stdout)
sys.stderr = EnglishStream(sys.stderr)
