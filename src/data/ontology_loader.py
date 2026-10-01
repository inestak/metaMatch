#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Chargement des ontologies OWL avec rdflib.

Extrait:
- Classes et leurs IRIs
- Labels (rdfs:label, skos:prefLabel)
- Synonymes (oboInOwl:hasExactSynonym, etc.)
- Hiérarchie (rdfs:subClassOf) pour les hard negatives
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, SKOS
from tqdm import tqdm


# Namespaces courants dans les ontologies biomédicales
OBOSCHEMA = Namespace("http://www.geneontology.org/formats/oboInOwl#")


class OntologyLoader:
    """Charge une ontologie OWL et extrait les informations des classes."""

    # Propriétés de label à extraire (par ordre de priorité)
    LABEL_PROPERTIES = [
        RDFS.label,
        SKOS.prefLabel,
        OBOSCHEMA.hasExactSynonym,
        SKOS.altLabel,
        OBOSCHEMA.hasSynonym,
        OBOSCHEMA.hasRelatedSynonym,
    ]

    ENTITY_MODES = {"classes", "classes_and_properties"}

    def __init__(
        self,
        owl_path: str | Path,
        entity_mode: str | None = None,
    ):
        """
        Initialise le loader avec un fichier OWL.

        Args:
            owl_path: Chemin vers le fichier OWL
        """
        self.owl_path = Path(owl_path)
        self.entity_mode = entity_mode or os.environ.get(
            "METAMATCH_ENTITY_MODE", "classes"
        )
        if self.entity_mode not in self.ENTITY_MODES:
            raise ValueError(
                f"entity_mode={self.entity_mode!r}; expected one of "
                f"{sorted(self.ENTITY_MODES)}"
            )
        self.graph = Graph()
        self._classes: Dict[str, dict] = {}
        self._hierarchy: Dict[str, Set[str]] = defaultdict(set)  # parent -> children
        self._parents: Dict[str, Set[str]] = defaultdict(set)    # child -> parents

    def load(self, show_progress: bool = True) -> "OntologyLoader":
        """
        Charge et parse l'ontologie.

        Args:
            show_progress: Afficher la barre de progression

        Returns:
            self pour chaînage
        """
        print(f"Chargement de {self.owl_path.name}...")
        self.graph.parse(str(self.owl_path))
        print(f"  {len(self.graph)} triplets chargés")

        self._extract_entities(show_progress)
        self._extract_hierarchy()

        return self

    def _extract_entities(self, show_progress: bool = True):
        """Extract matchable classes and, optionally, OWL/RDF properties."""
        # Trouver toutes les classes
        classes = set()

        # Classes déclarées explicitement
        for s in self.graph.subjects(RDF.type, OWL.Class):
            if isinstance(s, URIRef):
                classes.add(str(s))

        # Classes référencées comme domaine/range ou dans la hiérarchie
        for s, _, o in self.graph.triples((None, RDFS.subClassOf, None)):
            if isinstance(s, URIRef):
                classes.add(str(s))
            if isinstance(o, URIRef):
                classes.add(str(o))

        entity_types = {iri: "class" for iri in classes}
        properties: Set[str] = set()
        if self.entity_mode == "classes_and_properties":
            for property_type in (OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property):
                for subject in self.graph.subjects(RDF.type, property_type):
                    if isinstance(subject, URIRef):
                        properties.add(str(subject))
            # Some OWL serializations only expose a property through the
            # subPropertyOf hierarchy. Include both URI endpoints.
            for subject, _, parent in self.graph.triples((None, RDFS.subPropertyOf, None)):
                if isinstance(subject, URIRef):
                    properties.add(str(subject))
                if isinstance(parent, URIRef):
                    properties.add(str(parent))
            for iri in properties - classes:
                entity_types[iri] = "property"

        print(
            f"  {len(classes)} classes trouvées"
            + (
                f" + {len(properties - classes)} propriétés matchables"
                if self.entity_mode == "classes_and_properties"
                else " (propriétés ignorées)"
            )
        )

        # Extraire les labels pour chaque classe
        entity_iris = set(entity_types)
        iterator = (
            tqdm(entity_iris, desc="  Extraction labels")
            if show_progress
            else entity_iris
        )

        for class_uri in iterator:
            uri_ref = URIRef(class_uri)
            labels = []
            primary_label = None

            for prop in self.LABEL_PROPERTIES:
                for _, _, obj in self.graph.triples((uri_ref, prop, None)):
                    if isinstance(obj, Literal):
                        label = str(obj)
                        if label and label not in labels:
                            labels.append(label)
                            # Premier label trouvé = label principal
                            if primary_label is None:
                                primary_label = label

            # Si pas de label, utiliser le fragment de l'URI
            if not primary_label:
                primary_label = self._uri_to_label(class_uri)
                if primary_label not in labels:
                    labels.append(primary_label)

            self._classes[class_uri] = {
                "iri": class_uri,
                "label": primary_label,
                "synonyms": labels[1:] if len(labels) > 1 else [],
                "all_labels": labels,
                "entity_type": entity_types[class_uri],
            }

    def _extract_hierarchy(self):
        """Extrait la hiérarchie des classes (subClassOf)."""
        for predicate in (RDFS.subClassOf, RDFS.subPropertyOf):
            for s, _, o in self.graph.triples((None, predicate, None)):
                if not (
                    isinstance(s, URIRef)
                    and isinstance(o, URIRef)
                    and str(s) in self._classes
                    and str(o) in self._classes
                ):
                    continue
                child = str(s)
                parent = str(o)
                self._hierarchy[parent].add(child)
                self._parents[child].add(parent)

    def _uri_to_label(self, uri: str) -> str:
        """Convertit une URI en label lisible."""
        # Extraire le fragment ou le dernier segment
        if "#" in uri:
            fragment = uri.split("#")[-1]
        else:
            fragment = uri.rstrip("/").split("/")[-1]

        # Convertir CamelCase en mots séparés
        label = re.sub(r"([a-z])([A-Z])", r"\1 \2", fragment)
        # Remplacer underscores par espaces
        label = label.replace("_", " ")

        return label

    def get_class(self, iri: str) -> Optional[dict]:
        """Retourne les informations d'une classe par son IRI."""
        return self._classes.get(iri)

    def get_label(self, iri: str) -> str:
        """Retourne le label principal d'une classe."""
        cls = self._classes.get(iri)
        return cls["label"] if cls else self._uri_to_label(iri)

    def get_all_labels(self, iri: str) -> List[str]:
        """Retourne tous les labels (principal + synonymes) d'une classe."""
        cls = self._classes.get(iri)
        return cls["all_labels"] if cls else [self._uri_to_label(iri)]

    def get_parents(self, iri: str) -> Set[str]:
        """Retourne les parents directs d'une classe."""
        return self._parents.get(iri, set())

    def get_children(self, iri: str) -> Set[str]:
        """Retourne les enfants directs d'une classe."""
        return self._hierarchy.get(iri, set())

    def get_siblings(self, iri: str) -> Set[str]:
        """Retourne les frères/sœurs d'une classe (même parents)."""
        siblings = set()
        for parent in self.get_parents(iri):
            siblings.update(self.get_children(parent))
        siblings.discard(iri)  # Retirer la classe elle-même
        return siblings

    def get_ancestors(self, iri: str, max_depth: int = None) -> Set[str]:
        """Retourne tous les ancêtres d'une classe."""
        ancestors = set()
        to_visit = list(self.get_parents(iri))
        depth = 0

        while to_visit and (max_depth is None or depth < max_depth):
            current = to_visit.pop(0)
            if current not in ancestors:
                ancestors.add(current)
                to_visit.extend(self.get_parents(current))
            depth += 1

        return ancestors

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convertit les classes en DataFrame.

        Returns:
            DataFrame avec colonnes: iri, label, synonyms, all_labels, n_parents, n_children
        """
        records = []
        for iri, info in self._classes.items():
            records.append({
                "iri": iri,
                "label": info["label"],
                "synonyms": info["synonyms"],
                "all_labels": info["all_labels"],
                "entity_type": info.get("entity_type", "class"),
                "n_synonyms": len(info["synonyms"]),
                "n_parents": len(self._parents.get(iri, set())),
                "n_children": len(self._hierarchy.get(iri, set())),
            })

        return pd.DataFrame(records)

    @property
    def classes(self) -> Dict[str, dict]:
        """Dictionnaire des classes (IRI -> info)."""
        return self._classes

    def __len__(self) -> int:
        return len(self._classes)

    def __contains__(self, iri: str) -> bool:
        return iri in self._classes


def load_ontology_pair(
    source_owl: str | Path,
    target_owl: str | Path,
    show_progress: bool = True
) -> Tuple[OntologyLoader, OntologyLoader]:
    """
    Charge une paire d'ontologies.

    Args:
        source_owl: Chemin vers l'ontologie source
        target_owl: Chemin vers l'ontologie cible
        show_progress: Afficher la progression

    Returns:
        Tuple (source_loader, target_loader)
    """
    source = OntologyLoader(source_owl).load(show_progress)
    target = OntologyLoader(target_owl).load(show_progress)
    return source, target


# -----------------------------------------------------------------------------
# CLI pour tester
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ontology_loader.py <owl_file>")
        print("Example: python ontology_loader.py data/bioml/omim-ordo/omim.owl")
        sys.exit(1)

    owl_path = sys.argv[1]
    loader = OntologyLoader(owl_path).load()

    df = loader.to_dataframe()
    print(f"\n{len(df)} classes chargées")
    print("\nExemples de classes:")
    print(df.head(10).to_string())

    print("\nStatistiques:")
    print(f"  Labels uniques: {df['label'].nunique()}")
    print(f"  Moyenne synonymes: {df['n_synonyms'].mean():.2f}")
    print(f"  Classes avec synonymes: {(df['n_synonyms'] > 0).sum()}")
