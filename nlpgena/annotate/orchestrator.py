from typing import Dict, List, Any, Tuple
import pandas as pd
import json

from loguru import logger

from rag.llm.core import LLM
from rag.prompt_builder.core import prompt_builder_from_yaml

from nlpgena.annotate.classifier import *
from nlpgena.annotate.sent_level_annotate import *
from nlpgena.annotate.core import BaseAnnotator
from nlpgena.annotate.taxonomy import get_final_label
from nlpgena.annotate.utils import PROMPT_TEMPLATE_PATH

EXAMPLE_NAME2CLASSIFIER_TEMPLATE_PATH_NAME = {
    "about_outcome": {
        "class": DiscourseRoleClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "discourse_role.yaml",
        "template_name": "latest",
    },
    "contains_embedded_sentence": {
        "class": EmbeddedSentenceClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "embedded_sentence.yaml",
        "template_name": "latest",
    },
    "is_reporting_clause": {
        "class": IsReportingClauseClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "is_reporting_clause.yaml",
        "template_name": "latest",
    },
    "get_target_sent": {
        "class": GetTargetSentAnnotator,
        "template_path": PROMPT_TEMPLATE_PATH / "get_target_sent.yaml",
        "template_name": "latest",
    },
    "is_generalised": {
        "class": GeneralityClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "generality.yaml",
        "template_name": "latest",
    },
    "ref_study": {
        "class": ParticularStudyClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "particular_study.yaml",
        "template_name": "latest",
    },
    "ref_study_population": {
        "class": SamplePopulationClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "sample_population.yaml",
        "template_name": "latest",
    },
    "is_present_tense_main_verb": {
        "class": IsPresentTenseClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "tense.yaml",
        "template_name": "latest",
    },
    "contains_quantification_in_main_clause": {
        "class": QuantificationClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "quantification.yaml",
        "template_name": "latest",
    },
    "contains_hedging": {
        "class": HedgingClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "hedging.yaml",
        "template_name": "latest",
    },
    "contains_vague_descriptor": {
        "class": VagueDescriptorClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "vague_descriptor.yaml",
        "template_name": "latest",
    },
    "is_opinion": {
        "class": OpinionClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "opinion.yaml",
        "template_name": "latest",
    },
    "has_citation": {
        "class": CitationClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "citation.yaml",
        "template_name": "latest",
    },
}

class GeneralisationAnnotator:
    def __init__(self, llm: LLM, name2classifier_template_path_name: Dict[str, Dict[str, Any]], debug: bool = False):
        self.llm = llm
        self.debug = debug
        self.classifier, self.metadata = self.init_classifiers(name2classifier_template_path_name)

    def update_classifiers(self, name2classifier_template_path_name: Dict[str, Dict[str, Any]]):
        self.classifier, self.metadata = self.init_classifiers(name2classifier_template_path_name)

    def init_classifiers(self, name2classifier_template_path_name: Dict[str, Dict[str, Any]]):
        """
        Initialise classifiers from config dict.

        Note: Annotators will be executed in the order they appear in the config dict.
        """
        classifier_buffer = {}
        metadata_buffer = {}

        for name, config in name2classifier_template_path_name.items():
            # Parse config
            classifier_class = config["class"]
            scope = config.get("scope", "sentence")
            is_auxiliary = config.get("is_auxiliary", False)

            # Create classifier instance
            if "template_path" in config and "template_name" in config:
                template_path = config["template_path"]
                template_name = config["template_name"]
                prompt_builder = prompt_builder_from_yaml(template_path, template_name)
                classifier_buffer[name] = classifier_class(self.llm, prompt_builder=prompt_builder, debug=self.debug)
            else:
                assert (
                    "rule_based" in config and config["rule_based"] is True
                ), f"Classifier '{name}' has no 'template_path' and must set 'rule_based' to True."
                classifier_buffer[name] = classifier_class(debug=self.debug)

            # Store metadata
            metadata_buffer[name] = {
                "scope": scope,
                "is_auxiliary": is_auxiliary,
            }

        return classifier_buffer, metadata_buffer

    def _collect_skipped_results(self, placeholder: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Reconstruct result dicts for classifiers not in self.classifier but present in the placeholder."""
        skipped: Dict[str, Dict[str, Any]] = {}
        for name in EXAMPLE_NAME2CLASSIFIER_TEMPLATE_PATH_NAME:
            if name in self.classifier:
                continue
            label = placeholder.get(name)
            if label is None or (isinstance(label, float) and pd.isna(label)):
                continue
            logprobs = placeholder.get(f"{name}_logprobs")
            if isinstance(logprobs, str):
                logprobs = json.loads(logprobs)
            skipped[name] = {
                "label": label,
                "raw_response": placeholder.get(f"{name}_raw_response"),
                "logprobs": logprobs,
            }
        return skipped

    get_final_label = staticmethod(get_final_label)

    async def run_about_outcome_phase(
        self, sentence_placeholder: Dict[str, Any], paper_context: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Tuple[str, str]]]:
        """Run paper-level annotators and about_outcome only.

        Returns:
            Tuple of (about_outcome_result, buffer, unified_placeholder, errors)
            where *buffer* and *unified_placeholder* carry forward state needed
            by :meth:`run_downstream_phase`.
        """
        paper_context = paper_context or {}

        buffer = {name: {"label": None, "raw_response": None, "logprobs": None} for name in self.classifier.keys()}
        buffer["final_label"] = {"label": None}
        _errors: List[Tuple[str, str]] = []

        base_placeholder = {**paper_context}

        for name, classifier in self.classifier.items():
            if self.metadata[name]["scope"] == "paper":
                try:
                    result = await classifier.annotate(base_placeholder)
                    buffer[name] = result
                    for key, value in result.items():
                        base_placeholder[f"{name}_{key}"] = value
                except Exception as exc:
                    logger.error(
                        "classifier={} (paper-level) FAILED for sentence={!r}: {}: {}",
                        name, sentence_placeholder.get("claim", "")[:50],
                        type(exc).__name__, exc,
                    )
                    _errors.append((name, f"{type(exc).__name__}: {exc}"))

        unified_placeholder = {**base_placeholder, **sentence_placeholder}

        buffer.update(self._collect_skipped_results(unified_placeholder))

        if "about_outcome" in self.classifier:
            try:
                about_outcome_result = await self.classifier["about_outcome"].annotate(unified_placeholder)
            except Exception as exc:
                logger.error(
                    "classifier=about_outcome FAILED for sentence={!r}: {}: {}",
                    unified_placeholder.get("claim", "")[:50],
                    type(exc).__name__, exc,
                )
                _errors.append(("about_outcome", f"{type(exc).__name__}: {exc}"))
                about_outcome_result = buffer["about_outcome"]
        else:
            assert "about_outcome" in buffer, (
                "about_outcome was skipped but no existing label found in placeholder"
            )
            about_outcome_result = buffer["about_outcome"]

        buffer["about_outcome"] = about_outcome_result
        unified_placeholder["about_outcome"] = about_outcome_result["label"]

        return about_outcome_result, buffer, unified_placeholder, _errors

    async def run_downstream_phase(
        self,
        buffer: Dict[str, Any],
        unified_placeholder: Dict[str, Any],
        _errors: List[Tuple[str, str]],
        force_downstream: bool = False,
    ) -> Dict[str, Any]:
        """Run remaining sentence-level classifiers and compute final_label.

        Args:
            buffer: Classifier result buffer from :meth:`run_about_outcome_phase`.
            unified_placeholder: Merged placeholder from :meth:`run_about_outcome_phase`.
            _errors: Error list accumulated so far (appended to in-place).
            force_downstream: If True, run downstream classifiers even when
                about_outcome is False (used for cross-pass propagation).
        """
        if not force_downstream and not buffer["about_outcome"]["label"]:
            result = {k: v for k, v in buffer.items() if not self.metadata.get(k, {}).get("is_auxiliary", False)}
            result["_errors"] = _errors
            return result

        for name in self.classifier.keys():
            if self.metadata[name]["scope"] != "sentence" or name == "about_outcome":
                continue

            classifier = self.classifier[name]
            try:
                result = await classifier.annotate(unified_placeholder)
                buffer[name] = result
                for key, value in result.items():
                    unified_placeholder[f"{name}_{key}"] = value
            except Exception as exc:
                logger.error(
                    "classifier={} FAILED for sentence={!r}: {}: {}",
                    name, unified_placeholder.get("claim", "")[:50],
                    type(exc).__name__, exc,
                )
                _errors.append((name, f"{type(exc).__name__}: {exc}"))

        if not all(name in buffer for name in EXAMPLE_NAME2CLASSIFIER_TEMPLATE_PATH_NAME):
            buffer["final_label"]["label"] = "Annotation Incomplete"
        else:
            buffer["final_label"]["label"] = self.get_final_label(buffer)

        result = {k: v for k, v in buffer.items() if not self.metadata.get(k, {}).get("is_auxiliary", False)}
        result["_errors"] = _errors
        return result

    async def annotate_one(
        self, sentence_placeholder: Dict[str, Any], paper_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Annotate a single sentence (backward-compatible wrapper).

        Calls :meth:`run_about_outcome_phase` then :meth:`run_downstream_phase`
        sequentially without cross-pass forcing.
        """
        _, buffer, placeholder, errors = await self.run_about_outcome_phase(
            sentence_placeholder, paper_context
        )
        return await self.run_downstream_phase(buffer, placeholder, errors)

    def validate_logprob_columns(self, logprob_columns: List[str]) -> None:
        """Assert that logprob columns align with classifier logprob_pattern settings.

        For each registered non-auxiliary, LLM-based classifier: if it has a
        ``logprob_pattern`` set, a corresponding ``{name}_logprobs`` column must
        be present.  The reverse is not enforced -- a column may exist without a
        pattern (reserved for future use).

        Rule-based classifiers (no ``logprob_pattern`` attribute) and auxiliary
        classifiers are skipped.

        Raises:
            AssertionError when a classifier with a pattern has no column.
        """
        for name, classifier in self.classifier.items():
            if not hasattr(classifier, "logprob_pattern"):
                continue
            if self.metadata[name].get("is_auxiliary", False):
                continue
            if classifier.logprob_pattern is not None:
                col = f"{name}_logprobs"
                assert col in logprob_columns, (
                    f"Classifier '{name}' has logprob_pattern set but " f"'{col}' is missing from logprob_columns"
                )

    async def get_all_prompts(self, claim: str) -> Dict[str, str]:
        raise NotImplementedError("Not implemented")
