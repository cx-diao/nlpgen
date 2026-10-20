import re
from typing import Dict, Any, Callable

from nlpgena.utils.general import PROJ_ROOT

PROMPT_TEMPLATE_PATH = PROJ_ROOT / "nlpgena/annotate/prompt_template"
CONFIG_PATH = PROJ_ROOT / "nlpgena/annotate/config"
YESNO_LOGPROB_PATTERN = re.compile(r"(?i)\b(yes|no)\b")
ANSWER_PATTERN = re.compile(r"(?i)(?:^|\n|\.\s+)\s*\**\s*answer\s*[\*:.\-]*\s*")

ANNOTATION_COLUMNS = [
    "about_outcome",
    "contains_embedded_sentence",
    "is_reporting_clause",
    "get_target_sent",
    "is_generalised",
    "contains_quantification_in_main_clause",
    "ref_study_population",
    "ref_study",
    "is_present_tense_main_verb",
    "final_label",
    "notes",
    "contains_hedging",
    "contains_vague_descriptor",
    "is_opinion",
    "has_citation",
]

RAW_RESPONSE_COLUMNS = [
    "about_outcome_raw_response",
    "contains_embedded_sentence_raw_response",
    "is_reporting_clause_raw_response",
    "get_target_sent_raw_response",
    "is_generalised_raw_response",
    "contains_quantification_in_main_clause_raw_response",
    "ref_study_population_raw_response",
    "ref_study_raw_response",
    "is_present_tense_main_verb_raw_response",
    "contains_hedging_raw_response",
    "contains_vague_descriptor_raw_response",
    "is_opinion_raw_response",
    "has_citation_raw_response",
]

LOGPROB_COLUMNS = [
    "about_outcome_logprobs",
    "contains_embedded_sentence_logprobs",
    "is_reporting_clause_logprobs",
    "is_generalised_logprobs",
    "contains_quantification_in_main_clause_logprobs",
    "ref_study_population_logprobs",
    "ref_study_logprobs",
    "is_present_tense_main_verb_logprobs",
    "contains_hedging_logprobs",
    "contains_vague_descriptor_logprobs",
    "is_opinion_logprobs",
    "has_citation_logprobs",
]