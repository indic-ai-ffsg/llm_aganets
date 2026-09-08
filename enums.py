"""Generated from the migrations in postgres/sql/. Do not edit by hand.

Regenerate with:  python3 llm_agents_call/generate_enums.py

Every value here is a member of a Postgres enum, as the whole migration
series leaves it and not as 0001 declared it — org_type gains PRIVATE in
0024. A scholarship extracted with a value that is not in these lists
cannot be imported, so the pipeline rejects it rather than guessing what
was meant.
"""

from enum import Enum


class DisabilityType(str, Enum):
    BLINDNESS = "BLINDNESS"
    LOW_VISION = "LOW_VISION"
    LEPROSY_CURED = "LEPROSY_CURED"
    HEARING_IMPAIRMENT = "HEARING_IMPAIRMENT"
    LOCOMOTOR_DISABILITY = "LOCOMOTOR_DISABILITY"
    DWARFISM = "DWARFISM"
    INTELLECTUAL_DISABILITY = "INTELLECTUAL_DISABILITY"
    MENTAL_ILLNESS = "MENTAL_ILLNESS"
    AUTISM_SPECTRUM_DISORDER = "AUTISM_SPECTRUM_DISORDER"
    CEREBRAL_PALSY = "CEREBRAL_PALSY"
    MUSCULAR_DYSTROPHY = "MUSCULAR_DYSTROPHY"
    CHRONIC_NEUROLOGICAL_CONDITION = "CHRONIC_NEUROLOGICAL_CONDITION"
    SPECIFIC_LEARNING_DISABILITY = "SPECIFIC_LEARNING_DISABILITY"
    MULTIPLE_SCLEROSIS = "MULTIPLE_SCLEROSIS"
    SPEECH_AND_LANGUAGE_DISABILITY = "SPEECH_AND_LANGUAGE_DISABILITY"
    THALASSEMIA = "THALASSEMIA"
    HAEMOPHILIA = "HAEMOPHILIA"
    SICKLE_CELL_DISEASE = "SICKLE_CELL_DISEASE"
    MULTIPLE_DISABILITIES = "MULTIPLE_DISABILITIES"
    ACID_ATTACK_VICTIM = "ACID_ATTACK_VICTIM"
    PARKINSONS_DISEASE = "PARKINSONS_DISEASE"


class CourseLevel(str, Enum):
    SCHOOL = "SCHOOL"
    UNDERGRADUATE = "UNDERGRADUATE"
    POSTGRADUATE = "POSTGRADUATE"
    DOCTORAL = "DOCTORAL"


class SocialCategory(str, Enum):
    GENERAL = "GENERAL"
    EWS = "EWS"
    OBC = "OBC"
    SC = "SC"
    ST = "ST"


class SponsorType(str, Enum):
    NGO = "NGO"
    CORPORATE = "CORPORATE"
    GOVERNMENT = "GOVERNMENT"
    PRIVATE = "PRIVATE"


def values(enum_cls) -> list[str]:
    return [e.value for e in enum_cls]


DISABILITY_TYPES = values(DisabilityType)
COURSE_LEVELS = values(CourseLevel)
SOCIAL_CATEGORIES = values(SocialCategory)
SPONSOR_TYPES = values(SponsorType)
