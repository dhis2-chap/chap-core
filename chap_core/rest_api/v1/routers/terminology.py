from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from chap_core.auth.permissions import Permission
from chap_core.rest_api.authorization import require
from chap_core.terminology import SOURCE_PATH, SOURCE_REPOSITORY, load_terminology, lookup_term

router = APIRouter(prefix="/terminology", tags=["System"])


class TerminologyModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Term(TerminologyModel):
    """One term of the Chap team's terminology."""

    id: str = Field(description="Stable identifier of the term, in kebab-case.", examples=["baseline-model"])
    term: str = Field(description="The name as the team writes it.", examples=["Baseline model"])
    group: str = Field(description="`Products`, `Models`, `Evaluation` or `Model Marketplace`.", examples=["Models"])
    status: str = Field(
        description="`current` for a term the team uses as written, `proposed` for one it has not gone through yet.",
        examples=["current"],
    )
    definition: str = Field(description="One or two plain sentences, as glossaries show them.")
    note: str | None = Field(default=None, description="A short remark shown with the definition.")
    details: str | None = Field(default=None, description="The longer working definition.")
    aliases: list[str] = Field(default_factory=list, description="Other names and forms of the same thing.")
    avoid: list[str] = Field(default_factory=list, description="Names the team does not use for it.")
    see_also: list[str] = Field(default_factory=list, description="Ids of related terms.")


class TerminologySource(TerminologyModel):
    """Where the terms come from. This server holds a copy that is synced from the source."""

    repository: str = Field(description="GitHub repository that holds the source file.")
    path: str = Field(description="Path of the source file in that repository.")
    commit: str = Field(description="Commit of the source file that this copy was made from.")
    date: str = Field(description="Date of that commit, as an ISO-8601 string.")


class Terminology(TerminologyModel):
    """Every term, with the source they were copied from."""

    source: TerminologySource
    terms: list[Term]


class TermLookup(TerminologyModel):
    """The term a name leads to."""

    term: Term
    name_is_avoided: bool = Field(
        description="`true` when the name asked for is one the team does not use; write `term.term` instead.",
        examples=[False],
    )


@router.get(
    "",
    dependencies=[require(Permission.READ_SETTINGS)],
    response_model=Terminology,
    summary="List the Chap team's terms and where they come from",
)
def list_terminology():
    terminology = load_terminology()
    source = TerminologySource(
        repository=SOURCE_REPOSITORY, path=SOURCE_PATH, commit=terminology.commit, date=terminology.date
    )
    return Terminology(source=source, terms=[Term(**term) for term in terminology.terms])


@router.get(
    "/{name}",
    dependencies=[require(Permission.READ_SETTINGS)],
    response_model=TermLookup,
    summary="Look up the term for a name",
)
def read_term(name: str):
    """Find a term by its id, name, alias or a name the team avoids, ignoring case and a plural "s"."""
    match = lookup_term(name)
    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"No term goes by the name '{name}'. GET /v1/terminology lists every term with its names.",
        )
    return TermLookup(term=Term(**match.term), name_is_avoided=match.avoided)
