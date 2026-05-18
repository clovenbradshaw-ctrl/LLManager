/* The 9 EO terrains — the Site face of the EO classification (Domain ×
   Object). Every entity in the graph carries one as its `terrain`.

   These definition sentences are the semantic anchors for mechanical terrain
   classification: each is embedded once (see terrainReferenceVectors in
   local-store.js), and a candidate referent is classified by cosine
   similarity of its own embedding against these nine. Keep the sentences
   keyword-rich and concrete — the embedding model has only this text to go
   on. Single source of truth; imported by mechanical-extract.js and
   local-store.js, imports nothing itself (no import cycle). */

export const TERRAIN_DEFINITIONS = {
  Entity:
    "A specific, bounded, nameable existent — a particular person, a "
    + "particular object, a particular event. This one thing, here.",
  Kind:
    "A type or category — a genre, a species, a classification, a "
    + "recurring kind of thing visible only across many instances.",
  Void:
    "An ambient condition of being — weather, a drought, a mood of a "
    + "place, a background environmental state present before anything is "
    + "picked out.",
  Network:
    "An architecture of connections — an organization as a system, an "
    + "ecosystem, infrastructure, a market, an industry, a platform.",
  Link:
    "A specific connection between particular things — a relationship, a "
    + "contract, a dependency, a partnership, a single bond.",
  Field:
    "An ambient relational environment — power dynamics, unwritten rules, "
    + "an implicit hierarchy, the unspoken vibe of how a group relates.",
  Paradigm:
    "A worldview — an ideology, a scientific paradigm, a religion as an "
    + "interpretive system, a fundamental assumption about reality.",
  Lens:
    "A specific reading or interpretation — a diagnosis, an editorial take, "
    + "one theory applied to one situation, a particular point of view.",
  Atmosphere:
    "An ambient interpretive climate — a cultural mood, a political "
    + "climate, a zeitgeist, the shared assumptions of a moment that nobody "
    + "examines.",
};

export const TERRAIN_NAMES = Object.keys(TERRAIN_DEFINITIONS);
