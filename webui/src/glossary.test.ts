import { describe, expect, it } from "vitest";

import type { GlossaryEntry } from "./api";
import { draftOf, filterTerms, parseAliases, replaceTerm, statusCounts, termChange } from "./glossary";

function entry(id: number, source: string, target: string, extra: Partial<GlossaryEntry> = {}): GlossaryEntry {
  return {
    id,
    source,
    target,
    type: "other",
    gender: "unknown",
    pronouns: null,
    aliases: [],
    notes: null,
    status: "proposed",
    origin: "llm",
    first_seen_chapter: null,
    count: 0,
    ...extra,
  };
}

const JINWOO = entry(1, "성진우", "Sung Jinwoo", {
  type: "person",
  status: "locked",
  aliases: ["진우"],
  notes: "main character",
});
const GATE = entry(2, "게이트", "Gate", { type: "place" });
const HUNTER = entry(3, "헌터", "Hunter", { status: "rejected" });

describe("glossary view helpers", () => {
  it("filters by status and by text in the source, English, aliases or notes", () => {
    const all = [JINWOO, GATE, HUNTER];
    expect(filterTerms(all, "all", "").map((e) => e.id)).toEqual([1, 2, 3]);
    expect(filterTerms(all, "locked", "").map((e) => e.id)).toEqual([1]);
    expect(filterTerms(all, "all", " gate ").map((e) => e.id)).toEqual([2]);
    expect(filterTerms(all, "all", "진우").map((e) => e.id)).toEqual([1]);
    expect(filterTerms(all, "all", "MAIN").map((e) => e.id)).toEqual([1]);
    expect(filterTerms(all, "proposed", "hunter")).toEqual([]);
  });

  it("counts the entries of each status", () => {
    expect(statusCounts([JINWOO, GATE, HUNTER, entry(4, "마석", "mana stone")])).toEqual({
      all: 4,
      locked: 1,
      proposed: 2,
      rejected: 1,
    });
    expect(statusCounts([])).toEqual({ all: 0, locked: 0, proposed: 0, rejected: 0 });
  });

  it("splits aliases on Latin, CJK and Japanese commas", () => {
    expect(parseAliases(" 진우, 진우 씨，성 헌터、 ,")).toEqual(["진우", "진우 씨", "성 헌터"]);
    expect(parseAliases("   ")).toEqual([]);
  });

  it("sends only the fields a draft changes", () => {
    const draft = draftOf(JINWOO);
    expect(draft).toEqual({
      source: "성진우",
      target: "Sung Jinwoo",
      type: "person",
      notes: "main character",
      aliases: "진우",
    });
    expect(termChange(JINWOO, { ...draft, target: " Sung Jinwoo " })).toEqual({});
    expect(termChange(JINWOO, { ...draft, target: "Jinwoo Sung", type: "title" })).toEqual({
      target: "Jinwoo Sung",
      type: "title",
    });
    expect(termChange(JINWOO, { ...draft, source: "진우", notes: " " })).toEqual({ source: "진우", notes: " " });
    expect(termChange(JINWOO, { ...draft, aliases: "진우, 진우 씨" })).toEqual({ aliases: ["진우", "진우 씨"] });
    expect(termChange(JINWOO, { ...draft, aliases: "" })).toEqual({ aliases: [] });
    expect(termChange(GATE, { ...draftOf(GATE), notes: "portal" })).toEqual({ notes: "portal" });
  });

  it("replaces an entry by id", () => {
    const locked = { ...GATE, status: "locked" as const };
    expect(replaceTerm([JINWOO, GATE], locked)).toEqual([JINWOO, locked]);
  });
});
