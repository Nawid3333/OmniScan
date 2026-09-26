import { describe, expect, it } from "vitest";

import type { LearnedRule, MemoryEntry } from "./api";
import { describeRule, filterMemory, groupRules, replaceRule, ruleState } from "./learned";

function rule(id: string, kind: LearnedRule["kind"], count: number, extra: Partial<LearnedRule> = {}): LearnedRule {
  return { id, kind, wrong: "Jlnwoo", right: "Jinwoo", count, enabled: true, active: count >= 2, ...extra };
}

describe("learned rules", () => {
  it("describes what each kind of rule does", () => {
    expect(describeRule(rule("a", "ocr_fix", 2))).toBe("read “Jlnwoo” → write “Jinwoo”");
    expect(describeRule(rule("b", "preferred_term", 2))).toBe("“Jlnwoo” → “Jinwoo” in translations");
    expect(describeRule(rule("c", "drop_text", 2, { wrong: "site.com", right: "" }))).toBe(
      "delete regions reading “site.com”",
    );
    expect(describeRule(rule("d", "sfx_text", 1, { wrong: "쾅" }))).toBe("regions reading “쾅” are a sound effect");
  });

  it("tells on, off and how much evidence is missing", () => {
    expect(ruleState(rule("a", "ocr_fix", 3), 2)).toBe("on");
    expect(ruleState(rule("a", "ocr_fix", 1), 3)).toBe("needs 2 more");
    expect(ruleState(rule("a", "ocr_fix", 5, { enabled: false, active: false }), 2)).toBe("off");
  });

  it("groups by kind in a fixed order, most seen first", () => {
    const rules = [rule("s", "sfx_text", 1), rule("o1", "ocr_fix", 2), rule("o2", "ocr_fix", 7)];
    expect(groupRules(rules).map((g) => [g.title, g.rules.map((r) => r.id)])).toEqual([
      ["OCR fixes", ["o2", "o1"]],
      ["Marked as sound effect", ["s"]],
    ]);
    expect(groupRules([])).toEqual([]);
  });

  it("replaces one rule by id", () => {
    const rules = [rule("a", "ocr_fix", 2), rule("b", "ocr_fix", 2)];
    const off = { ...rules[1], enabled: false, active: false };
    expect(replaceRule(rules, off)).toEqual([rules[0], off]);
  });
});

describe("filterMemory", () => {
  const entries: MemoryEntry[] = [
    { source: "안녕", english: "Hello", count: 1, typed: true, chapter: "1" },
    { source: "가자", english: "Let's go", count: 2, typed: false, chapter: "2" },
  ];

  it("matches the source or the English, ignoring case", () => {
    expect(filterMemory(entries, "hello")).toEqual([entries[0]]);
    expect(filterMemory(entries, "가자")).toEqual([entries[1]]);
    expect(filterMemory(entries, "  ")).toBe(entries);
  });
});
