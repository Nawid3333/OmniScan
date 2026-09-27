import { describe, expect, it } from "vitest";

import type { ReplaceChange } from "./api";
import { changeSummary, replaceRequest, sameRule } from "./replace";
import type { ReplaceForm } from "./replace";

const FORM: ReplaceForm = {
  find: "Jinwoo",
  replace: "Jin-Woo",
  target: "english",
  scope: "chapter",
  wholeWord: true,
  matchCase: false,
  regex: false,
};

describe("replaceRequest", () => {
  it("limits the chapter scope to the open chapter and the series scope to none", () => {
    expect(replaceRequest(FORM, "Chapter 1", true)).toEqual({
      find: "Jinwoo",
      replace: "Jin-Woo",
      target: "english",
      chapters: ["Chapter 1"],
      regex: false,
      whole_word: true,
      case_sensitive: false,
      dry_run: true,
    });
    expect(replaceRequest({ ...FORM, scope: "series" }, "Chapter 1", false).chapters).toBeNull();
  });
});

describe("sameRule", () => {
  it("ignores dry_run but nothing else", () => {
    const preview = replaceRequest(FORM, "Chapter 1", true);
    expect(sameRule(preview, replaceRequest(FORM, "Chapter 1", false))).toBe(true);
    expect(sameRule(preview, replaceRequest({ ...FORM, replace: "Jinu" }, "Chapter 1", false))).toBe(false);
    expect(sameRule(preview, replaceRequest(FORM, "Chapter 2", false))).toBe(false);
  });
});

describe("changeSummary", () => {
  const change = (chapter: string): ReplaceChange => ({ chapter, region_id: "r0001", before: "a", after: "b" });

  it("counts lines and chapters", () => {
    expect(changeSummary([change("1"), change("1"), change("2")], "english")).toBe("3 English lines in 2 chapters");
    expect(changeSummary([change("1")], "source")).toBe("1 source text in 1 chapter");
    expect(changeSummary([], "english")).toBe("0 English lines in 0 chapters");
  });
});
