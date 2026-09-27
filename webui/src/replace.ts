import type { ReplaceChange, ReplaceRequest } from "./api";

/** The find & replace form of the Studio. */
export interface ReplaceForm {
  find: string;
  replace: string;
  target: "english" | "source";
  scope: "chapter" | "series";
  wholeWord: boolean;
  matchCase: boolean;
  regex: boolean;
}

/** The request for `form` in the open chapter (or the whole series). */
export function replaceRequest(form: ReplaceForm, chapter: string, dryRun: boolean): ReplaceRequest {
  return {
    find: form.find,
    replace: form.replace,
    target: form.target,
    chapters: form.scope === "chapter" ? [chapter] : null,
    regex: form.regex,
    whole_word: form.wholeWord,
    case_sensitive: form.matchCase,
    dry_run: dryRun,
  };
}

/** Whether two requests ask for the same replacement (dry run or not): a preview is only good for its own rule. */
export function sameRule(a: ReplaceRequest, b: ReplaceRequest): boolean {
  return JSON.stringify({ ...a, dry_run: false }) === JSON.stringify({ ...b, dry_run: false });
}

/** "3 English lines in 2 chapters", "1 source text in 1 chapter". */
export function changeSummary(changes: ReplaceChange[], target: "english" | "source"): string {
  const chapters = new Set(changes.map((change) => change.chapter)).size;
  const noun = target === "english" ? "English line" : "source text";
  return `${changes.length} ${noun}${changes.length === 1 ? "" : "s"} in ${chapters} chapter${chapters === 1 ? "" : "s"}`;
}
