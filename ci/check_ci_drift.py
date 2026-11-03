"""The drift check between `.github/workflows/ci.yml` and `ci/gates.sh`.

`ci/gates.sh` runs ci.yml's jobs without GitHub Actions. This check passes only while the two agree:
- **The jobs.** ci.yml's job ids, in order, equal gates.sh's `JOBS`, and each job has a `<id>_job()` function.
- **The steps.** Each job's `run:` steps, in order, equal the `step '…'` lines of its function. A step's `env:`
  becomes a `K="v"` prefix, and `./gitleaks` (ci.yml's downloaded binary) is `gitleaks` on PATH.
- **The pins.** The Rust toolchain and its components, gitleaks' version and its tarball's sha256, and which jobs check
  out the full history. `rust-toolchain.toml`, when present, names the same toolchain.
- **Nothing unknown.** Every other step is one this check knows changes no result: the checkout, the toolchain action,
  the cache, the gitleaks install. A new action, a key such as `if:` or `continue-on-error:`, or an `env` outside a run
  step (other than the cosmetic CARGO_TERM_COLOR) fails the check. gates.sh and this check then change together.

Standard library only. It reads the YAML subset ci.yml uses and refuses anything outside it.
Usage: `python3 -I ci/check_ci_drift.py [repo_root]`. Exit 0 when they agree, 1 on any drift.
"""

import re
import sys
from pathlib import Path


class Unsupported(Exception):
    pass


class Yaml:
    """Block mappings, block sequences, plain and double-quoted scalars, `|` block scalars, full-line comments."""

    def __init__(self, text):
        self.lines = text.split("\n")
        self.i = 0

    def skip(self):
        while self.i < len(self.lines):
            s = self.lines[self.i].strip()
            if s and not s.startswith("#"):
                return
            self.i += 1

    def indent(self):
        self.skip()
        if self.i >= len(self.lines):
            return -1
        line = self.lines[self.i]
        if "\t" in line[: len(line) - len(line.lstrip())]:
            raise Unsupported(f"line {self.i + 1}: a tab in the indentation")
        return len(line) - len(line.lstrip(" "))

    def node(self, ind):
        if self.lines[self.i].strip().startswith("- "):
            return self.seq(ind)
        return self.mapping(ind)

    def seq(self, ind):
        out = []
        while self.indent() == ind and self.lines[self.i].strip().startswith("- "):
            first = self.lines[self.i].strip()[2:]
            self.i += 1
            out.append(self.mapping(ind + 2, first))
        return out

    def mapping(self, ind, first=None):
        out = {}
        while True:
            if first is not None:
                text, first, n = first, None, self.i - 1
            elif self.indent() == ind:
                text, n = self.lines[self.i].strip(), self.i
                self.i += 1
            else:
                break
            m = re.fullmatch(r"([A-Za-z0-9_.-]+):(?: (.*))?", text)
            if not m:
                raise Unsupported(f"line {n + 1}: not a `key: value` line: {text!r}")
            key, raw = m.group(1), (m.group(2) or "").strip()
            if key in out:
                raise Unsupported(f"line {n + 1}: the key {key!r} twice")
            if raw == "|":
                out[key] = self.block(ind)
            elif raw:
                out[key] = self.scalar(raw, n)
            else:
                nxt = self.indent()
                out[key] = self.node(nxt) if nxt > ind else None
        return out

    def block(self, ind):
        body = []
        while self.i < len(self.lines):
            line = self.lines[self.i]
            if line.strip() and len(line) - len(line.lstrip(" ")) <= ind:
                break
            body.append(line)
            self.i += 1
        while body and not body[-1].strip():
            body.pop()
        cut = min(len(b) - len(b.lstrip(" ")) for b in body if b.strip())
        return "\n".join(b[cut:] for b in body) + "\n"

    @staticmethod
    def scalar(raw, n):
        if raw[0] in "[{&*!'>":
            raise Unsupported(f"line {n + 1}: YAML this check does not read: {raw!r}")
        if raw[0] == '"':
            if not (raw.endswith('"') and len(raw) > 1) or "\\" in raw:
                raise Unsupported(f"line {n + 1}: a quoted scalar this check does not read: {raw!r}")
            return raw[1:-1]
        if " #" in raw:
            raise Unsupported(f"line {n + 1}: a trailing comment: {raw!r}")
        return raw

    def parse(self):
        if self.indent() != 0:
            raise Unsupported("the document does not start at column 0")
        doc = self.mapping(0)
        if self.indent() != -1:
            raise Unsupported(f"line {self.i + 1}: indentation this check does not read")
        return doc


INSTALL = re.compile(
    r"curl -sSfL -o gitleaks\.tar\.gz \\\n"
    r"  https://github\.com/gitleaks/gitleaks/releases/download/v(?P<v>[0-9.]+)/gitleaks_(?P=v)_linux_x64\.tar\.gz\n"
    r'echo "(?P<sha>[0-9a-f]{64})  gitleaks\.tar\.gz" \| sha256sum -c -\n'
    r"tar -xzf gitleaks\.tar\.gz gitleaks\n"
)


def workflow(doc):
    """What ci.yml runs: {job: [commands]} and its pins. Raises Unsupported on anything it does not know."""
    extra = set(doc) - {"name", "on", "permissions", "env", "jobs"}
    if extra:
        raise Unsupported(f"workflow keys this check does not know: {sorted(extra)}")
    if set(doc.get("env") or {}) - {"CARGO_TERM_COLOR"}:
        raise Unsupported(f"a workflow-level env this check does not know: {sorted(doc['env'])}")
    jobs, pins = {}, {"toolchain": set(), "components": set(), "gitleaks": set(), "full_history": []}
    for jid, job in doc["jobs"].items():
        unknown = set(job) - {"name", "runs-on", "steps"}
        if unknown:
            raise Unsupported(f"job {jid}: keys this check does not know: {sorted(unknown)}")
        cmds = []
        for k, st in enumerate(job["steps"], 1):
            where = f"job {jid} step {k} ({st.get('name', 'unnamed')})"
            if set(st) - {"name", "uses", "with", "run", "env"} or ("uses" in st) == ("run" in st):
                raise Unsupported(f"{where}: keys this check does not know: {sorted(st)}")
            if "uses" in st:
                if "env" in st:
                    raise Unsupported(f"{where}: env on an action")
                uses, w = st["uses"], st.get("with") or {}
                if uses == "actions/checkout@v4" and set(w) <= {"fetch-depth"}:
                    if "fetch-depth" in w:
                        if w["fetch-depth"] != "0":
                            raise Unsupported(f"{where}: fetch-depth {w['fetch-depth']}")
                        pins["full_history"].append(jid)
                elif uses.startswith("dtolnay/rust-toolchain@") and set(w) <= {"components"}:
                    pins["toolchain"].add(uses.split("@", 1)[1])
                    pins["components"].add(" ".join(sorted(c.strip() for c in w.get("components", "").split(","))))
                elif uses == "Swatinem/rust-cache@v2" and not w:
                    pass
                else:
                    raise Unsupported(f"{where}: an action this check does not know: {uses} {w}")
                continue
            if "with" in st:
                raise Unsupported(f"{where}: `with` on a run step")
            run = st["run"]
            if "\n" in run.rstrip("\n"):
                m = INSTALL.fullmatch(run)
                if not m:
                    raise Unsupported(f"{where}: a multi-line run this check does not know")
                pins["gitleaks"].add((m["v"], m["sha"]))
                continue
            env = "".join(f'{k2}="{v2}" ' for k2, v2 in (st.get("env") or {}).items())
            cmds.append(env + re.sub(r"^\./gitleaks ", "gitleaks ", run.strip()))
        jobs[jid] = cmds
    return jobs, pins


def gates(text):
    """What gates.sh runs: {job: [commands]} and its pins."""
    var = {}
    names = ("TOOLCHAIN", "COMPONENTS", "GITLEAKS_VERSION", "GITLEAKS_LINUX_X64_SHA256", "JOBS", "FULL_HISTORY_JOBS")
    for name in names:
        found = re.findall(rf'^{name}=("?)([^"\n]*)\1$', text, re.M)
        if len(found) != 1:
            raise Unsupported(f"gates.sh: {name}= is set {len(found)} times, not once")
        var[name] = found[0][1]
    jobs = {}
    for fn, body in re.findall(r"^([a-z_]+)_job\(\) \{\n(.*?)^\}$", text, re.M | re.S):
        cmds = []
        for line in body.split("\n"):
            if not line.strip() or line.strip().startswith("#"):
                continue
            m = re.fullmatch(r"  step '([^']+)'(?: &&)?", line)
            if not m:
                raise Unsupported(f"gates.sh {fn}_job: a line that is not one `step '…'`: {line.strip()!r}")
            cmds.append(m.group(1))
        jobs[fn] = cmds
    return jobs, var


def main(root):
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and not ok else ""))

    try:
        wf_jobs, pins = workflow(Yaml((root / ".github/workflows/ci.yml").read_text(encoding="utf-8")).parse())
        sh_jobs, var = gates((root / "ci/gates.sh").read_text(encoding="utf-8"))
    except (Unsupported, OSError, KeyError, TypeError, AttributeError) as e:
        print(f"[FAIL] unreadable: {type(e).__name__}: {e}")
        print("DRIFT CHECK FAIL")
        return 1

    ids = list(wf_jobs)
    check("the jobs: ci.yml's job ids equal gates.sh's JOBS", ids == var["JOBS"].split(), f"{ids} vs {var['JOBS']}")
    fns = {j.replace("-", "_") for j in ids}
    funcs = f"{sorted(fns)} vs {sorted(sh_jobs)}"
    check("each job has its function in gates.sh, and no other", fns == set(sh_jobs), funcs)
    for jid in ids:
        a, b = wf_jobs[jid], sh_jobs.get(jid.replace("-", "_"), [])
        detail = "".join(
            f"\n    {i + 1}: ci.yml {x!r}\n    {i + 1}: gates.sh {y!r}"
            for i, (x, y) in enumerate(zip(a + [None] * len(b), b + [None] * len(a)))
            if x != y and (x or y)
        )
        check(f"job {jid}: the same steps in the same order ({len(a)})", a == b, detail)
    check("the toolchain pin", pins["toolchain"] == {var["TOOLCHAIN"]}, f"{pins['toolchain']} vs {var['TOOLCHAIN']}")
    check(
        "the toolchain components",
        pins["components"] == {" ".join(sorted(var["COMPONENTS"].split()))},
        f"{pins['components']} vs {var['COMPONENTS']}",
    )
    check(
        "the gitleaks pin (version and tarball sha256)",
        pins["gitleaks"] == {(var["GITLEAKS_VERSION"], var["GITLEAKS_LINUX_X64_SHA256"])},
        f"{pins['gitleaks']} vs {(var['GITLEAKS_VERSION'], var['GITLEAKS_LINUX_X64_SHA256'])}",
    )
    check(
        "the jobs that need the full history",
        pins["full_history"] == var["FULL_HISTORY_JOBS"].split(),
        f"{pins['full_history']} vs {var['FULL_HISTORY_JOBS']}",
    )
    tc = root / "rust-toolchain.toml"
    if tc.exists():
        m = re.search(r'^channel = "([^"]+)"$', tc.read_text(encoding="utf-8"), re.M)
        check("rust-toolchain.toml names the same toolchain", bool(m) and m.group(1) == var["TOOLCHAIN"],
              f"{m.group(1) if m else 'no channel'} vs {var['TOOLCHAIN']}")
    ok = all(results)
    print("DRIFT CHECK PASS" if ok else "DRIFT CHECK FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent))
