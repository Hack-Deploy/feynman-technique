# Oracle sandbox review

Review of the isolation boundary around submitted laws (`dm/oracle/__init__.py`,
`_worker.py`, `_sandbox.py`), at `real-attempts` 6d88347. It covers the isolation
boundary only, not the scoring maths. No code under `dm/` was changed.

## Summary

**Fine for the demo: yes.** A law cannot read the ground truth, cannot run the simulator,
cannot see `DM_ORACLE_SECRET` or API keys in its environment, cannot spawn processes or open
TCP/UDP connections, and cannot forge a pass. The verdict is computed by the trusted worker,
a separate process, from the law's predictions. A forged result line only supplies
predictions, and those are still scored against the real ground truth. This holds for the
scripted solvers and for real LLM solvers run by the owner.

**Safe for outside (untrusted) solvers: no.** The sandbox is a Python audit hook, as
`STATUS.md` says ("a guard rail, not a security boundary"). These attacks work today:

1. **The whole filesystem is readable** except `vendor/` and `/proc`: `poc/.env` (API keys),
   `~/.ssh`, shell rc files (where `DM_ORACLE_SECRET` may be exported), the attempt store,
   and `.git`.
2. **The filesystem is writable** wherever the user can write, including `dm/` and the
   venv's `site-packages`. A `.pth` file there would run attacker code in the next trusted
   worker, which does hold the ground truth. That is a path to a forged pass. I only wrote and
   deleted an inert file; I did not run the full chain.
3. **Exfiltration.** Up to 500 characters of law-controlled text go into `verdict["reason"]`
   and from there into the public attempt record. DNS lookups also leave the box.
4. **`ctypes.pythonapi` is live.** numpy imports `ctypes` before the hook is installed, so
   the law gets a C-API handle and raw memory reads without triggering `ctypes.dlopen`. That is
   enough in principle to disable the audit hook. **I did not demonstrate a full escape.**
5. **No resource limits.** Memory, file size and CPU are unbounded inside the wall-clock
   timeout. A law can also signal any process owned by the same user.

**Recommendation.** Leave the demo as it is. Before any outside solver submits code, run the
sandbox in a container with no network, a read-only root, no repo mount, and memory, PID and
CPU caps (profile at the end). A few hook-level additions (R2–R5) are cheap, run on macOS and
add no latency. They close the easy paths, but they are optional and do not replace the
container.

## Trust boundary

```
dm.settle.settle ──> dm.oracle.score            (parent: the app / market process)
                       │ env scrubbed of *KEY*/*TOKEN*/*SECRET*/*PASSWORD*; cwd = repo root
                       │ subprocess.run(timeout = timeout_s)
                       ▼
                     dm.oracle._worker           (trusted: builds the world, holds ground truth,
                       │                          never runs law code)
                       │ same scrubbed env + PYTHONPATH=repo; cwd = fresh tempdir
                       │ subprocess.run(timeout = budget - elapsed - 2 s)
                       ▼
                     dm.oracle._sandbox          (untrusted: runs the law via the vendor
                                                  evaluator against zeroed trajectories;
                                                  audit hook installed just before the law)
   job: stdin JSON (hidden case inputs, zeroed outputs, law source)
   result: last stdout line (predictions + reason); the worker scores it against its own ground truth
```

What the law process can reach:

| Asset | Reachable? | Why |
|---|---|---|
| Ground-truth trajectories | **No** | Only in the worker process. The sandbox gets zeroed copies (`_zeroed`). |
| Simulator (`scienceagent.worlds`, `physchool`, `jax`) | **No** | Import refused by name; reading files under the vendor tree is refused. |
| `DM_ORACLE_SECRET`, API keys in the environment | **No** | Scrubbed by name in `score()` and again in `_run_sandbox`. |
| Salt | **No** (not passed in) | Only the cases are sent. The salt is in the parent's `Preregistration`. |
| Hidden case *inputs* (p1, p2, pos2, velocity2, times) | **Yes** | Needed to call the law. All 6 cases can be read at once via `job["cases"]`. |
| Secrets in *files* (`poc/.env`, `~/.bashrc`, `~/.ssh`) | **Yes** | `open` is refused only under vendor/ and /proc. |
| Repo, `.git`, `attempts/`, `output/` | **Yes**, read and write | Same user, same filesystem. |
| `.venv/site-packages`, `dm/` (code the worker runs next) | **Yes**, write | Same as above. This is the escalation path. |
| Network | TCP/UDP **no**; DNS **yes** | `socket.connect`/`sendto`/`getaddrinfo` are refused; `socket.gethostbyname` is not. |
| Other processes | Signal **yes**; spawn **no** | `os.kill` is not audited; fork, exec, posix_spawn and Popen are. |
| Result channel (stdout) | **Yes**, forgeable | It carries only predictions, which the worker scores independently. |
| Verdict (`passed`, `normalised_mse`) | **No** | Computed in the worker and parent from the worker's own ground truth. |

## Attack table

Each attack was run as a submitted law through `oracle.score` on
`prereg_for("discoverphysics", "gravity", 0)` with `DM_ORACLE_SECRET` and a fake
`ANTHROPIC_API_KEY` set in the parent. Throwaway scripts lived outside the repo. **Blocked**
means the operation was refused or produced nothing useful. **Succeeded** means the law did
what it tried to do. No attack produced a pass.

| # | Attack | Result | One-line repro (body of `discovered_law`) |
|---|---|---|---|
| 1 | Read `DM_ORACLE_SECRET` / API keys from env | Blocked (absent) | `[k for k in os.environ if 'SECRET' in k or 'KEY' in k]` → `[]` |
| 2 | Read a secret file outside the repo | **Succeeded** | `open(os.path.expanduser('~/x/secret.txt')).read()` → contents in `reason` |
| 3 | Read repo source / `.git/config` / list `attempts/`, `output/` | **Succeeded** | `open(REPO+'/dm/oracle/_worker.py').read()` |
| 4 | Read vendor source (simulator) | Blocked | `open(VENDOR+'/scienceagent/worlds.py')` → PermissionError (hook) |
| 5 | Read `/proc/self/environ` | Blocked | `open('/proc/self/environ','rb')` → PermissionError (hook) |
| 6 | `import scienceagent.worlds` (simulator) | Blocked | ImportError (hook) |
| 7 | Load vendor `worlds.py` by path under a fake module name | Blocked | `spec_from_file_location('zz', path)` → the read is refused |
| 8 | Recompute answers via cached modules | Blocked (nothing useful) | `sys.modules` has `scienceagent.evaluator` + a stub executor, not the simulator |
| 9 | Frame walk to `gt_pos2` | Blocked (zeros) | walk `sys._getframe()` → `gt_pos2` is all zeros |
| 10 | Frame walk to the hidden case inputs | **Succeeded** (expected) | `f.f_locals['job']['cases']` → all 6 cases (inputs only, no answers) |
| 11 | `gc.get_objects()` search for ground truth / secret | Blocked (zeros) | only zeroed `pos2` arrays and case inputs; no secret strings |
| 12 | `np.dot` monkeypatch | Blocked (no effect) | `numpy.dot = lambda a,b: 0.0`; errors are recomputed in the worker (pinned) |
| 13 | Forge the result via fd 1 / `sys.__stdout__` (the real line comes after) | Blocked | parent reads the last line, which is the real result |
| 14 | Forge the result as the *only* line (`os.write(1, fake); os._exit(0)`) | Forgeable, no pass | the worker parses the fake preds and scores them → non-finite, fail |
| 15 | Disable the vendor per-call SIGALRM, then loop forever | Disabled, but **outer timeout holds** | `signal.setitimer(ITIMER_REAL,0); signal.signal(SIGALRM,SIG_IGN); while 1: pass` → "timeout after 27s" |
| 16 | Non-daemon thread keeps the sandbox alive after the law returns | **Succeeded** (bounded by timeout) | `threading.Thread(target=spin15s).start()` → score takes 16.5 s |
| 17 | Allocate 1.5 GB | **Succeeded** | `bytearray(1500*2**20)`; `RLIMIT_AS`/`DATA` = unlimited |
| 18 | Write a 50 MB file | **Succeeded** | write in cwd; `RLIMIT_FSIZE` = unlimited (no disk quota on the tempdir) |
| 19 | CPU cap | **None** | `RLIMIT_CPU` = unlimited; only the wall-clock timeout applies |
| 20 | Write outside the temp dir (`attempts/`, `$HOME`, `dm/oracle/`, `.venv/site-packages`) | **Succeeded** | `open(REPO+'/dm/oracle/_probe.txt','w')` (inert, deleted) |
| 21 | TCP connect / UDP sendto | Blocked | `socket.socket().connect(('1.1.1.1',80))` → PermissionError (hook) |
| 22 | DNS lookup (exfiltration channel) | **Succeeded** | `socket.gethostbyname('example.com')` resolves; arbitrary names are queried |
| 23 | `os.fork` | Blocked | PermissionError (hook) (pinned) |
| 24 | `subprocess.run` / `os.system` | Blocked | PermissionError (hook) |
| 25 | `object.__subclasses__()` → `_wrap_close.__init__.__globals__['system']` | Blocked | the call still raises `os.system` audit → PermissionError |
| 26 | `ctypes.CDLL(None)` (load libc) | Blocked | `ctypes.dlopen` audit → PermissionError (pinned) |
| 27 | `ctypes.pythonapi` C-API call + raw memory read | **Succeeded** (primitive only) | `f=ctypes.pythonapi.Py_GetVersion; f.restype=c_char_p; f()`; `(c_char*11).from_address(id(b))` |
| 28 | Use #27 to disable the audit hook | **Not demonstrated** | hook still refused `import jax` afterwards; the escape is plausible but version-specific |
| 29 | Signal the parent worker / other same-user processes | **Succeeded** | `os.kill(os.getppid(), 0)` succeeds; `SIGKILL` would crash the worker (fail, DoS only) |
| 30 | Exfiltrate data through the verdict | **Succeeded** | `raise RuntimeError(secret)` → text is stored in `verdict["reason"]` (≤ 500 chars) |

Chains these make possible for an outside solver (each step verified, the chain itself not run):

- **Steal keys:** #2 on `poc/.env` (the live market's API keys), then #30. The key ends up in a
  public attempt record.
- **Persistent compromise → forged pass:** #20 drops a `.pth` into `.venv/site-packages`. The
  next worker process imports it at startup with the ground truth in memory and can write any
  verdict. This is the most serious finding for an untrusted deployment.
- **Peek at the hidden cases:** #10 then #30 publishes the hidden case inputs. On the
  free-form DiscoverPhysics venue a second solver could buy experiments at exactly those
  initial conditions and submit a lookup table. (ForceBench's fixed 13-launch menu limits
  this.)

## Recommendations (ranked)

Severity is for an outside-solver deployment. In the demo nothing below is exploitable by
anyone other than the owner's own solvers. "Demo impact" covers latency, macOS support and
behaviour on the owner's laptop. None of these is implemented. Per the owner's rule, harden
only when a real deployment needs it.

| # | Change | Severity | Effort | Demo impact |
|---|---|---|---|---|
| R1 | Run the sandbox (or worker + sandbox) in a container: no network, read-only root, no repo mount, rlimits | **Critical** | M (1–2 days) | Linux/Docker only; +0.3–1 s per score; on macOS keep the current path |
| R2 | Audit hook: refuse writes outside the sandbox cwd; refuse reads outside an allowlist | High | S | None (pure Python, macOS OK); needs the full suite to confirm the allowlist |
| R3 | Audit hook: refuse all `ctypes.*` events except during setup | High | S | None if numpy/scipy don't need ctypes at law time (check with the suite) |
| R4 | Drop law-controlled text from `reason`; record a category instead | Medium | S | Small: per-case error text gets less detailed in the UI |
| R5 | Audit hook: refuse `socket.*` (incl. `gethostbyname`), `os.kill`, `os.killpg`, `signal.pthread_kill` | Medium | XS | None |
| R6 | `setrlimit` in the sandbox child (AS, FSIZE, CPU, NOFILE) | Medium | S | None on Linux; on macOS `RLIMIT_AS` is not enforced (CPU/FSIZE are) |
| R7 | Kill the whole process tree on timeout (`start_new_session` + `killpg`) | Low | XS | None |
| R8 | `python -I -S` + explicit `sys.path` for the sandbox | Low | S | None; small startup gain; ignores `PYTHON*` env and user site |
| R9 | Separate result channel (fd 3) + "degraded" flag in the verdict | Low | S | None |

Sketches:

- **R1 – container (the actual fix).** The sandbox needs only Python, numpy, scipy, the
  vendor `evaluator.py` (not the simulator) and `dm/oracle/_sandbox.py`. Build a small image
  with exactly those files. The worker stays on the host with the ground truth and starts:
  ```
  docker run --rm -i --network=none --read-only --tmpfs /tmp:size=64m,noexec \
    --memory=2g --memory-swap=2g --pids-limit=64 --cpus=1 --user 65534:65534 \
    --cap-drop=ALL --security-opt no-new-privileges [--runtime=runsc] \
    dm-oracle-sandbox python -I -m dm.oracle._sandbox
  ```
  in place of `[sys.executable, "-m", "dm.oracle._sandbox"]` in `_worker._run_sandbox`. The
  job still goes on stdin and the result on stdout, so the rest of the code is unchanged. Add
  `--runtime=runsc` (gVisor) where available for kernel-level isolation. On Linux hosts
  without Docker, `bwrap --unshare-all --die-with-parent --ro-bind <minimal tree> / --tmpfs /tmp`
  or `nsjail` gives the same. Even `unshare -rn --kill-child` (verified here: adds no
  measurable time, network unreachable) closes network but not the filesystem. Make it
  opt-in with something like `DM_ORACLE_SANDBOX=container`, so the demo default on macOS
  stays the current subprocess.
- **R2 – filesystem allowlist in the hook.** The `open` audit event carries
  `(path, mode, flags)`. Refuse any write (`'w'`, `'a'`, `'x'`, `'+'`, or `O_WRONLY`/`O_RDWR`)
  whose realpath is not under the sandbox cwd. Refuse reads not under `sys.prefix`,
  `sys.base_prefix`, the stdlib, the sandbox cwd or the `dm/oracle` directory. This blocks
  #2, #3 and #20 and the `.pth` chain without a container. It is still bypassable through
  #27/28.
- **R3 – ctypes.** The hook only refuses `ctypes.dlopen`, but numpy has already imported
  ctypes. Refuse every event starting with `ctypes.` (`dlsym`, `cdata`, `addressof`,
  `string_at`, `call_function`, …) once the hook is installed. A refused `ctypes.dlsym`
  stops `pythonapi.<anything>`. Run the suite to confirm the vendor evaluator and typical
  numpy/scipy laws don't hit these at law time.
- **R4 – reason channel.** In `_sandbox.main`, keep the exception *type* and a short
  category (`timeout`, `shape`, `non-finite`, `refused: <event>`) and store the full text
  only in a private log, not in `verdict["reason"]`. That stops #30 being a publishing
  channel.
- **R5 – more blocked events.** Add `socket.__new__` (covers every socket, including DNS),
  `socket.gethostbyname`, `socket.gethostbyaddr`, `os.kill`, `os.killpg` and
  `signal.pthread_kill` to `BLOCKED_EVENTS`. The vendor's SIGALRM wrapper uses `setitimer`,
  not `kill`, so it keeps working.
- **R6 – rlimits.** In `_run_sandbox`, add a `preexec_fn` that calls
  `resource.setrlimit(RLIMIT_AS, (2<<30,)*2)`, `RLIMIT_FSIZE` (64 MB), `RLIMIT_CPU`
  (≈ timeout + 5 s) and `RLIMIT_NOFILE` (256). The sandbox doesn't load jax, so 2 GB is
  plenty. macOS ignores `RLIMIT_AS`, so pair it with the container on Linux.
- **R7 – process tree.** Start the worker and the sandbox with `start_new_session=True`, use
  `Popen` + `communicate(timeout=…)`, and on timeout call `os.killpg(proc.pid, SIGKILL)`.
  Today a timeout in `score()` kills only the worker. The sandbox normally times out first
  (its budget is 2 s shorter), so an orphan is unlikely but possible.
- **R8 – interpreter flags.** `[sys.executable, "-I", "-S", "-c", BOOT]`, where `BOOT` puts
  the repo root and site-packages on `sys.path` explicitly. This ignores `PYTHONPATH` and the
  user site and skips `.pth` processing in the sandbox. The `.pth` risk is in the *worker*,
  so R2/R1 matter more.
- **R9 – fd 3 + degraded flag.** Pass the result on `pass_fds=(w,)` instead of stdout and add
  `verdict["sandbox"] = "container" | "subprocess"`, so records say which profile scored
  them. Forgery is already harmless (#14), so this is mostly hygiene and transparency.

## Profiles

| | **Demo profile (current)** | **Real-deployment profile (recommended)** |
|---|---|---|
| Who submits laws | Owner's scripted / fake-LLM solvers; owner-run live LLMs | Outside solvers, untrusted code |
| Isolation | Two Python processes + audit hook | Worker on host; sandbox in Docker + gVisor (or nsjail/bwrap), no repo mount |
| Network | TCP/UDP refused by hook; DNS open | `--network=none` |
| Filesystem | Read everything but vendor and /proc; write wherever the user can | Read-only minimal image; 64 MB tmpfs; nothing from the host |
| Secrets | Scrubbed from env; files on disk readable | No host files visible; secrets never on the sandbox host path |
| Resources | Wall-clock timeout only | Memory 2 GB, PIDs 64, 1 CPU, FSIZE 64 MB, CPU rlimit, wall-clock timeout, kill process group |
| Result channel | Last stdout line; scored by the trusted worker | Same (or fd 3), plus `sandbox` tag in the verdict |
| Reason text | ≤ 500 chars of law output, public | Category only; full text in a private log |
| Platform | Linux + macOS | Linux (Docker Desktop on macOS works for testing, with more latency) |
| Latency | ≈ 1.3 s per score (measured here) | + 0.3–1 s container start (estimate; not measured) |
| Changes to the demo | None | Opt-in switch; the default path stays as today |

## Pinned behaviour

`tests/readiness/oracle_sandbox_pins_test.py` pins guard rails that hold today and aren't
already in `tests/test_phase2_oracle.py::TestHostileLaws`: `/proc` reads, `os.fork` and
`ctypes.CDLL` are refused; a forged sole result line cannot pass; patching `numpy.dot` does
not change the score; the ground truth on the sandbox's stack is all zeros; disabling the
vendor alarm still hits the outer timeout. Attacks that currently succeed are documented
above and are deliberately not in the suite.
