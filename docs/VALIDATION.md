# Audit and Validation

This record separates source-level verification, automated checks, existing demonstration evidence, and work that still requires a consenting participant and a real webcam. It does not report recognition accuracy or benchmark results.

## Audit Findings and Changes

The audit covered both entry points, all four Streamlit pages, both HTML files, dependency pins, ignored/generated data, tracked files, model resources, `Details/`, and all five screenshots plus sampled frames and codec metadata from the existing recording.

| Finding in the original implementation | Targeted change |
| --- | --- |
| Opening Extraction rewrote the CSV; its button displayed success before extraction | Extraction runs only on the button click; success follows a completed, atomic CSV replacement |
| Missing image directories and unreadable images could crash extraction; unusable enrollments became zero descriptors | Missing/invalid input produces an error; unreadable/non-single-face images are skipped; no usable images abort without replacing the CSV |
| Enrollment accepted unsafe/empty names and immediately deleted photos/CSV on Clear | Names are validated, folders cannot be overwritten, duplicate enrollment names are rejected, and Clear requires confirmation |
| Failed name input could leave a previous enrollment active | Invalid input disables saving until a valid enrollment folder is selected |
| Captures included display annotations and reported success without checking image writes | Saving uses an unannotated frame and checks the write result |
| CSV values were not checked for shape, numeric content, or finiteness; a zero first coordinate excluded an otherwise valid descriptor | Shared validation requires a name and 128 finite, nonzero-as-a-vector numeric values |
| Recognition could retain a known name indefinitely when the face count did not change | Periodic descriptor reclassification also applies to previously known faces |
| Recognition used BGR images while enrollment detection used RGB | Extraction and recognition consistently convert to RGB before dlib processing; regenerate existing descriptors before a new live demo |
| Duplicate prevention used a separate SELECT followed by INSERT | A unique database constraint/index and conflict-aware INSERT handle repeated submissions atomically |
| Flask assumed a populated database/table and a valid form date | Missing storage is initialized, bad dates return HTTP 400, storage errors return a readable HTTP 503 response |
| Relative paths depended on the launch directory; camera exceptions could skip release | Shared repository-root paths and camera cleanup in exit/failure paths |
| Startup pages reported Done before a working camera operation; Flask enabled its debugger | Explicit camera-start buttons, operation-dependent messages, and loopback-only Flask startup with debugging disabled |

The UI frameworks, pretrained models, recognition threshold, descriptor format, attendance schema, and separate Streamlit/Flask processes are preserved. `attendance_core.py` isolates shared operations for webcam-free testing. The unused root `index.html` and historical `Details/` illustration remain; Flask renders `templates/index.html`.

## Environment and Reproduction

Validation used Windows x64, Python 3.11.9, the included dlib 19.24.1 wheel, and the pinned requirements. A fresh `.venv` was created because the existing ignored `venv311` pointed to a Python installation that is no longer present. The first dependency-download attempt failed because sandbox DNS was unavailable; installation succeeded after network access was approved.

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install .\dlib-19.24.1-cp311-cp311-win_amd64.whl
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q Main_page.py app.py attendance_core.py pages tests
```

- Virtual environment creation, wheel installation, and pinned dependency installation succeeded.
- `pip check` reported no broken requirements.
- All 31 automated tests passed. They use temporary databases/directories under ignored `.test-artifacts/`, synthetic descriptor values, mocked camera/model responses, Flask's test client, and Streamlit's `AppTest`.
- Syntax compilation passed.
- Both actual bundled dlib models loaded successfully. The actual frontal detector ran on an all-black synthetic image and returned no faces. This checks basic loading/execution, not recognition quality.
- Existing CSV rows passed validation without displaying their names or descriptor values. The local SQLite schema was inspected read-only; existing attendance records and enrollment photos were not modified.
- Streamlit's bare-context warning appears during `AppTest` initialization; it is not a test failure.
- Both documented startup commands passed HTTP smoke checks: Streamlit returned `ok` from `http://127.0.0.1:8501/_stcore/health`, and `python app.py` served the attendance form at `http://127.0.0.1:5000/`. The sandboxed startup attempt did not become ready; the checks passed with approved local execution outside that sandbox. Temporary servers were stopped afterward.
- All README local image/document/video paths resolved. The Mermaid diagram parsed successfully using Mermaid 11 with a Node DOM shim in an ignored validation directory. This validates syntax, not GitHub's rendered layout. FFprobe read the MP4 and FFmpeg decoded sampled frames; no remote media playback guarantee is implied.

## Test Matrix

| Workflow | Automated validation | Still manual/unverified |
| --- | --- | --- |
| Registration | Unsafe/empty/reserved names, distinct folders, invalid-input state reset, clear cancellation/confirmation, failed reads/writes, camera release, no automatic camera opening | Camera permissions, actual cropping/framing and capture quality, full Tkinter interaction |
| Extraction | Missing directory, mean of synthetic descriptors, atomic preservation on failure, Streamlit click versus page load/rerun, truthful success/error state | End-to-end extraction from newly captured consenting participants |
| Recognition | Euclidean distance, nearest match, distances below/equal/above 0.4, zero first coordinate, bad CSV/descriptors, per-face matching, unknown exclusion | Live single/multi-person recognition, crossing-face tracking, identity swaps, lighting/pose variation |
| Attendance | Empty database initialization, first time preserved, same-day duplicate rejection, different-day insertion, direct SQL uniqueness, safe failure on legacy duplicates | Long-running/midnight webcam session and large concurrent workloads |
| History | Homepage, date filtering, empty database, missing/invalid date HTTP 400, unavailable database HTTP 503 | Browser styling across devices and accessibility |
| Cleanup | Recognition failure/unavailable camera cleanup and registration close release | Repeated physical-camera disconnect/reconnect cycles |

Before recording a new end-to-end demo, use participants who have agreed to enrollment, regenerate descriptors, verify a known and unknown face, repeat a same-day recognition, and inspect the date-filtered history. Do not use the Clear action on data that has not been backed up.

## Existing Asset Mapping

Only filenames changed; the original media content was preserved.
Git blob hashes confirmed that all six renamed files are byte-identical to their originals.

| Original file in `Demo/` | Current file | Observed content |
| --- | --- | --- |
| `Screenshot 2026-07-04 235128.png` | `home.png` | Streamlit home and navigation |
| `Screenshot 2026-07-04 235221.png` | `registration.png` | Tkinter controls and name input; disabled camera |
| `Screenshot 2026-07-04 235104.png` | `attendance-confirmation.png` | Streamlit attendance message; no live face shown |
| `Screenshot 2026-07-04 235257.png` | `attendance-history.png` | Flask date selection and stored attendance row |
| `Screenshot 2026-07-04 235436.png` | `server-logs.png` | Prior Streamlit/Flask logs; includes a debugger PIN, so omitted from the README gallery |
| `Screen Recording 2026-07-04 235635.mp4` | `walkthrough.mp4` | Navigation, disabled-camera windows, and attendance viewer |

All five screenshots were visually inspected. The MP4 was probed and sample frames were decoded/inspected; its audio was not reviewed. Neither the recording nor the screenshots establish live-recognition correctness, full enrollment/extraction completion, or benchmark results. No fabricated screenshots were added. `Details/day 1.png` is an older annotated repository layout, not an application screenshot.

## Privacy and Repository Hygiene

- `data/features_all.csv` is already tracked and contains named facial descriptors. These are biometric enrollment data, not an anonymous test fixture; provenance and participant consent cannot be established from the code.
- Local face images and `attendance.db` are ignored and are not tracked in the current tree. New environments, test artifacts, descriptor CSVs, SQLite sidecars, and captures are ignored. Ignore rules do not protect files that are already tracked.
- Existing demonstration assets contain personal names, timestamps, local paths/network details, and, in `server-logs.png`, a debugger PIN. The recording has an audio track that still requires privacy review. Flask now starts without debugging; that does not remove information from old media/history.
- Maintainer approval is required before untracking/replacing existing descriptors, sanitizing existing public media, or removing material from Git history. A safe next step is to back up private enrollment data outside the repository, confirm consent for public demo evidence, replace sensitive assets with approved captures, and then separately decide whether history remediation is necessary.
- No project-level license was found. Do not assume the repository is MIT-licensed. A maintainer must choose a suitable project license and verify third-party distribution requirements.
- The included Windows wheel installed successfully and remains necessary for the documented no-build setup. Its build provenance is not documented in the repository. It and both large model files were retained; no replacement download process was assumed.
- Upstream [dlib model documentation](https://github.com/davisking/dlib-models#shape_predictor_68_face_landmarksdatbz2) warns about commercial use of the 68-point landmark model because of its training dataset. The upstream model collection's general public-domain notice does not erase that warning. Consult the upstream terms before commercial reuse; bundled binary provenance has not been independently established. The [dlib library license](https://github.com/davisking/dlib/blob/master/dlib/LICENSE.txt) is separate from the model/dataset and this project's license.

No real participant images were captured, new biometric descriptors generated, existing enrollment/attendance data deleted, or Git history rewritten during validation. Automated fixtures are synthetic. No secrets were added to the application or tests. This was a targeted hygiene audit, not a comprehensive dependency-vulnerability or security assessment.
