# Vulkan raw and tiled JPEG performance validation

## Composed CFA gain-coordinate fix — 2026-11-17

Gain sampling now carries an independent Bayer-parity offset with the total
source scale. A nearest proxy's parity offset is multiplied by the preceding
binning scale before adding the binning representative's parity offset. Thus
bin-2 + proxy-2 maps output (1,1) to sensor (7,7), not (5,5). The offset uses the
previously unused fourth source-geometry word in the shared gallery gain payload,
which is already part of the spatial-gain cache key. Ordinary Bayer proxies and
bin-only sampling retain their original coordinates.

The shared CPU gain traversal accumulates the same offset when topology and
proxy reductions are composed, and deferred CPU recovery captures it explicitly.
The GPU shader uses it for both direct and cached/captured spatial gain fields.
No sampling transform was added to individual codec adapters.

A regression uses a spatially restricted gain region at sensor rows/columns
6..7, computes expected pixels by composing the two sampling operations
independently, and checks CPU preview, GPU colour output and CPU recovery.
It covers explicit binning plus proxy, implicit binning plus proxy, binning alone
and ordinary Bayer proxying. Linux regression and release builds passed.
Android debug build and raw self-tests passed, including this regression and
exact rotation/edge comparisons. Updated APK installed; normal playback restored
and ten resume clips preserved.

The Android run also exposed incorrect output in a plain-RGB reference display
pass with unused CFA gain/LUT/LOG branches present in its shader. Those branches
are now compiled only into CFA preparation variants: the host always prepares
CFA gains before the final display pass. The scalar GPU demosaic is additionally
checked against the independent CPU bilinear implementation; exact rotation,
edge and display comparisons pass without relaxed tolerances. Failures now
report phase, orientation, first mismatch and maximum code error for diagnosis.
This is a correctness fix, with no new throughput claim.

## CFA integration cleanup review — 2026-11-16

Removed the MCRAW adapter's duplicate binning, proxy sampling and linearization
recovery callback. Shared DNG preview reduction now constructs recovery callbacks
from the original source callback and geometry for both binning and proxies.
MCRAW only supplies decoding of the original recording. Proxy recovery no longer
downloads an intermediate reduced GPU buffer; it follows the source callback,
then repeats the CPU reduction. Recovery closures capture source recovery and
geometry rather than an extra GPU image object. Existing shared linearization
wrapping remains responsible for applying a stored-code LUT once.

The binning shader avoids unused nearest-neighbor coordinate work and channel
division in its single-channel variant. Reduction names and API contracts now
cover both binning and proxy sampling. Diagnostics report total source scale,
topology scale and proxy scale separately. No codec-specific averaging remains
in the MCRAW adapter, and Unspektra coding was not changed.

Focused Linux raw/CFA tests passed, including exact CPU/GPU averages and proxy
composition. Added checks verify that resident processing does not call CPU
recovery, and that recovery invokes the original CPU source once and reproduces
the CPU preview for implicit binning and explicit binning with a proxy. Linux
release and Android debug builds passed. The updated Android APK was installed;
a single hidden-overlay check of the affected clip completed with all 45 frames
using Vulkan decode and resident shared-image display, with no render failure.
Normal playback was restored and the ten resume clips were preserved. No new
throughput claim is made for this cleanup.

## GPU 4×4 CFA gallery binning — 2026-11-16

4×4 CFA is a processing layout, not a different MotionCam compression format.
The raw decoder now admits calibrated 4×4 input for the fast gallery path.
Metadata-only MCRAW staging preserves its 4×4 CFA tags; a staging rejection
recovers on CPU instead of failing the preview. Export/HQ paths retain their
existing full-resolution topology requirements.

The codec-neutral resident reduction helper now averages same-color blocks on
GPU, matching `binHigherCFA` integer half-up rounding. A 4608×3456 4×4 image
becomes 2304×1728 ordinary Bayer. It then joins shared gain/LOG preparation,
Bayer rendering and shared-image display without pixel readback. Original sensor
geometry remains attached for gain sampling. Implicit binning contributes toward
proxy scale; explicit binning preserves its independent proxy scale. Leases and
decode validity survive reduction. Failed processing restarts from the original
layout; CPU source recovery bins from the recording before optional proxy sampling.
The eligible CPU raw fallback also uses the shared bin-first DNG preview path,
keeping its processing order aligned with resident output.

Linux release app/test and Android debug builds passed. Focused tests compare
GPU block averages at factors 2/3/4 against the independent CPU helper exactly,
including full-range values, malformed geometry, explicit binning and proxy
composition. Resident 4×4 preview samples match CPU preview samples. Android
raw self-tests additionally verify metadata-only staging with a 4×4 sidecar;
MCRAW/MC24/UFR1 correctness and resource-lifetime checks passed.

Affected resume clip, histogram and thumbnail row hidden, original LOG/gain and
4×4 calibration retained, muted uncapped playback, four-second benchmark phases:

| Final build | Initial presented fps | After seek fps |
| --- | ---: | ---: |
| GPU decode + bin + shared display | 64.86 | 68.81 |
| CPU decoding forced | 8.74 | 7.53 |

These are short measurements of the 30-frame clip, not a prolonged thermal
stress test. All 45 frames across the GPU run logged Vulkan decoding, resident
Bayer processing and shared-image display with resident input, with no fallback.
Earlier GPU run measured 56.57/66.19 fps; same-build CPU reference 8.04/7.77 fps.
Updated APK installed, normal GPU playback restored and ten resume clips kept.
Logs: ignored `build-linux-release/vulkan-focus/mcraw-slowdown/`.

## Android 16 MP MCRAW slowdown investigation — 2026-11-16

Focused on resume clip `260406_131446_IMAGE.mcraw` (4608×3456, compression 7),
with histogram and thumbnail row hidden. Original processing settings were kept;
no source calibration was modified. An unprofiled presentation-timing run gave
8.42 fps initially and 7.56 fps after seeking. The reported initial 20 fps phase
was not reproduced, so the time-dependent transition remains unconfirmed.

The recording declares ordinary Bayer (`needRemosaic=false`, no cfaSize override),
but its sibling calibration JSON explicitly sets `cfaSize: 4`. Effective 4×4 CFA
processing is not admitted to the current resident raw preview path. Trace
confirmed CPU decoding/processing from the first frame, not a mid-run GPU failure.
CPU preprocessing took roughly 60–85 ms and preview generation another 24–40 ms
in the profiled run. The rendered output was 2304×1728 under these saved settings.
This identifies the slow path, not a proven cause of a 20→7 fps transition.
The FPS indicator itself shows the slowest interval in its last 20 samples,
not mean playback throughput; its readout alone cannot establish an equivalent
change in sustained decode rate.

Fixed a separate confirmed inefficiency: preview/export GPU admission now rejects
a known non-2×2 calibration before loading encoded bytes, avoiding the subsequent
second read/CPU decode of the same frame. Calibration remains authoritative.
Diagnostic fallback records include compression, source dimensions and effective
calibration CFA to make this visible. Linux release and Android debug builds
passed; Android verification retained the same calibrated output dimensions and
reported gpu_candidate=false before the first read. The short profiled check
measured 8.59/8.93 fps before/after seek, without establishing a performance claim.
Updated APK installed, normal playback restored, ten resume clips preserved.
Logs and metadata observations: ignored `vulkan-focus/mcraw-slowdown/`.

## Redundant work audit — 2026-11-15

Reviewed changed JPEG decode/encode, raw validation, gain bake and gallery
transfer/wait paths. Removed confirmed duplication rather than weakening input
validation or removing required interop synchronization:

- Source JPEG decoding no longer requires the resident-slot metadata shader;
  resident-slot decoding no longer initializes source-header/stream validators.
  Successful initialization remains cached, and missing required passes retry.
- GPU-assembled JPEG output no longer builds CPU header bytes. Tile length
  validation, byte totals and output offsets now share one traversal rather
  than three. CPU compressed-byte assembly retains actual header construction;
  duplicate tile length checks during payload copying are removed. Padded
  offsets are explicitly bounded to their 32-bit representation.
- Terminal gain bake reuses its serialized device's validation staging word
  instead of allocating and destroying a four-byte buffer on every output.

The parallel JPEG byte-stuffing prescan and tile status reduction have different
jobs; both remain. JPEG shared-table CPU keys are needed to assign exact GPU
codebook ownership, while GPU headers still validate each tile. Raw preview
checks validity before handing a frame to gallery processing; raw export checks
at its terminal processing boundary. Required output/readback waits remain;
GPU-wide waits inspected here were teardown, explicit profiling, or platform
interop completion requirements. None were removed on a performance assumption.

Linux release and Android debug builds passed. Focused Linux JPEG checks passed
with GPU and CPU compressed-byte assembly, including two differently valued
one-pixel RGB tiles to verify header lengths and offsets. The Android 12 MP
resident roundtrip and precision fixtures passed exact pixel comparisons; the
updated APK was installed and normal playback restored. This audit does not
establish a throughput improvement or prove every remaining cost unavoidable.

## Matched overhead-cleanup comparison — 2026-11-15

Compared the recent setup/upload/allocation/traversal cleanups against a build
with only those changes reversed. Both variants retain shader initialization
retry/dependency fixes and all prior codec tuning. Sources were restored exactly
to the candidate snapshots after producing the baseline binaries. Linux release
and Android debug used A/B/B/A order, four-second playback phases before and
after seeking, full resolution and muted uncapped playback. JPEG input is the
same 60-frame 12 MP / 16×64-tile sequence copied from the phone. Each platform's
MC24 source and settings remained identical across variants (MC24 recordings
and builds differ between platforms). Android thumbnails were disabled; Linux
retained its benchmark thumbnail setting. Absolute cross-platform rates should
not be compared. No extra GPU stage profiling was enabled.

After-seek playback rates, two samples per variant:

| Platform / codec | Baseline fps | Cleanup fps |
| --- | --- | --- |
| Android JPEG | 22.24, 23.60 | 23.90, 23.51 |
| Android MC24 | 26.74, 25.83 | 25.57, 25.70 |
| Linux JPEG | 57.24, 58.33 | 58.06, 75.64 |
| Linux MC24 | 65.76, 60.72 | 55.97, 61.24 |

These short runs do not establish a reliable throughput improvement. Android's
final JPEG baseline overlaps the candidate; Linux has substantial variation.
MC24 samples are lower with the cleanup (mean -2.45% Android, -7.34% Linux), so
validation pooling is not a demonstrated performance win and merits isolation
before any performance claim. There are only two samples per variant; these
percentages are descriptive, not significance estimates.

An additional Android roundtrip (two warmups, three measured iterations) passed
exact pixel checks for both builds. Median encode was 27.83 ms baseline versus
28.49 ms cleanup; resident re-decode 7.84 versus 7.73 ms, and total roundtrip
69.51 versus 71.06 ms. No encoder speedup was demonstrated. Precision fixtures
also passed. Full logs, source snapshots, comparison binaries/APKs and JSON
summary are in ignored `build-linux-release/vulkan-focus/overhead-ab/`.
The current candidate APK was restored after measurement.

## Validation/control and dispatch allocation cleanup — 2026-11-15

MC24 control words now use a separate bounded pool (eight retired words), keeping
their ownership tied to each decoded frame and proxy lease. Only idle retired
words are reused, and the existing pool-disable diagnostic bypasses reuse.
Validation reads share one staging word under the context submission lock rather
than allocating and destroying a staging buffer per check. Encoder dispatch
helpers use five stack descriptor bindings instead of per-pass vectors. Resident
JPEG dimensions/precision checks run in the existing admission traversal.

Linux release app/test and Android native debug builds passed. Decoder-only
MCRAW/MC24/Unspektra fixtures and focused JPEG tests passed. New MC24 coverage
holds eight alternating valid/invalid frames before reading their independent
status words, then repeats after lease retirement to exercise pool reuse.
These changes have not been packaged/deployed to the phone in this pass.

## Repeated setup work cleanup — 2026-11-15

Successful JPEG decoder initialization now returns before shader-source setup;
Android device properties are queried only on the first initialization attempt.
Missing passes still retry. Contiguous GPU-parsed input skips the otherwise empty
second tile loop. Full-frame and tiled encoders share a histogram clear helper:
the zero staging buffer is uploaded once per allocation, while the necessary
GPU histogram clear copy remains per encode. Full-frame initialization also
returns immediately once all required passes are present.

Linux release and Android native debug builds passed; the focused Linux JPEG
correctness check passed. These changes remove repeated setup work; no new
throughput claim or exhaustive absence-of-overhead claim is made.

## Shader initialization follow-up — 2026-11-15

JPEG decode and full-frame encode now create only missing passes on retry,
preserving successful compilation and rejecting a mismatched GPU. Tiled encode
initializes the shared Huffman table and selected entropy variant independently
of full-frame packing, scans, byte stuffing, and the other input representation.
Texture histogram compilation is shared without requiring full-frame packing.
JPEG assembly and CPU-output RGB compaction initialize only when requested.

Linux release app/test and Android debug builds passed. Focused Linux JPEG
fixtures passed with both GPU assembly and compressed-byte CPU assembly, including
the one-pixel RGB edge case. The Android 12 MP resident roundtrip and existing
8/12/16-bit CFA/RGB fixtures passed exact pixel comparisons. No throughput
improvement is claimed by this cleanup.

## JPEG cleanup review — 2026-11-15

The review fixed source-decoder static lifetime ordering: cached passes and
buffers are destroyed before their Vulkan context owner, including when no
retained image lease remains. Focused coverage now ends with a successful decode
and released image, rather than leaving only a failed decode's allocation.

Tiled JPEG assembly now bounds leading entropy bytes by the actual body length
and preserves the partial header word for very short bodies. A one-pixel RGB
fixture is independently decoded with liblj92. Assembly initialization retries
missing passes individually; an unavailable assembly shader preserves GPU
entropy encoding with CPU assembly of compressed bytes for ordinary output.
Resident direct-slot encoding does not require the assembly shader.

Linux release build and all 24 focused decode layouts, malformed-input checks,
and the short RGB encode fixture passed. Android debug build and one 12 MP
roundtrip run passed exact pixel comparisons, including the existing 8/12/16-bit
CFA/RGB fixtures. Three timed iterations measured median encode 28.13 ms and
resident re-decode 7.86 ms; this correctness review makes no encode speedup claim.
The normal Android resume session retains its ten clips.

Decode Huffman/predictor/workgroup tuning does not alter encoder workgroup or
tile dimensions. Shared gain/LOG packing fixes affect encode preparation; the
resident packed-buffer API additionally supports GPU-only roundtrip consumers.

## Complete-symbol, cross-frame cache and word-reader trials — 2026-11-15

All three proposals were implemented and compared against the current shared
9-bit decoder. None produced a repeatable playback improvement, so production
decoder/main code was restored byte-for-byte to its state before these trials.
The existing shared 9-bit preview selection and latency-oriented streaming path
remain in place.

CPH2581, 12 MP / 16×64 tiles, ordinary full-resolution playback without GPU
profiling, matched short runs with identical histogram/thumbnail/zero-copy flags:

| Decoder | Playback after seek |
| --- | ---: |
| Current decoder | 24.04 fps |
| Complete-residual lookup entries | 23.20 fps |
| Persistent GPU codebook cache | 23.64 fps |
| Complete-residual entries + cache | 22.95 fps |
| Portable GPU unstuffing + word reservoir | 21.63 fps |

Complete-residual entries used the existing 32-bit table storage: an entry flag,
consumed-bit count and decoded signed residual when the complete symbol fit in
nine bits. Short tails and longer symbols retained canonical decoding. Exact
roundtrips passed. A separate synchronized comparison showed source decode
33.55→32.56 ms, resident re-decode 7.72→7.74 ms and roundtrip 69.20→68.81 ms;
the small isolated latency change did not translate into faster playback.

The cache retained up to 8,192 codebooks in a separate 32 MiB GPU buffer with
stable table IDs/bank stride. New entries were committed only after full GPU
frame validation succeeded; failed frames could not publish unbuilt tables.
Format changes cleared the dictionary and capacity pressure decoded the current
frame through the ordinary shared-table path. Initial repeated-frame roundtrip
latency worsened from 67.82 to 71.61 ms. A refinement kept compressed-header
preparation outside the GPU mutex and remapped per-frame IDs under the existing
lock. Its playback comparison measured 23.76 fps before, 23.31 fps with the cache,
23.26 fps with cache + complete entries, and 23.64 fps in the final control.
That refinement was also discarded.

The unstuffing prototype replaced the separate entropy validation scan for
eligible direct-output frames. A workgroup validated FF/00 pairs, compacted bytes
through shared memory, and wrote aligned GPU slots. The reservoir then loaded
up to four bytes per word without stuffing branches. A subgroup-scan version
failed the independent compressed-byte oracle (243 expected compacted bytes
versus 227 produced in one fixture) and was excluded from timing. The correct
portable workgroup-scan version passed both compressed-byte and exact-pixel
checks. With diagnostic downloads disabled, source decode measured 38.34 ms
versus 35.29 ms control, resident re-decode 16.97 versus 7.75 ms, and roundtrip
84.13 versus 71.13 ms. No raw pixel readback was added to the measured chain;
only the temporary diagnostic oracle downloaded compressed input/output bytes.

The initial combined implementations passed the instrumented Linux focused GPU
suite, including mixed ordinary/paired CFA, malformed entropy and bad first
shared-table owners. The malformed-owner fixture now runs before the positive
cases, covering a cold lookup allocation. The later cache scheduling refinement
was playback-profiled and rejected; it is not shipped.

Artifacts live in `build-linux-release/vulkan-focus/`: `jpeg-symbol-*`,
`jpeg-cache-*`, `jpeg-wordstream-*`, `jpeg-three-*`, and `jpeg-cache-overlap-*`.
The complete portable prototypes are saved as `jpeg-three-final-prototypes.cpp`
and `main-three-final-prototypes.cpp`; the subgroup/oracle investigation is in
`jpeg-three-subgroup-prototype.cpp`. Baseline snapshots are
`jpeg-before-symbol-cache-unstuff.cpp` / `main-before-symbol-cache-unstuff.cpp`.
No experimental flags or extra buffers remain in production code.

## Shared wide Huffman lookups and reconstruction trials — 2026-11-15

The retained Adreno preview decoder now prefers a shared 9-bit prefix table.
The first device initialization uses the existing 7-bit path; subsequent retained
frames with at least 128 tiles can use 9 bits. Lightweight CPU keys cover exact
DHT payload bytes, while the existing GPU parser still validates every header.
A designated tile builds each shared lookup table on GPU; metadata selects the
shared table and its independent table-0/table-1 bank. A reusable key buffer
avoids allocating a string for every tile. The 12 MP reference frame has 3,006
DHT keys across 12,288 tiles.

Automatic preview admission requires at most half as many distinct tables as
tiles and caps lookup storage at 64 MiB or the GPU SSBO limit, whichever is
smaller. Low reuse, excessive storage, or a wide shader creation failure retries
the existing narrow GPU path. Other GPUs retain 7 bits by default. Resident
encoded JPEG already shares one codebook, so Adreno re-decode also prefers 9 bits.
Header, bounds, entropy and status checks remain enabled; raw pixels stay on GPU.

Final ordinary playback, CPH2581, 4096×3072, full resolution, no histogram or
thumbnail row, strict zero-copy, **without GPU profiling**, six seconds per phase:

| Measurement | Previous 7-bit lookup | Automatic shared lookup |
| --- | ---: | ---: |
| Playback after seek | 22.10 fps | 23.64 fps |

A reverse-order ordinary playback check measured 23.92 fps with forced 9 bits
and 22.08 fps with 7 bits (the baseline completed on retry after an Android
startup stall with an empty diagnostic log). Separate short profiled playback
measured JPEG entropy GPU time 12.50→10.25 ms and parser GPU time 1.46→1.13 ms.
Ten bits measured 12.60 ms entropy time and 22.20 fps in that profiling comparison,
so 9 bits was retained.

**There is a latency tradeoff.** The synchronized retained-image roundtrip benchmark
measured source decode 25.73→33.67 ms, resident re-decode 8.24→7.77 ms, and the
full roundtrip 62.25→69.41 ms. GPU encode remained approximately 28 ms. That
benchmark continues to use the retained preview API, so it records this cost.
Streaming/validated source decode and ordinary CPU-output imports retain the
shorter 7-bit setup path by default; the wider automatic source choice is limited
to retained previews. No processing/display pipeline or tile dimensions changed.

Two-stage decode was implemented and rejected. Entropy decoding wrote packed
16-bit residuals to a GPU buffer, followed by cooperative reconstruction with
serial fallback for other predictors. An initial diagonal-sum reconstruction was
correct but slow. A linear-work version computed independent diagonal chains
into padded shared memory and packed pixels in parallel. On 16×64 tiles it still
measured source decode 33.70 ms and resident re-decode 21.81 ms versus approximately
25.46/8.47 ms for the current decoder. On 64×64 tiles its complete roundtrip was
97.41 ms versus 69.88 ms. Neither version remains in production code.

Larger shared tables were first prototyped with CPU canonical-table expansion;
that increased source decode to 64–66 ms and was removed. GPU construction reduced
this cost and produced the measured playback gain above. Experimental source
snapshots remain as `jpeg-all-experiments.cpp` / `main-all-experiments.cpp` under
`build-linux-release/vulkan-focus/`; final production code contains only the
shared-table implementation.

Validation: Android exact-code roundtrips and CPU decoding of GPU-encoded CFA/RGB
8/12/16-bit fixtures passed. The optimized standalone Linux test target now
explicitly enables diagnostic codec switches: its inherited Release definition
previously ignored them, so earlier override-based Linux runs only exercised the
default path. With this harness fixed, all 24 focused layout fixtures passed
separately for 7 bits, 9 bits and automatic admission. Checks include mixed ordinary/
paired CFA layouts, partial groups, malformed entropy, trailing stuffing and a
malformed first tile sharing its tables with valid later tiles. The automatic
run logged 81 narrow-fallback decisions. Release application diagnostics remain
disabled; this CMake override applies only to the standalone JPEG test target.

Debug flags: `jpeglookup7`, `jpeglookup9`, `jpeglookup10` force a width for source
and resident decode; `jpeglookupauto9` forces automatic 9-bit admission for tests
on other GPUs. Artifacts include `jpeg-shared-final-*`, `jpeg-shared-gallery-*`,
`jpeg-shared-playback-*`, `jpeg-stage-*`, `jpeg-stage64-*`, `jpeg-lookup-*`, and
`jpeg-shared-final-{7,9,fallback}-correctness.log`. Previous code is saved as
`jpeg-before-two-stage.cpp` / `main-before-two-stage.cpp` in the same artifact folder.

## Decode-only Adreno workgroup tuning — 2026-11-15

Android Adreno now uses 16 invocations per workgroup for source JPEG decode and
32 for resident encoded-JPEG re-decode. The simpler known-component resident
shader benefits from a wider group than the mixed-layout source shader. The
sizes were retested after the entropy/predictor simplifications; the earlier
32/64-thread results on the older shader no longer described resident decode.
Other GPUs retain eight invocations. Device properties are checked once per
cached decode state. A variant compilation failure still dispatches the original
eight-thread fallback with its actual workgroup size.

Final matched CPH2581, 12 MP, 12,288 complete 16×64 tiles, five synchronized
roundtrips after two warmups:

| Measurement | Previous eight-thread default | Tuned default |
| --- | ---: | ---: |
| Source decode wall median | 27.84 ms | 24.65 ms |
| Resident JPEG re-decode wall median | 14.66 ms | 8.72 ms |
| GPU encode wall median | 27.55 ms | 27.74 ms |
| Complete resident roundtrip wall median | 69.37 ms | 60.86 ms |
| Gallery JPEG entropy GPU median | 15.03 ms | 12.43 ms |
| Gallery playback after seek | 20.89 fps | 21.47 fps |

All processing, display, bit precision, predictor arithmetic and validation
settings are unchanged. No pipeline overlap or frame duplication changes were
introduced. Roundtrip wall timings include synchronization and exclude file I/O,
CPU references and pixel comparisons. Gallery is a separate short profiling run;
its small FPS difference is less pronounced than the decode-only GPU gain.

The initial size sweep showed why the two paths need separate defaults:
8/16/32-thread source decode measured 28.24/25.52/41.24 ms, while resident re-decode
measured 14.24/9.36/8.86 ms. Debug `jpegwg8`, `jpegwg16`, `jpegwg32` (or
`RAWPREAMP_JPEG_GROUP_SIZE`) override both paths. `jpegwg8` reproduces the previous
default. Dispatch diagnostics now include `group_size`.

Rejected experiments were removed: a fully unrolled predictor row cache increased
source decode 26.75→46.81 ms; a one-word predictor cache had mixed source/resident
results; paired-byte refill increased source decode 24.63→27.28 ms; filtered CFA
component dispatches passed correctness but increased source decode 24.78→44.22 ms.
Only the workgroup tuning remains from these experiments.

Android exact-code 12 MP roundtrips and independent CPU decoding of GPU-encoded
CFA/RGB 8/12/16-bit fixtures passed. The focused Linux suite passed all 24 layouts
with 16-thread dispatch, including partial groups, malformed entropy, trailing
stuffing and newly mixed ordinary/paired CFA tiles within one image. Linux
production defaults remain eight threads. Host artifacts are under
`build-linux-release/vulkan-focus/`; final reports use `jpeg-workgroups-*`, with
`jpeg-workgroups-final-correctness.log` for the fixture result. Earlier experiments
use `jpeg-row-*`, `jpeg-word-*`, `jpeg-group-*`, `jpeg-refill-*` and
`jpeg-components-{mixed,filtered}-result.json`. The updated APK retains the ten-clip
resume session.

## Diagonal predictor and combined entropy consumption — 2026-11-15

The decoder now selects a constant diagonal-predictor body once per
single-component tile when its parsed JPEG predictor is 3. It generates this
body from the same checked decoder, retaining dynamic predictor handling for
other modes and layouts. Large tiles with an already-specialized predictor do
not add another body. The 12 MP benchmark frame uses predictor 3 in all 12,288
tiles; this choice comes from each tile's metadata, not a clip-name assumption.

The reservoir decoder also consumes a short Huffman prefix and its residual in
one update when both are already buffered. This removes the second refill check
and reservoir update from the common path. Category zero and category sixteen
retain their original semantics; long Huffman codes and short stream tails use
the previous checked path. This part applies to CFA, paired CFA and RGB, both
source decode and resident encoder roundtrips. Both optimizations are default;
header/bounds/status/entropy validation and CPU recovery remain active.

CPH2581 / 4096×3072 CFA, same complete 16×64 tiles, five synchronized iterations
after two warmups, matched previous-default/current-default comparison:

| Measurement | Previous default | Current default |
| --- | ---: | ---: |
| Source decode wall median | 30.12 ms | 27.95 ms |
| Resident JPEG re-decode wall median | 17.31 ms | 13.94 ms |
| GPU encode wall median | 28.02 ms | 27.64 ms |
| Complete resident roundtrip wall median | 75.84 ms | 70.68 ms |
| Gallery JPEG entropy GPU median | 17.36 ms | 15.02 ms |
| Gallery playback after seek | 20.29 fps | 20.90 fps |

The gallery comparison is separate, short full-resolution playback with
profiling and strict zero-copy enabled. Wall times exclude file I/O and pixel
comparison, and synchronize each measured stage. The small encode difference
is not an encoder optimization claim. Per-iteration values remain in the JSON
reports; these measurements do not predict every device or tile layout.

Incremental checks also favored both changes: diagonal specialization reduced
source decode 30.89→27.59 ms; combined reservoir consumption then reduced
27.69→26.68 ms in a separate pair. Use the matched table above for the combined
result rather than adding percentages across separate runs.

Android exact-code roundtrips, independent CPU decoding of GPU-encoded CFA/RGB
8/12/16-bit fixtures, and the wide-header fixture passed. The focused Linux suite
passed all 24 layouts, including malformed stuffing/trailing entropy rejection.
No wider codec matrix was repeated. Debug `jpegdynamicpredictor` restores the
per-sample predictor selection; `jpegseparateresidual` restores separate reservoir
updates. Both flags together reproduce the previous default. Artifacts are in
`build-linux-release/vulkan-focus/`, prefixed `jpeg-predictor-`, `jpeg-residual-`
and `jpeg-entropy-`; the final comparison is `jpeg-entropy-{previous,current}-result.json`
and `jpeg-entropy-gallery-{previous,current}.json`.

## Parallel JPEG entropy validation — 2026-11-15

Small GPU-parsed JPEG tiles now use the existing parallel byte-stuffing validator
instead of scanning entropy serially in the header parser. Large tiles already
used this pass. Header, stream-bounds, tile-status and malformed entropy checks
remain active by default, including surplus entropy after the decoded samples.
No raw pixels return to the host during decode/encode/re-decode.

CPH2581, same 12 MP / 16×64-tile clip, five synchronized roundtrip iterations:

| Measurement | Previous serial small-tile scan | Parallel scan |
| --- | ---: | ---: |
| Source decode wall median | 32.22 ms | 28.29 ms |
| Complete resident roundtrip wall median | 76.95 ms | 72.94 ms |
| Gallery header parser GPU median | 4.29 ms | 1.46 ms |
| Gallery playback after seek | 19.65 fps | 19.92 fps |

The header parser timer excludes the separate validator; wall times include all
decode work. Gallery entropy decode itself remained 17.38 ms. The gallery runs
were short full-resolution comparisons, so small FPS differences should not be
read as stable device limits.

Debug `jpegskipscan` / `RAWPREAMP_JPEG_SKIP_SCAN=1` skips the separate entropy
byte-stuffing scan for a trusted-input comparison. Header/bounds/status checks
remain, but malformed stuffing is no longer guaranteed to be rejected. It is
not the default. Against parallel validation, source decode measured 28.61 ms
with the scan and 29.28 ms without; roundtrip measured 72.84 vs 73.79 ms. Gallery
measured 19.92 vs 20.32 fps. These mixed small differences do not establish a
consistent overall benefit from skipping validation. `jpegserialvalidate`
restores the former small-tile parser scan for reproduction.

Rejected: a compressed-word register cache made negligible difference; fused
stuffing checks inside refill slowed resident re-decode from 16.42 to 19.32 ms.
Both candidates were removed. Android exact-code roundtrip/precision fixtures
passed with parallel and skipped scans. The 24 focused Linux fixtures passed,
now also rejecting bad stuffing inside entropy and in appended trailing entropy.
Logs/configurations are under `build-linux-release/vulkan-focus/` with prefixes
`jpeg-validation-`, `jpeg-fused-validation-`, and `jpeg-byte-`.

## Android tiled JPEG specialization and resident roundtrips — 2026-11-15

The entropy shader now generates constant-component bodies from its existing
checked decode loop. Known layouts specialize directly; GPU-parsed small tiles
select single-component or paired-CFA bodies once per tile, retaining the generic
body for other layouts. Predictor state can remain in scalar registers. The
reservoir, Huffman checks, reconstruction, stored precision and CPU recovery are
unchanged. This is the default on Android and Linux; Debug `jpegloops` restores
the component loop for comparison.

CPH2581 / Adreno, 4096×3072 CFA, 16-bit stored samples, 12,288 tiles of 16×64:

| Measurement | Component loop | Specialized |
| --- | ---: | ---: |
| Source decode, synchronized wall | 37.51 ms | 32.61 ms |
| GPU packed-buffer tiled encode | 28.09 ms | 27.77 ms |
| Resident JPEG re-decode | 23.24 ms | 16.09 ms |
| Complete decode → encode → decode | 89.05 ms | 77.32 ms |
| Gallery JPEG entropy GPU timer | 21.13 ms | 17.38 ms |
| Gallery playback after seek | 18.48 fps | 19.73 fps |

Roundtrip times are medians of five iterations after two warmups. File reading,
CPU reference decoding, fixture validation and pixel comparisons are excluded.
Each stage waits for GPU completion; these are latency measurements, not a
pipelined encoder throughput claim. Gallery used a separate short full-resolution
run per decoder with `profile,nohist,hidethumbnails,skipbackfill,preserveproxy,zerocopy,jpegprofile`.
The encode difference is small; no separate encode speedup is claimed.

The Debug `--tiled-jpeg-benchmark <config.json>` command now benchmarks the
original tiled DNG decode, resident packed-buffer encode and resident re-decode.
The stored bit depth is carried into resident metadata rather than assumed to be
16. Below 16 bits, encode timing includes identity gain-bake packing on the GPU.
Only small validation/status words return to the CPU during the measured chain;
GPU buffers hold raw pixels and encoded JPEG slots throughout. It compares the
original GPU decode against liblj92 and every roundtrip against the original
stored codes, with exact equality required. Six small CFA/RGB × 8/12/16-bit
fixtures and a 256-pixel-wide tile additionally check the GPU encoder using the
independent CPU JPEG decoder. Their compressed-byte download is outside timing.

This uncovered incorrect Android output from the dynamic CFA 8-bit and CFA/RGB
12-bit packing loops. Explicit packing for these common formats now passes the
independent exact-code checks. Direct JPEG header emission also masks each byte,
including dimensions above 255. The resident packed-buffer encoder uses a direct
slot shader and avoids compiling the unused assembly shader, whose pipeline
creation failed on this Android driver. Ordinary export retains its existing
assembly and CPU recovery paths.

Validation: Android precision/header/CPU interoperability fixtures and both 12 MP
roundtrip modes passed; gallery runs completed without errors; the focused Linux
JPEG suite passed all 24 CPU/GPU header-parsing fixtures. No broader codec matrix
was repeated for this JPEG change. APK and configs remain on the phone. Host
results and runners are under `build-linux-release/vulkan-focus/`, including
`jpeg-{loops,specialized}-result.json`, `jpeg-components-{loops,unrolled}.json`,
and `jpeg-components-final-correctness.log`.

## Raw word reads, processing setup and direct display coordinates — 2026-11-15

The shared raw unpacker now extracts a field from one 32-bit word, fetching the
next word only when the field crosses the boundary. This replaces repeated
byte extraction for UFR1/MC24 bit packing, MC24 lane words and MCRAW's 16-bit
blocks. Host payload validation and final-word padding still bound every read.
The previous byte shader remains a compile-failure fallback and diagnostic
`rawbytefields` override; original-stream CPU recovery remains the final fallback.

Inverse LOG tables now have an immutable per-worker cache keyed by stored white
and curve, capped at 128 KiB per worker. LOG-disabled/invalid levels return an
empty table without evicting an active one. Gallery setup, CPU recovery, preview
processing and DNG metadata reuse these exact values. Valid JSON metadata arrays
are read by reference, shading planes are moved into their result, and successful
Vulkan raw admission reuses its parsed frame metadata instead of parsing it
again. CPU recovery still parses the freshly decoded metadata. Missing neutral
and black arrays have deterministic identity/zero defaults so moved records do
not carry indeterminate values.

Shared-image color processing now samples the source pixel already owned by the
invocation. It retains the same output rotation, removing the forward output-index
construction and inverse coordinate mapping before color processing. Buffer and
thumbnail paths retain their existing mapping. Diagnostic `coordsold` selects
the prior shared-image mapping. No dither or color math was changed, and no
intermediate pixel readback was added. CFR duplicate handling was not changed.

Focused same-APK Android measurements:

| Stage / source | Previous | Current |
| --- | ---: | ---: |
| UFR1 12MP unpack | 1.174 ms | 1.099 ms |
| MC24 unpack | 0.962 ms | 0.800 ms |
| CFA 12MP gallery color, both A/B orders | 8.84–8.89 ms | 8.38–8.40 ms |
| Actual 4096×3072 HLG Native gallery color | 4.076 ms | 2.576 ms |
| HLG Native after-seek playback, 3-second windows | 58.37 FPS | 60.36 FPS |

The short CFA runs did not show an overall playback FPS gain; their results are
GPU-stage improvements only. The longer HLG comparison recorded 320/334 color
timer samples and no reported errors, with strict GPU-only presentation. The
desktop setup check measured inverse-table generation at about 0.75 ms for
16-bit white, eliminated on cache reuse. A synthetic 4×33×25 shading map plus
4096-entry source table parsed in 65.66 → 22.58 µs, with identical tested values;
these CPU setup numbers are not end-to-end playback claims.

Both platform builds pass. Linux's focused raw/resident-processing suite passes.
Android's decoder-only fixtures and 35 GPU processing/display cases pass, including
partial workgroups, all rotations, source-type switches and DCP shader fallback.
Final real-clip checks exercise both resident MC24 decode and forced original-stream
CPU recovery after metadata reuse. The inverse-table check covers both curves,
white/curve changes and invalid/LOG-disabled inputs, with exact table equality.

The broader Android CPU/GPU LOG oracle reports `Resident table gain/LOG differs
from CPU` on both the previous byte/shared predictor path and the candidate path.
Its existing tolerance was not relaxed. The decoder-only entry point excludes
that separate processing oracle; the broader standalone suite retains it.

A subgroup-register MC24 predictor decoded correctly but had slower measured
predictor stages on both GPUs, so it was removed. No new Linux playback speedup
is claimed. Artifacts are in `build-linux-release/vulkan-focus/`, notably
`word-fields-*`, `mc24-*`, `coords-*`, `hlg-coords-*`, `metadata-*`,
`log-cache-check.log` and `processing-smoke-results.log`. Debug builds expose
`--raw-decode-self-test <report>` (also an Android intent extra) to run the
focused decoder fixtures without the unrelated processing oracle.

## Fused CFA gain-field capture — 2026-11-15

CFA preparation now captures spatial gains in the same pass that applies them
to the frame. The next matching frame reads that field directly, removing the
separate field-generation dispatch and avoiding repeated interpolation during
cache warmup. This processing path is shared by MCRAW, MC24, Unspektra and CFA
tiled JPEG, including CPU- and texture-backed CFA sources. Pixels remain on
the GPU between processing stages. RGB processing and DNG bake interpolation
retain their existing two-observation policy.

One field is retained with the existing 64 MiB CFA/device-storage limit. After
two captures without reuse, capture waits for a repeated key, preventing
continuously changing maps from writing unused fields indefinitely. Allocation
or shader failures retain ordinary preparation. Buffer/texture switches ensure
the correct consuming shader even when the other source type populated the
field. Diagnostic `gaincaptureoff` selects the previous separate-build policy.

A short same-APK Android A/B on 4096×3072 Unspektra measured after-seek playback
**35.00 → 41.71 FPS (+19%)**, with GPU-only presentation and no reported errors.
Only that clip was benchmarked; the other codecs share the implementation but
have no new individual performance measurement. Linux and Android builds pass;
no new Linux GPU performance result is claimed.

All 35 focused Android cases pass, including cache invalidation, continuously
changing maps, buffer/texture switches, DCP thumbnail shader fallback, and
shared-image output at partial workgroup edges and all four rotations. A final
GPU comparison matched **19 full-resolution frames, 239,075,328 CFA samples,
zero differences**; only comparison status was downloaded. Artifacts are under
`build-linux-release/vulkan-focus/gain-capture-*`.

Shared-memory Bayer tiles and selective green-site gathers were pixel-exact
but slightly slower on Android, so neither was retained. A two-field cache did
not help this clip's changing two-frame map cadence and was also removed.

## Review cleanup — 2026-11-15

Fixed a thumbnail-only fallback case: if a DCP image shader had succeeded but
the later thumbnail shader rejected texture sampling, the cached gain field
was still bound at the texture variant's descriptor slot. Thumbnail submission
now selects the pass once and binds the field for that actual pass. It retains
GPU processing through the SSBO fallback.

The Android diagnostic now forces this late shader failure after shared-image
warmup and compares the resulting thumbnail exactly with the uncached SSBO
reference. All 30 focused Android cases pass. Temporary test GPU resources use
scoped cleanup, including error exits; obsolete quantization-difference logging
was removed. Linux/Android builds, the nine-case CPU bake check, and repository
and decoder-submodule whitespace checks pass. No codec benchmark matrix was run.

## Android Native Video regression fix — 2026-11-15

The generalized RGB preview shader initially declared a cached-field descriptor
on every frame and bound the small empty gain buffer when no field existed.
The Android HLG Native clip in the resume session then produced a black clip
thumbnail and an empty gallery, despite reporting successful GPU presentation.

Cached and uncached RGB passes are now separate. Only a pass with a complete
spatial field declares and binds that field; identity/no-map inputs and warmup
use the uncached pass. Thumbnail processing selects the same variant as the
frame, and cached shader creation failure retains the uncached pass. This keeps
spatial reuse available without the placeholder descriptor on Android.

The actual 4096×3072 HLG Native clip reproduced the failure before installation
and rendered correctly after the fix, also confirmed on-device by the user.
The focused Android check passes 29 cases with exact cached/uncached output.
Its image size now exceeds the placeholder buffer, and it includes identity
and absent RGB maps, nonblack output checks and GPU failure detection. Linux
and Android builds pass. No codec performance matrix was rerun.

## General spatial gain reuse — 2026-11-15

Spatial gain caching is part of image processing and does not depend on a
Vulkan decoder. CPU RGB and CFA baking, GPU RGB and CFA DNG baking, and gallery
RGB textures and Bayer buffers/textures now reuse stable spatial fields.
CPU-decoded images and native video use the same policy as raw decoder output.
The GPU bake and gallery paths share the cache storage/stability helper.

Only map interpolation is cached. Levels, linearization, LOG and the shared
RGB dither remain per-frame. Changing maps or geometry invalidate the field;
allocation, shader and storage-limit failures retain the original processing
pass. GPU fields are bounded to 64 MiB for CFA or 192 MiB for RGB, further
limited by the device storage-buffer limit. CPU fields use a two-entry cache
with a 192 MiB retained budget. A stable map warms up before field generation.
The performance measurements below apply to the original CFA gallery case;
no additional codec benchmark matrix was run for this generalization.

Focused validation passed nine CPU bake cases and 27 Android GPU bake/gallery
cases, including cold processing, field generation, reuse and map mutation.
The Android cases cover CPU arrays, resident buffers and textures, single and
multiple maps, and linear16/LOG12 output; cached and uncached outputs matched
exactly. Packed RGB now uses the same per-pixel RGB calculation as 8/16-bit
output, removing the separate scalar calculation that differed for multi-map
LOG output on Android. Dither is still a single value broadcast to RGB.
Linux builds pass; no Linux GPU runtime check was possible in this session.

## Cached spatial gains across CFA codecs — 2026-11-15

The shared Vulkan Bayer preparation path now caches the spatial gain product
for stable maps. MCRAW, MC24, UFR1 and CFA tiled JPEG all use this path. A GPU
pass generates the field; subsequent frames read it while continuing to apply
source linearization, per-frame black/scale, LOG quantization and dither.
No decoded image is downloaded.

The key includes every map's contents, dimensions, bounds, axes and pitch,
plus output dimensions, original geometry, proxy scale and CFA phase. Levels,
LOG and dither are excluded because they remain per-frame operations. The cache
waits for two consecutive identical spatial inputs, so continuously changing
maps retain the original GPU pass. It holds one field, capped at the lesser of
64 MiB and the device storage-buffer limit, and frees it when no eligible
spatial pass remains. Failed allocation or shader creation retains the original
GPU pass; CPU recovery remains the final fallback.

One focused same-APK Android comparison on the 4096×3072 Unspektra clip measured
median preparation time **8.7955 → 2.076 ms**, with after-seek throughput
**35.93 → 39.60 FPS**. The runs completed with GPU-only presentation and no
reported errors (82/87 timer samples). These are short warm-cache results;
other codecs share the implementation but were not rebenchmarked this round.

A diagnostic GPU comparison then matched all prepared samples against the
uncached pass: **18 full-resolution frames, 226,492,416 samples, zero differing
codes**. Only an eight-byte comparison result was read back. `gainfieldcheck`
enables this check; `gainfieldoff` selects the old pass. Both are diagnostic
flags. A focused changing-input fixture was also added for map contents, axes,
levels, LOG, dither, dimensions, scale and CFA patterns.

Linux and Android builds pass. Linux Vulkan device initialization was unavailable
in this session, so the new Linux changing-input GPU fixture could not execute;
no Linux GPU or performance result is claimed for this cache. The final adaptive cache also passed a full-resolution-to-proxy switch on
Android: 20 frames and 138,412,032 prepared samples matched exactly at both
4096×3072 and 2048×1536, with no reported errors. This check used a test-only
copy to avoid gallery path deduplication. Its results are recorded in
`build-linux-release/vulkan-focus/gain-final-verify.log`. Other runner/log artifacts
for this change are under the same persistent build directory.

## Focused JPEG reservoir tuning — 2026-11-14

The retained entropy reader prefills its existing 32-bit reservoir to 24 bits
before decoding the Huffman prefix, rather than targeting seven bits. This
amortizes refill work over prefix and residual decoding. Stuffed bytes, bounds,
short-tail validation, predictor math and GPU residency remain unchanged. A
failed specialized shader still uses the existing generic seven-bit reader.

One focused comparison on the Android 4096×3072 tiled-JPEG clip measured median
GPU decode time **21.575 → 21.141 ms** (51/53 timer samples). After-seek playback
was **16.57 → 16.83 FPS**. Both short runs completed with strict GPU-only
presentation and no reported errors (50/52 shared images). This is a modest
kernel improvement, not a sustained-throughput claim or an across-codec gain.

Larger 32/64-thread workgroups were slower than the retained eight-thread
workgroups. A previous-row shared cache was neutral (21.577 → 21.723 ms), and
three-byte batched loads regressed (21.571 → 24.384 ms). Those candidates and
their diagnostic overrides were removed.

The focused JPEG test entry point covers 24 fixtures with CPU and GPU header
parsing, ordinary CFA, paired CFA components, RGB, 12/16-bit data, unaligned
input, five-tile partial workgroups, narrow/wide rows and malformed tile tails.
All passed with 24-bit prefill. Run it without the full regression matrix:

```sh
RAWPREAMP_TEST_REQUIRE_VULKAN=1 RAWPREAMP_JPEG_DECODE_FOCUSED_TEST=1 ./build-linux-release/rawpreamp-lossless-jpeg-test
```

Instrumented builds accept `jpegprefill7` and `jpegprefill24` for comparison.
The APK with the retained default is installed on the test phone. New runner,
logs and JSON results live under `build-linux-release/vulkan-focus/`, which
survives session resumes; earlier `/tmp` benchmark artifacts may no longer exist.

## Faster shared Bayer reconstruction — 2026-11-14

The retained color kernel replaces per-channel neighbor loops with fixed gathers,
clamped loads and validity masks. Integer rounding is unchanged, including image
edges. This benefits MCRAW, UFR1, MC24 and CFA tiled JPEG through their common
resident renderer. RGB video processing and channel-shared dither are unchanged.

Same-APK Android full-resolution after-seek means, two short samples per mode
in opposite orders, using the storage-buffer transfer lookup in both modes:

| Source | Previous FPS | Retained kernel FPS |
| --- | ---: | ---: |
| MCRAW | 38.71 | 46.23 |
| Unspektra/UFR1 | 29.32 | 35.94 |
| MC24 | 19.11 | 21.41 |
| Tiled JPEG | 15.46 | 17.06 |

Median GPU color time fell from 11.68 to 7.11 ms (MCRAW), 15.36 to 8.81 ms
(UFR1), 15.44 to 8.77 ms (MC24), and 21.32 to 15.71 ms (JPEG). FFmpeg color
stage times stayed approximately 5.3/4.3 ms for HEVC/AV1; AV1 playback sample
rates varied and no video speedup is claimed.

Rejected candidates: an interior/edge branch split increased color time; four-slot
MC24 predictor histories increased predictor time even with a padded stride;
an exact R32F sRGB texture lookup gave no consistent GPU-stage benefit. Their
production changes were removed. The previous MC24 predictor and transfer buffer
lookup remain defaults. `bayerloops` selects the previous reconstruction in
instrumented builds for reproduction.

The retained gathers passed the Release Vulkan suite against the independent
scalar demosaic shader for all four CFA phases, four rotations, 16/8-bit output
and partial edges. Odd MC24 strip heights now extend the existing CPU/GPU
predictor tests. Raw/JPEG resident processing, proxies, decode-validity lifetime,
DCP and shared RGB dither checks passed. Android comparison runs completed with
strict GPU-only presentation and no reported errors. Linux was used for
correctness during gaming, not controlled FPS comparisons.

Artifacts are under `/tmp/raw-vulkan-validation/five-path/`, including
`gather-*`, `android-gather-*`, and `android-transfer-*-stages*`. These short
warm-cache results do not establish sustained thermal throughput. Future tuning
uses focused checks per change and one broad final validation, rather than
repeating the complete comparison matrix for every candidate.

## Review and cleanup — 2026-11-14

GPU image retirement and MC24 validation-buffer destruction now use the shared
submission lock, including release from consumer threads. The lock permits
nested retirement during submissions; separate pool locking was removed.
The upload cache also detects a changed GPU/target allocation instead of
relying exclusively on its callers to invalidate it. Resident proxy/download
operations reject undersized storage and use overflow-safe geometry checks;
the UFR1 entry point rejects unsupported CFA declarations.

Regression checks cover cross-thread and nested lease release, replacement
target uploads, partial-update contents, undersized resident buffers and
unsupported UFR1 CFA sizes. The Release Vulkan suite passed. The reviewed
Android APK completed the six-clip strict GPU-only run with 555 shared-image
presentations and no reported errors. Both platform builds and root/submodule
whitespace checks passed. Logs are in `five-path/review-*` under the validation
artifact directory. These checks do not claim a clean full-repository test run;
the separately documented pre-existing finalize assertion remains outside this
focused suite.

## Shared GPU resources — 2026-11-14

The raw, JPEG and proxy buffer paths now use `VulkanResidentImage`; the old JPEG
image name remains an alias for compatibility. Raw decoding no longer requires
the JPEG decoder header. Proxy reduction and explicit download live in
`VulkanResidentImage.cpp`, so raw proxies do not initialize JPEG entropy/parser
passes. CFA parity and RGB pixel-center sampling remain unchanged. Native-video
texture imports retain their direct hardware-buffer path.

A shared output pool serves MCRAW, MC24, UFR1, retained tiled JPEG and buffer
proxies on the existing shared device. A frame lease keeps its storage and
device alive. Buffers return to the pool only after the last consumer releases
them; acquisition skips buffers still in flight instead of waiting. The pool
retains at most four buffers/96 MiB, evicts older sizes when clips or proxies
change, and releases cached storage before retrying a failed allocation. This
budget applies to retired buffers, not images still leased by consumers.

`VulkanUploadCache` is shared by gallery gain/LOG/profile resources and DNG bake
gain maps/LOG lookup tables. It compares contents, not source addresses, and
uploads only the changed aligned range. A frame-seed-only change transfers one
word instead of the adjacent lookup tables. Allocation growth invalidates the
cache; other writers invalidate the DNG LOG lookup cache. Cached target buffers
are not passed through `pl_buf_recreate` while their contents must be preserved.
No decoded-pixel readback was added, and channel-shared RGB dither is unchanged.

### Same-APK Android comparison

Two 1.5-second after-seek samples per mode, measured in both orders over all six
full-resolution sources. These averages are short cache-warm observations, not
sustained thermal measurements or evidence of a speedup for every codec.

| Source | Pool/cache disabled FPS | Enabled FPS |
| --- | ---: | ---: |
| MCRAW | 38.08 | 38.62 |
| Unspektra/UFR1 | 29.18 | 29.56 |
| MC24 | 18.92 | 18.55 |
| Tiled JPEG | 15.38 | 15.42 |
| FFmpeg HEVC | 56.28 | 57.69 |
| FFmpeg AV1 | 47.44 | 53.82 |

All four suites completed with strict GPU-only presentation and no reported
errors (1,094 shared-image presentations disabled, 1,101 enabled). Raw/JPEG
throughput is broadly similar; AV1 playback remains variable between runs.
The final pool eviction/retry revision also passed a six-clip full-resolution
Android run (556 shared images) and a three-clip raw proxy run (381 shared
images), with strict GPU-only presentation and no reported errors. Pooling and
caching are enabled by default.

The final Release Vulkan suite passes, including live-frame non-aliasing,
recycling across size changes, cache invalidation and readback of a partial
four-byte update to verify neighboring data. Existing raw/JPEG, gain/LOG,
changing DCP table, shared-dither and proxy scale 2/4/8 checks pass. Linux was
used for correctness while gaming; no controlled Linux FPS claim is made.

Diagnostic builds accept `gpupooloff` and `uploadcacheoff` for same-build
comparisons. Session copies, JSON summaries, logs and runners are under
`/tmp/raw-vulkan-validation/five-path/resources-*` and
`/tmp/raw-vulkan-validation/five-path/android-resources-*`.

## Cross-codec comparison — 2026-11-14

Compared full-resolution MCRAW, MC24, Unspektra/UFR1, tiled lossless JPEG,
FFmpeg HEVC and FFmpeg AV1 using resume-session sources. Android baseline
results below are uncapped, cache-warm, 1.5-second samples after seeking, with
histograms/thumbnails disabled and gain/LOG enabled. Different resolutions,
color metadata and codecs mean these numbers describe the tested clips, not
an equal-work codec ranking.

| Source | Resolution | Android baseline FPS | Final vector RGB FPS | Decode path |
| --- | --- | ---: | ---: | --- |
| MCRAW | 4032×2264 | 38.30 | 37.76 | Vulkan compressed-raw unpack |
| Unspektra/UFR1 | 4096×3072 | 28.34 | 28.90 | Vulkan compressed-raw unpack |
| MC24 | 4096×3072 | 18.46 | 18.46 | Vulkan unpack/prediction |
| Tiled JPEG | 4096×3072 | 15.47 | 15.36 | Vulkan lossless entropy/prediction |
| FFmpeg HEVC / HLG | 4096×3072 | 52.58 | 55.89 | MediaCodec hardware → Android hardware buffer → Vulkan |
| FFmpeg AV1 / LOG60 | 3840×2160 | 51.18 | 47.71 | MediaCodec hardware → Android hardware buffer → Vulkan |

The complete Android baseline had 533 shared-image presentations, no reported
errors and strict zero-copy checks enabled. The final vector-RGB run completed with 544
shared-image presentations and no reported errors; small FPS changes are within
the variation of these short samples. Both native-video decoders logged
`hardware=true`, with fused GPU conversion enabled. Compressed input and small
metadata/validation words still cross the CPU; decoded images remain resident.

### Reuse and defaults retained

All paths already converge on the shared Vulkan device/context, gain/LOG and
color processing, DCP tables, and presentation transport. The common
`rawpreampGainGrid` helper now also supplies gain-grid coordinates for native
RGB display, CFA display preparation and multi-map DNG baking. It preserves
sample-center conventions, edge clamping and interpolation order.

Keep codec-specific decode kernels and memory import: raw bitstreams, JPEG
entropy streams and FFmpeg hardware surfaces need different unpacking. Keep
CFA gain/LOG preparation before demosaic, fused RGB processing, tiled display
traversal, cached DCP textures and the JPEG register bit reservoir as defaults.
Native Android retains fused hardware-buffer conversion. A matching video-only
comparison in the final APK measured HEVC **37.75 → 54.33 FPS** and AV1
**42.30 → 49.55 FPS** (unfused → fused, after seeking). Both two-clip suites
completed with strict GPU-only presentation and no reported errors (192 versus
245 shared-image presentations). Unsupported layouts
or failed GPU stages retain the existing CPU recovery paths; hardware-specific
imports are not forced through a common host staging buffer.

RGB texture processing now fetches and normalizes the three channels together,
shares primary/secondary map coordinates, and broadcasts **one scalar dither
sample** to all RGB channels before LOG quantization. The hash depends only on
pixel coordinates, frame seed and quality mode, never the channel. LOG
quantization can differ from the old scalar shader; bit-exact LOG agreement is
not required. Both low-cost and HQ modes retain their original hash.

Same-APK Android comparisons in both orders measured HEVC after-seek throughput
of 52.80/53.72 FPS for scalar processing versus 58.02/55.24 FPS for vector
processing. AV1 end-to-end throughput remained variable (scalar 59.43/50.90,
vector 49.26/51.25 FPS), so no AV1 playback-rate improvement is claimed.
Separate timing runs completed and reduced median GPU color-stage time from
**7.208 to 5.342 ms** for HEVC (158/177 samples) and **6.401 to 4.299 ms** for AV1
(83/85 samples). Vector RGB is the default; `rgbscalar` selects the old path in
diagnostic builds for reproduction.

The Release Vulkan suite passes with the vector path. Additional neutral-ramp
tests cover 10/12/16-bit LOG, both dither modes and two frame seeds. Every RGB
output code remains equal on neutral input, while changing the seed changes
the image, verifying active dither without independent color-channel noise.

Linux completed the six-clip gallery suite during gaming. HEVC logged Vulkan
hardware decode. AV1 initially selected Vulkan, hit a device-memory allocation
failure, then recovered through libdav1d software decode and GPU processing.
Linux FPS is therefore excluded from this comparison, and Linux AV1 is not
claimed as a successful GPU-only decode run.

Final Release verification covers resident raw/JPEG, gains, LOG, RGB texture
normalization, proxy scales 2/4/8 and six new RGB gain/LOG fixtures (secondary
maps, per-channel levels, 10/12/16-bit LOG, both dither modes and debug gain).
The coordinate-helper-only revision matched all six scalar captures byte for
byte; the newer vector revision permits LOG quantization changes and adds the
neutral-ramp assertions above. Runtime logs and session copies are under
`/tmp/raw-vulkan-validation/five-path/`. Original source clips are unchanged.

## Further full-resolution optimization — 2026-11-14

DCP tables now use cached RGBA32F textures and explicit full-precision shader interpolation. Integer texel fetches preserve the existing float values; hardware linear filtering and reduced-precision formats are not used. Table dimensions and values are checked before reusing cached textures.

The reverse-order Android comparison, using the same APK, 4096×3072 frames, gain/LOG and embedded DCP tables, measured **13.69 → 15.35 fps** (12.1% higher throughput). The forward-order comparison measured 13.63 → 15.39 fps. Median GPU color time fell from **28.35 ms to 21.34 ms**. Both comparisons retained strict shared-image zero-copy presentation.

The user was gaming during the Linux follow-up runs. Linux validation is functional evidence; its absolute FPS and small timing differences are not treated as isolated performance measurements in this follow-up.

All twelve changing-table, single-map and combined-map fixtures produced **bit-identical results to the previous GPU sampler** on Linux. The expanded look-only fixture has an existing five-code CPU floating-point difference in both GPU implementations; other fixtures remain within the previous bounds. The full Release raw/JPEG, transfer-curve and proxy suite passed with the new path enabled by default.

Unsupported texture formats/dimensions, upload failures or shader failures retain the previous GPU storage-buffer sampler. Forced texture-shader failure passed on Linux and Android with resident input, GPU presentation and no CPU color stage. CPU recovery remains the final fallback.

Diagnostics: `dcpbuffer` selects the previous sampler and `dcptexturefail` exercises GPU fallback in optimized Debug builds. `gallery_dcp_sampling` records the selected backend. Follow-up logs and fixture captures are in `/tmp/raw-vulkan-validation/`, including `*-dcp-texture*`, `*-dcp-buffer*`, `*-dcp-fallback*` and the `dcp-*-reference-*.bin` captures.

Tested 2026-11-13 on Linux with an NVIDIA GeForce RTX 3060 Ti and Android 16 on a OnePlus CPH2581. Original source clips and the existing resume session were preserved. The final optimized Debug APK was installed in place with app data retained.

## Full-resolution results

These are uncapped gallery throughput samples after seeking, using the same source files before and after optimization. Each sample lasts 1.5 seconds. Source files were warmed for the final raw comparison; a separate post-transfer run with half-second read stalls is retained in the artifacts. These are short cache-warm measurements, not sustained thermal or storage benchmarks. Gain baking and LOG are enabled; histogram and thumbnails are disabled.

| Source | Resolution | Linux before → after (fps) | Android before → after (fps) |
|---|---|---:|---:|
| MCRAW | 4032×2264 | 44.1 → 120.9 | 10.1 → 37.6 |
| Unspektra UFR1 | 4096×3072 | 37.2 → 70.6 | 3.4 → 29.3 |
| MC24 | 4096×3072 | 28.7 → 58.6 | 3.2 → 19.0 |

For tiled JPEG, the controlled comparison below changes the entropy reader in the same application build. Both paths use the optimized renderer and identical 4096×3072 input frames, with 16×64 tiles, gain, LOG and embedded DCP tables.

| JPEG entropy reader | Linux (fps) | Android (fps) |
|---|---:|---:|
| Previous reader | 67.7 | 12.4 |
| Register bit reservoir (default) | 72.6 | 13.5 |

Before the November 14 DCP optimization, the Android 12 MP JPEG result was below the recalled 15 fps result; these measurements do not establish a regression against that earlier capture or configuration. An original-decoder comparison on these same derived frames measured about 12.7 fps. The controlled same-build test measures an improvement from 12.4 to 13.5 fps. A final run without any decoder-enabling flag confirmed the default path at 13.4 fps on Android and 78.2 fps after seeking on Linux; the Linux opening window was slower, reinforcing the need to distinguish startup/cache effects from throughput. Android profiling identifies entropy decode and DCP color processing as the remaining dominant costs. A gain-only preset measured approximately 13.6 fps, so removing LOG alone did not materially change this result.

A matching 64×64 tile sequence was also tested; it was slower on this footage, so the application tile geometry was retained. The existing resume-session 3840×2160 sequence uses 256×240 tiles encoded as two JPEG components with separate DC Huffman tables. It previously fell back to CPU. This layout now decodes on GPU and matches CPU pixels exactly; its small number of large, serial entropy tasks remains a throughput limitation.

## Retained implementation changes

- Decode and supported processing stay resident on GPU; only final DNG payloads and small validation words are read back. GPU selection is the default, with CPU recovery for unsupported settings or failures.
- Bayer gain/LOG is prepared once per sample before demosaic. RGB reconstruction shares coordinate calculations, and the quantized sRGB output transfer is generated once as a GPU lookup table.
- Shared-image output traverses source tiles; rotated buffer output uses tiled traversal to avoid full-row memory strides. Linux retains direct DMA-BUF transport, while Android passes strict shared-image zero-copy checks.
- Raw unpack shaders specialize by codec. Common MC24 strips reconstruct rows concurrently with a shared-memory wavefront; larger supported strip layouts retain bounded serial reconstruction.
- JPEG supports paired CFA components and independent Huffman tables. Lookups use compact separate banks; the default entropy reader retains unused bits in registers. Large entropy validation scans run in parallel; small tiles avoid extra CPU header inspection.
- Raw renderer-failure recovery decodes the original recording on CPU, including proxy sampling and table lookup, rather than downloading GPU proxy pixels.

## Validation

- Final Release raw/JPEG suite passed with Vulkan required and resume-session raw frames. GPU raw source decode matches CPU exactly. Gain/LOG processing differences are at most one code value.
- Paired-component and ordinary CFA JPEG fixtures, resident RGB, packing/predictor variants, malformed entropy rejection, quantized-code validation, four rotations and partial edges passed.
- Proxy scales 2, 4 and 8 match the reference renders exactly in the tested gain/LOG/table combinations.
- All 65,536 sRGB input codes match the CPU transfer curve within one code; DCP comparisons differ by at most three 16-bit codes, consistent with the previous shader.
- MC24 and source-input tests passed.
- Final Linux gallery proxy, forced decode-failure fallback and renderer-failure recovery suites completed. Final Android proxy playback, forced decode-failure fallback and renderer-failure recovery suites also completed successfully; per-case JSON and source-recovery logs record the evidence.
- Root and decoder-submodule whitespace checks passed. A separate broad finalize/interpolation suite has an existing CFA-phase assertion failure reproduced on unchanged HEAD; the complete repository suite is not claimed to pass.

Android playback suites complete without reported frame/render errors. Android Qt/HWUI has also produced a post-suite process-shutdown mutex abort, including with the original JPEG decoder; this existing shutdown issue was not resolved by the decoding work.

## Reproduction and artifacts

Configure Debug with `-DRAWPREAMP_OPTIMIZED_DEBUG=ON` to retain diagnostics without the usual `-O0`. Release already uses its normal optimization settings. Run gallery perf with `profile,nohist,hidethumbnails,skipbackfill,preserveproxy`; add `zerocopy` on Android. Use `rawcpu`, `rawdecodefail`, or `rawcolorfail` for raw comparisons and recovery checks; `jpegcpu` selects CPU JPEG decode, `jpegoldentropy` selects the previous GPU reader, and `jpegprofile,rawstages` records GPU timings. Timing flags should be measured separately from throughput comparisons.

Focused verification command:

```sh
RAWPREAMP_TEST_REQUIRE_VULKAN=1 RAWPREAMP_TEST_RAW_SESSION='/home/leonards/.local/share/MotionCam/MotionCam Fuse/sessions/last_session.json' RAWPREAMP_JPEG_DECODE_RESIDENT_TEST=1 RAWPREAMP_GPU_PROXY_TEST=1 RAWPREAMP_GPU_JPEG_RGB_DECODE_TEST=1 ./build-linux-release/rawpreamp-lossless-jpeg-test
```

`RAWPREAMP_GPU_JPEG_REAL_DNG_BENCHMARK=<dng>` additionally compares real tiled JPEG DNGs against CPU decoding and checks invalid entropy rejection.

Detailed logs, per-case JSON, sessions and runners are under `/tmp/raw-vulkan-validation/`. The Android test files remain under the application private `files/raw-vulkan-test` directory for reproduction. The 12 MP JPEG sequence was losslessly retiled from resume-session DNG footage into test-owned directories; original DNGs were not changed.

Crop, higher-CFA processing, bad-pixel correction, camera-native export and source sample retention can require CPU fallback. CFA4 raw unpacking itself is supported. Android validation covers gallery playback and recovery; mounted DNG output was validated on Linux.
