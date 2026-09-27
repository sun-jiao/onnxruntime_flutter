# ONNX Runtime Lab

The plugin's example is an interactive Material 3 application with four feature
pages. All models and the audio sample are bundled; no model download, account,
microphone permission or additional package is required.

## Run

From `example/`:

```sh
flutter pub get
flutter run -d linux
# Or select an Android/iOS/macOS/Windows device:
flutter devices
# Browser:
flutter run -d chrome
```

Use the parent package's platform prerequisites. Desktop builds may download the
native runtime on the first build. The Web entry point loads ONNX Runtime Web
1.23.2 from the CDN, so Web needs network access unless those runtime files are
self-hosted (see the parent README). All demo sessions explicitly use CPU/WASM.
The runtime card shows compiled provider availability, not actual GPU execution.

## Features

| Page | What to try | APIs demonstrated |
| --- | --- | --- |
| Tensors | Choose FLOAT, DOUBLE, INT32, INT64, BOOL or STRING; edit five values; inspect model names, types, shapes and outputs | `inputInfo`, `outputInfo`, `validateInputs`, `runAsyncOrThrow` |
| Benchmark | Choose 10–100 measured runs; compare mean, P50 and P95; copy all samples as JSON | `benchmarkInference`, 3 excluded warmups |
| Queue | Submit a 12-frame burst with each backpressure policy; inspect completed/dropped/cancelled frames | `OrtRealtimeSession`, `OrtQueuePolicy`, ticket cancellation |
| VAD | Adjust speech threshold; analyze bundled audio; inspect probability chart and speech intervals; stop between frames | Stateful Silero inference, `toTypedData`, scoped tensors |

Errors are displayed in the app, controls are disabled during a run, and the last
eight runs remain visible. Each page retains its latest result. Copy JSON exports
the complete result, including timings or per-frame probabilities.

### Interpreting results

- Identity models have shape `[1, 5]` and return their input. The editor accepts
  comma-separated values, so string values cannot themselves contain commas.
  INT64 is restricted to ±9007199254740991 for identical native/Web input handling.
- Benchmark session initialization is outside the measured interval. One preview
  inference precedes the three benchmark warmups. Samples include async scheduling
  and output disposal; small identity models mainly expose wrapper overhead.
- Queue capacity is three **pending** requests plus one active request.
  `keepLatest` retains only one pending request. The final ticket is cancelled if
  still pending; with `rejectNew` it has already been dropped, so cancellation
  does not succeed. This is a deterministic burst demonstration, not a camera feed.
- VAD uses 16 kHz signed little-endian mono PCM, normalized to Float32. A 1024-sample
  window is 64 ms. The final partial window is zero-padded but timestamps are
  clamped to the actual recording duration. Hidden/cell state starts fresh for
  every run and is carried between frames; speech ends below threshold minus
  0.15. There is no additional silence smoothing or padding. Stopping returns a
  partial analysis after active inference finishes.
- VAD frames are processed sequentially. Dropping frames with a realtime queue
  would break this recurrent model's state continuity; the queue page uses
  independent identity-model frames instead.

## Code structure and ownership

- `lib/main.dart`: responsive feature pages, controls, result rendering, chart,
  clipboard export, progress, cancellation and error handling.
- `lib/demo_service.dart`: model loading, input parsing, inference, benchmarking,
  queue management and audio preprocessing, independently testable from the UI.
- `test/`: real-model smoke tests and a narrow-screen widget test.

`OrtEnv` is initialized once and retained for the application's lifetime. Each
operation opens a session, explicitly awaits initialization and closes it with
`usingOrtScope` / `closeAsync`. All inference futures are awaited before their
inputs/options are released. Queue inputs survive until their tickets settle,
including dropped/cancelled tickets. Every returned output is released after
copying its data to Dart. Leaving the UI requests VAD cancellation without
freeing resources still used by active inference.

## Verify

```sh
flutter analyze
flutter test
flutter build web
```

Native tests need the host's ONNX Runtime library. On Linux, for example:

```sh
ORT_TEST_LIBRARY_PATH=/absolute/path/to/libonnxruntime.so.1.30.0 flutter test
```

The tests exercise six identity models, input overflow and malformed input,
benchmark sample counts, all queue policies, repeatable VAD state, final partial
frames, cooperative cancellation and layout on a 360-pixel phone screen. A Web
build verifies compilation; runtime browser/device behavior should also be
checked on the intended target.
