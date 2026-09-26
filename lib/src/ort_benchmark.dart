import 'dart:async';
import 'ort_session.dart' if (dart.library.js_interop) 'web/ort_web.dart';
import 'ort_value.dart' if (dart.library.js_interop) 'web/ort_web.dart';

/// Wall-clock timings. Samples exclude initialization and warmup; they include
/// Dart scheduling/FFI overhead and any work performed by the run callback.
class OrtBenchmarkResult {
  final Duration? initialization;
  final List<Duration> warmup;
  final List<Duration> samples;
  OrtBenchmarkResult({this.initialization, List<Duration> warmup = const [],
      required List<Duration> samples})
      : warmup = List.unmodifiable(warmup), samples = List.unmodifiable(samples) {
    if (samples.isEmpty || [...warmup, ...samples].any((d) => d.isNegative) ||
        (initialization?.isNegative ?? false)) {
      throw ArgumentError('Timings must be nonnegative with at least one sample.');
    }
  }
  Duration get firstRun => warmup.isEmpty ? samples.first : warmup.first;
  Duration get mean => Duration(microseconds:
      (samples.fold<int>(0, (sum, d) => sum + d.inMicroseconds) / samples.length).round());
  /// Nearest-rank percentile. 0 returns the minimum, 100 the maximum.
  Duration percentile(double percent) {
    if (!percent.isFinite || percent < 0 || percent > 100) {
      throw ArgumentError.value(percent, 'percent');
    }
    final sorted = samples.toList()..sort();
    final index = (percent / 100 * sorted.length).ceil() - 1;
    return sorted[index < 0 ? 0 : index];
  }
  Duration get p50 => percentile(50);
  Duration get p95 => percentile(95);
  Duration get min => percentile(0);
  Duration get max => percentile(100);
  Map<String, Object?> toJson() => {
    'initializationUs': initialization?.inMicroseconds,
    'warmupUs': warmup.map((d) => d.inMicroseconds).toList(),
    'samplesUs': samples.map((d) => d.inMicroseconds).toList(),
    'meanUs': mean.inMicroseconds, 'p50Us': p50.inMicroseconds,
    'p95Us': p95.inMicroseconds,
  };
}

Future<OrtBenchmarkResult> benchmarkOrt({required FutureOr<void> Function() run,
    FutureOr<void> Function()? initialize, int warmupRuns = 3,
    int iterations = 20}) async {
  if (warmupRuns < 0 || iterations < 1) {
    throw ArgumentError('warmupRuns must be >= 0 and iterations >= 1.');
  }
  Future<Duration> time(FutureOr<void> Function() body) async {
    final clock = Stopwatch()..start();
    await body();
    return clock.elapsed;
  }
  final initialization = initialize == null ? null : await time(initialize);
  final warmup = <Duration>[];
  final samples = <Duration>[];
  for (var i = 0; i < warmupRuns; i++) { warmup.add(await time(run)); }
  for (var i = 0; i < iterations; i++) { samples.add(await time(run)); }
  return OrtBenchmarkResult(initialization: initialization,
      warmup: warmup, samples: samples);
}

/// Benchmarks strict inference and output disposal; caller owns session/inputs.
Future<OrtBenchmarkResult> benchmarkInference(OrtSession session,
    OrtRunOptions options, Map<String, OrtValue> inputs,
    {List<String>? outputNames, int warmupRuns = 3, int iterations = 20,
    FutureOr<void> Function()? initialize}) => benchmarkOrt(
  initialize: initialize, warmupRuns: warmupRuns, iterations: iterations,
  run: () async {
    final outputs = await session.runAsyncOrThrow(options, inputs, outputNames);
    for (final value in outputs) { value?.release(); }
  });
