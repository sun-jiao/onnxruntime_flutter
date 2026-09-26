import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';
void main() {
  test('benchmark separates initialization, warmup and samples', () async {
    var initialized = false, runs = 0;
    final result = await benchmarkOrt(initialize: () { initialized = true; },
      run: () async { expect(initialized, isTrue); runs++; }, warmupRuns: 2, iterations: 4);
    expect(runs, 6); expect(result.warmup, hasLength(2));
    expect(result.samples, hasLength(4)); expect(result.initialization, isNotNull);
    expect(() => result.samples.clear(), throwsUnsupportedError);
    final stats = OrtBenchmarkResult(samples: [5, 1, 3, 2, 4].map((i) => Duration(microseconds: i)).toList());
    expect(stats.p50.inMicroseconds, 3); expect(stats.p95.inMicroseconds, 5);
    expect(stats.mean.inMicroseconds, 3); expect(stats.min.inMicroseconds, 1);
    expect(stats.toJson()['p95Us'], 5);
    await expectLater(benchmarkOrt(run: () {}, iterations: 0), throwsArgumentError);
    await expectLater(benchmarkOrt(run: () => throw StateError('run')), throwsStateError);
  });
  test('native profiling produces readable operator trace and benchmark releases outputs', () async {
    final dir = Directory.systemTemp.createTempSync('ort-profile-');
    final options = OrtSessionOptions()..setIntraOpNumThreads(1)
      ..enableProfiling('${dir.path}/性能');
    final session = OrtSession.fromFile(File('test/fixtures/metadata.onnx'), options);
    final input = OrtValueTensor.createTensorWithDataList(Float32List.fromList([1, 2]), [1, 2]);
    final run = OrtRunOptions();
    try {
      final report = await benchmarkInference(session, run, {'input': input}, warmupRuns: 1, iterations: 3);
      expect(report.samples, hasLength(3));
      final path = session.endProfiling();
      expect(path, isNotNull);
      final events = jsonDecode(File(path!).readAsStringSync()) as List;
      expect(events.any((e) => e['cat'] == 'Node'), isTrue);
    } finally {
      await session.closeAsync(); input.release(); run.release(); options.release(); dir.deleteSync(recursive: true);
    }
  });
}
