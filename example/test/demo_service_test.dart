import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:onnxruntime_example/demo_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final service = DemoService();

  test('rejects malformed input and numeric overflow before inference', () {
    for (final input in ['1,2', '1,2,3,4,NaN', '1,2,3,4,1e100']) {
      expect(() => service.parseValues('FLOAT', input), throwsFormatException);
    }
    expect(
      () => service.parseValues('INT32', '1,2,3,4,2147483648'),
      throwsFormatException,
    );
    expect(
      () => service.parseValues('INT64', '1,2,3,4,9007199254740992'),
      throwsFormatException,
    );
    expect(
      () => service.parseValues('BOOL', 'true,false,0,true,false'),
      throwsFormatException,
    );
  });

  test('all six bundled identity models preserve input values', () async {
    for (final type in DemoService.types) {
      final text =
          type == 'BOOL'
              ? 'true,false,true,false,true'
              : type == 'STRING'
              ? 'hello,ONNX,Flutter,世界,runtime'
              : '1,2,-3,-99,99999';
      final result = await service.tensor(type, text);
      expect(result['value'], [service.parseValues(type, text)]);
      expect((result['inputs'] as List).single['shape'], [1, 5]);
    }
  });

  test('benchmark records exactly the requested measured runs', () async {
    final result = await service.tensor('FLOAT', '1,2,3,4,5', iterations: 5);
    final timings = result['benchmark'] as Map;
    expect(timings['samplesUs'], hasLength(5));
    expect(timings['warmupUs'], hasLength(3));
    expect(timings['p95Us'], greaterThanOrEqualTo(timings['p50Us']));
  });

  test(
    'queue policies settle every frame and preserve completed values',
    () async {
      for (final policy in OrtQueuePolicy.values) {
        final result = await service.queue(policy);
        expect(result['submitted'], 12);
        expect(
          (result['completed'] as int) +
              (result['dropped'] as int) +
              (result['cancelled'] as int),
          12,
        );
        expect(result['dropped'], greaterThan(0));
        for (final frame in result['frames'] as List) {
          if (frame['status'] == 'completed') {
            expect(frame['output'], [
              List.filled(5, (frame['frame'] as int).toDouble()),
            ]);
          }
        }
      }
    },
  );

  test(
    'VAD consumes final partial frame and resets state on repeated runs',
    () async {
      Future<Map<String, Object?>> run() => service.vad(
        threshold: 0.5,
        cancelled: () => false,
        onProgress: (_, __) {},
      );
      final first = await run();
      final second = await run();
      expect(first['processedSeconds'], first['audioSeconds']);
      expect(first['segments'], isNotEmpty);
      expect(second['segments'], first['segments']);
      expect(second['probabilities'], first['probabilities']);
    },
  );

  test(
    'VAD cancellation returns partial results and permits a fresh run',
    () async {
      var stopped = false;
      final result = await service.vad(
        threshold: 0.5,
        cancelled: () => stopped,
        onProgress: (_, __) => stopped = true,
      );
      expect(result['cancelled'], true);
      expect(result['frames'], 1);
      final fresh = await service.tensor('FLOAT', '1,2,3,4,5');
      expect(fresh['value'], [
        [1.0, 2.0, 3.0, 4.0, 5.0],
      ]);
    },
  );
}
