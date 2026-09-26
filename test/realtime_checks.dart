import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';
Future<void> checkRealtime(OrtSession session) async {
  final realtime = OrtRealtimeSession(session, closeSession: true);
  final input = OrtValueTensor.createTensorWithDataList(Float32List.fromList([1, 2]), [1, 2]);
  final options = OrtRunOptions();
  try {
    final first = realtime.enqueue(options, {'input': input});
    final second = realtime.enqueue(options, {'input': input});
    final last = realtime.enqueue(options, {'input': input});
    final closing = realtime.closeAsync();
    final results = await Future.wait([first.result, second.result, last.result]);
    try {
      expect(results.map((r) => r.status), [OrtTaskStatus.completed, OrtTaskStatus.dropped, OrtTaskStatus.completed]);
      expect(results.first.value!.single!.value, [[1.0, 2.0]]);
      expect(results.last.value!.single!.value, [[1.0, 2.0]]);
    } finally { for (final result in results) { for (final output in result.value ?? <OrtValue?>[]) { output?.release(); } } }
    await closing;
    await expectLater(session.initialize(), throwsStateError);
  } finally { await realtime.closeAsync(); input.release(); options.release(); }
}
