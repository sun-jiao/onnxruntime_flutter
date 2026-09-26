import 'dart:async';
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'realtime_checks.dart';
void main() {
  for (final policy in OrtQueuePolicy.values) {
    test('bounded queue $policy preserves active work and completes dropped tickets', () async {
      final queue = OrtTaskQueue<int>(maxPending: 1, policy: policy);
      final gate = Completer<int>();
      final first = queue.submit(() => gate.future);
      final second = queue.submit(() async => 2);
      final third = queue.submit(() async => 3);
      expect(queue.pendingCount, 1); expect(queue.activeCount, 1);
      expect(first.cancel(), isFalse);
      final closing = queue.closeAsync();
      expect(() => queue.submit(() async => 4), throwsStateError);
      gate.complete(1);
      final results = await Future.wait([first.result, second.result, third.result]);
      await closing;
      expect(results.first.value, 1);
      expect(results[policy == OrtQueuePolicy.rejectNew ? 2 : 1].status, OrtTaskStatus.dropped);
      expect(results[policy == OrtQueuePolicy.rejectNew ? 1 : 2].status, OrtTaskStatus.completed);
      expect(queue.activeCount, 0); expect(queue.pendingCount, 0);
    });
  }
  test('cancellation, close cancellation and errors do not stop draining', () async {
    final queue = OrtTaskQueue<int>(maxPending: 3, policy: OrtQueuePolicy.dropOldest);
    final gate = Completer<int>();
    final first = queue.submit(() => gate.future);
    final errorCheck = expectLater(first.result, throwsStateError);
    final second = queue.submit(() async => 2);
    final third = queue.submit(() async => 3);
    expect(second.cancel(), isTrue); expect(second.cancel(), isFalse);
    final closing = queue.closeAsync(cancelPending: true);
    expect((await second.result).status, OrtTaskStatus.cancelled);
    expect((await third.result).status, OrtTaskStatus.cancelled);
    gate.completeError(StateError('failed'));
    await errorCheck; await closing;
  });
  test('native real-time inference drops only pending requests', () async {
    final options = OrtSessionOptions()..setIntraOpNumThreads(1);
    final session = OrtSession.fromFile(File('test/fixtures/metadata.onnx'), options);
    options.release();
    await checkRealtime(session);
  });
}
