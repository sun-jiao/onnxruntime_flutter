import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';

Future<void> checkClose(OrtSession session) async {
  final input = OrtValueTensor.createTensorWithDataList(Float32List.fromList([1, 2]), [1, 2]);
  final options = OrtRunOptions();
  try {
    final pending = session.runAsyncOrThrow(options, {'input': input});
    session.release();
    final closed = session.closeAsync();
    expect(identical(closed, session.closeAsync()), isTrue);
    final result = await pending;
    try { expect(result.single!.value, [[1.0, 2.0]]); }
    finally { for (final value in result) { value?.release(); } }
    await closed;
    await expectLater(session.runAsyncOrThrow(options, {'input': input}), throwsStateError);
  } finally { await session.closeAsync(); input.release(); options.release(); }
}
