import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:onnxruntime/src/web/model_info.dart';
import 'model_info_checks.dart';

void main() {
  test('native model descriptions and opt-in validation', () {
    final options = OrtSessionOptions()..setIntraOpNumThreads(1);
    final session = OrtSession.fromFile(File('test/fixtures/dynamic_identity.onnx'), options);
    options.release();
    try { checkModelInfo(session); } finally { session.release(); }
  });
  test('scalar and complex descriptions, parser/native parity', () {
    final options = OrtSessionOptions()..setIntraOpNumThreads(1);
    try {
      for (final fixture in ['scalar_identity', 'tensor_sequence', 'dynamic_identity']) {
        final bytes = File('test/fixtures/$fixture.onnx').readAsBytesSync();
        final parsed = ModelInfo.read(bytes);
        final session = OrtSession.fromBuffer(bytes, options);
        try {
          for (var i = 0; i < session.outputCount; i++) {
            final native = session.outputInfo[i];
            final web = parsed.outputInfo[i];
            expect(native.type, web.type);
            expect(native.elementType, web.elementType);
            expect(native.shape, web.shape);
            expect(native.symbolicDimensions, web.symbolicDimensions);
          }
          if (fixture == 'scalar_identity') expect(session.inputInfo.single.shape, isEmpty);
        } finally { session.release(); }
      }
    } finally { options.release(); }
  });
}
