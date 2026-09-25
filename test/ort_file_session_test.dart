import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:onnxruntime/src/util/native_path.dart';

void main() {
  const paths = ['model with spaces.onnx', '模型目录/语音模型.onnx', '模型🧠.onnx'];
  for (final path in paths) {
    for (final windows in [true, false]) {
      test('native path uses ${windows ? 'UTF-16' : 'UTF-8'} for $path', () {
        final allocator = _RecordingAllocator();
        final pointer =
            allocateOrtPath(path, isWindows: windows, allocator: allocator);
        try {
          final expected = windows ? path.codeUnits : utf8.encode(path);
          // Check the allocation before reading, so a wrong encoding cannot
          // cause an out-of-bounds read in the regression test itself.
          expect(
              allocator.byteCount, (expected.length + 1) * (windows ? 2 : 1));
          final actual = windows
              ? pointer.cast<Uint16>().asTypedList(expected.length + 1)
              : pointer.cast<Uint8>().asTypedList(expected.length + 1);
          expect(actual, [...expected, 0]);
        } finally {
          allocator.free(pointer);
        }
      });
    }
  }

  for (final name in ['model with spaces', '模型目录', '模型🧠']) {
    test('fromFile loads and runs a model in $name', () {
      final directory = Directory.systemTemp.createTempSync('ort-path-');
      final modelDirectory = Directory('${directory.path}/$name')..createSync();
      final model = File('example/assets/models/test_types_FLOAT.pb')
          .copySync('${modelDirectory.path}/$name.pb');
      final options = OrtSessionOptions()..setIntraOpNumThreads(1);
      OrtSession? session;
      OrtSession? reference;
      OrtRunOptions? runOptions;
      OrtValueTensor? input;
      final values = <OrtValue?>[];
      try {
        session = OrtSession.fromFile(model, options);
        reference = OrtSession.fromBuffer(model.readAsBytesSync(), options);
        runOptions = OrtRunOptions();
        input = OrtValueTensor.createTensorWithDataList(
            Float32List.fromList([1, 2, 3, 4, 5]), [1, 5]);
        expect(session.inputNames, reference.inputNames);
        expect(session.outputNames, reference.outputNames);
        final inputs = {session.inputNames.single: input};
        final actual = session.run(runOptions, inputs);
        values.addAll(actual);
        final expected = reference.run(runOptions, inputs);
        values.addAll(expected);
        expect(actual.map((value) => value!.value).toList(),
            expected.map((value) => value!.value).toList());
      } finally {
        for (final value in values) {
          value?.release();
        }
        input?.release();
        runOptions?.release();
        session?.release();
        reference?.release();
        options.release();
        directory.deleteSync(recursive: true);
      }
    });
  }

  test('fromFile still throws for a missing Unicode path', () {
    final directory = Directory.systemTemp.createTempSync('ort-path-');
    final options = OrtSessionOptions()..setIntraOpNumThreads(1);
    try {
      expect(
          () => OrtSession.fromFile(
              File('${directory.path}/不存在🧠.onnx'), options),
          throwsException);
    } finally {
      options.release();
      directory.deleteSync(recursive: true);
    }
  });
}

class _RecordingAllocator implements Allocator {
  int? byteCount;

  @override
  Pointer<T> allocate<T extends NativeType>(int byteCount, {int? alignment}) {
    this.byteCount = byteCount;
    return calloc.allocate<T>(byteCount, alignment: alignment);
  }

  @override
  void free(Pointer pointer) => calloc.free(pointer);
}
