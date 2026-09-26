import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';

void checkModelInfo(OrtSession session) {
  final info = session.inputInfo.single;
  expect(info.name, 'input');
  expect(info.type, ONNXType.tensor);
  expect(info.elementType, ONNXTensorElementDataType.float);
  expect(info.shape, [null, 2]);
  expect(info.symbolicDimensions, ['batch', null]);
  expect(session.outputInfo.single.shape, [null, 2]);
  expect(() => info.shape![0] = 3, throwsUnsupportedError);
  final good = OrtValueTensor.createTensorWithDataList(Float32List(6), [3, 2]);
  final wrongShape = OrtValueTensor.createTensorWithDataList(Float32List(6), [2, 3]);
  final wrongRank = OrtValueTensor.createTensorWithDataList(Float32List(6), [6]);
  final wrongType = OrtValueTensor.createTensorWithDataList(Int32List(6), [3, 2]);
  try {
    session.validateInputs({'input': good});
    expect(good.shape, [3, 2]);
    expect(good.elementType, ONNXTensorElementDataType.float);
    expect(() => session.validateInputs({}), throwsArgumentError);
    expect(() => session.validateInputs({'other': good}), throwsArgumentError);
    for (final bad in [wrongShape, wrongRank, wrongType]) {
      expect(() => session.validateInputs({'input': bad}), throwsArgumentError);
    }
    good.release();
    expect(() => session.validateInputs({'input': good}), throwsStateError);
  } finally {
    for (final value in [good, wrongShape, wrongRank, wrongType]) { value.release(); }
  }
  session.release();
  expect(() => session.inputInfo, throwsStateError);
}
