import 'dart:ffi';
import 'package:ffi/ffi.dart';
import '../bindings/onnxruntime_bindings_generated.dart' as bg;
import '../ort_env.dart';
import '../ort_model_info.dart';
import '../ort_status.dart';
import '../ort_value.dart';
import 'native_memory.dart';

OrtValueInfo readSessionValueInfo(
  Pointer<bg.OrtSession> session,
  int index,
  String name,
  bool input,
) {
  final api = OrtEnv.instance.ortApiPtr.ref;
  return usingNative((arena) {
    final out = arena<Pointer<bg.OrtTypeInfo>>();
    final get =
        input ? api.SessionGetInputTypeInfo : api.SessionGetOutputTypeInfo;
    OrtStatus.checkOrtStatus(
      get
          .asFunction<
            bg.OrtStatusPtr Function(
              Pointer<bg.OrtSession>,
              int,
              Pointer<Pointer<bg.OrtTypeInfo>>,
            )
          >()(session, index, out),
    );
    final info = out.value;
    try {
      final kind = arena<Int32>();
      OrtStatus.checkOrtStatus(
        api.GetOnnxTypeFromTypeInfo.asFunction<
          bg.OrtStatusPtr Function(Pointer<bg.OrtTypeInfo>, Pointer<Int32>)
        >()(info, kind),
      );
      final type = ONNXType.valueOf(kind.value);
      if (type != ONNXType.tensor && type != ONNXType.sparseTensor) {
        return OrtValueInfo(name, type);
      }
      final tensor = arena<Pointer<bg.OrtTensorTypeAndShapeInfo>>();
      OrtStatus.checkOrtStatus(
        api.CastTypeInfoToTensorInfo.asFunction<
          bg.OrtStatusPtr Function(
            Pointer<bg.OrtTypeInfo>,
            Pointer<Pointer<bg.OrtTensorTypeAndShapeInfo>>,
          )
        >()(info, tensor),
      );
      final dtype = arena<Int32>();
      final count = arena<Size>();
      OrtStatus.checkOrtStatus(
        api.GetTensorElementType.asFunction<
          bg.OrtStatusPtr Function(
            Pointer<bg.OrtTensorTypeAndShapeInfo>,
            Pointer<Int32>,
          )
        >()(tensor.value, dtype),
      );
      OrtStatus.checkOrtStatus(
        api.GetDimensionsCount.asFunction<
          bg.OrtStatusPtr Function(
            Pointer<bg.OrtTensorTypeAndShapeInfo>,
            Pointer<Size>,
          )
        >()(tensor.value, count),
      );
      final dims = arena<Int64>(count.value);
      final symbols = arena<Pointer<Char>>(count.value);
      OrtStatus.checkOrtStatus(
        api.GetDimensions.asFunction<
          bg.OrtStatusPtr Function(
            Pointer<bg.OrtTensorTypeAndShapeInfo>,
            Pointer<Int64>,
            int,
          )
        >()(tensor.value, dims, count.value),
      );
      OrtStatus.checkOrtStatus(
        api.GetSymbolicDimensions.asFunction<
          bg.OrtStatusPtr Function(
            Pointer<bg.OrtTensorTypeAndShapeInfo>,
            Pointer<Pointer<Char>>,
            int,
          )
        >()(tensor.value, symbols, count.value),
      );
      return OrtValueInfo(
        name,
        type,
        elementType: ONNXTensorElementDataType.valueOf(dtype.value),
        shape: List.generate(count.value, (i) => dims[i] < 0 ? null : dims[i]),
        symbolicDimensions: List.generate(count.value, (i) {
          final text =
              symbols[i] == nullptr
                  ? ''
                  : symbols[i].cast<Utf8>().toDartString();
          return text.isEmpty ? null : text;
        }),
      );
    } finally {
      api.ReleaseTypeInfo.asFunction<void Function(Pointer<bg.OrtTypeInfo>)>()(
        info,
      );
    }
  });
}
