import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

/// UI-independent examples. Only Dart snapshots leave a resource scope.
class DemoService {
  static bool _initialized = false;

  static OrtRuntimeCapabilities initializeRuntime() {
    if (!_initialized) {
      OrtEnv.instance.init();
      _initialized = true;
    }
    return OrtEnv.instance.capabilities;
  }

  static const types = ['FLOAT', 'DOUBLE', 'INT32', 'INT64', 'BOOL', 'STRING'];

  List parseValues(String type, String text) {
    if (!types.contains(type)) throw ArgumentError.value(type, 'type');
    final values = text.split(',').map((e) => e.trim()).toList();
    if (values.length != 5 || values.any((e) => e.isEmpty)) {
      throw const FormatException('Enter exactly five comma-separated values.');
    }
    switch (type) {
      case 'STRING':
        return values;
      case 'BOOL':
        return values.map((e) {
          if (e == 'true') return true;
          if (e == 'false') return false;
          throw const FormatException('Boolean values must be true or false.');
        }).toList();
      case 'INT32':
      case 'INT64':
        final numbers = values.map(int.parse).toList();
        final limit = type == 'INT32' ? 2147483647 : 9007199254740991;
        final lower = type == 'INT32' ? -2147483648 : -limit;
        if (numbers.any((e) => e < lower || e > limit)) {
          throw FormatException('Values must be between $lower and $limit.');
        }
        return type == 'INT32' ? Int32List.fromList(numbers) : numbers;
      default:
        final numbers = values.map(double.parse).toList();
        final result =
            type == 'DOUBLE'
                ? Float64List.fromList(numbers)
                : Float32List.fromList(numbers);
        if (result.any((e) => !e.isFinite)) {
          throw const FormatException(
            'Enter finite numbers within the type range.',
          );
        }
        return result;
    }
  }

  Future<T> withModel<T>(
    String asset,
    Future<T> Function(OrtSession session, OrtScope scope) body,
  ) async {
    final data = await rootBundle.load('assets/models/$asset');
    return usingOrtScope((scope) async {
      initializeRuntime();
      final options = scope.own(OrtSessionOptions(), (v) => v.release());
      options.setIntraOpNumThreads(1);
      options.setSessionGraphOptimizationLevel(
        GraphOptimizationLevel.ortEnableAll,
      );
      options.configureProviders([OrtProviderConfig.cpu()]);
      final session = OrtSession.fromBuffer(
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
        options,
      );
      scope.defer(session.closeAsync);
      await session.initialize();
      return body(session, scope);
    });
  }

  Future<Map<String, Object?>> tensor(
    String type,
    String text, {
    int? iterations,
  }) async {
    final values = parseValues(type, text);
    return withModel('test_types_$type.pb', (session, scope) async {
      final input = scope.own(
        OrtValueTensor.createTensorWithDataList(values, [1, 5]),
        (v) => v.release(),
      );
      final options = scope.own(OrtRunOptions(), (v) => v.release());
      final inputs = {session.inputNames.single: input};
      session.validateInputs(inputs);
      final watch = Stopwatch()..start();
      final outputs = await session.runAsyncOrThrow(options, inputs);
      late Object? result;
      try {
        result = outputs.first?.value;
      } finally {
        for (final output in outputs) {
          output?.release();
        }
      }
      watch.stop();
      final benchmark =
          iterations == null
              ? null
              : await benchmarkInference(
                session,
                options,
                inputs,
                warmupRuns: 3,
                iterations: iterations,
              );
      Map<String, Object?> describe(OrtValueInfo info) => {
        'name': info.name,
        'type': '${info.elementType}',
        'shape': info.shape,
        'symbolicDimensions': info.symbolicDimensions,
      };
      return {
        'model': 'test_types_$type.pb',
        'inputs': session.inputInfo.map(describe).toList(),
        'outputs': session.outputInfo.map(describe).toList(),
        'value': result,
        'firstInferenceUs': watch.elapsedMicroseconds,
        if (benchmark != null) 'benchmark': benchmark.toJson(),
      };
    });
  }

  Future<Map<String, Object?>> queue(OrtQueuePolicy policy) => withModel(
    'test_types_FLOAT.pb',
    (session, scope) async {
      final options = scope.own(OrtRunOptions(), (v) => v.release());
      final queue = OrtRealtimeSession(session, maxPending: 3, policy: policy);
      // Scope cleanup drains the queue before options and session are freed.
      scope.defer(() => queue.closeAsync(cancelPending: true));
      final pending = <Future<Map<String, Object?>>>[];
      Future<Map<String, Object?>> submit(int frame) async {
        final input = OrtValueTensor.createTensorWithDataList(
          Float32List.fromList(List.filled(5, frame.toDouble())),
          [1, 5],
        );
        try {
          final ticket = queue.enqueue(options, {
            session.inputNames.single: input,
          });
          // Cancellation is a no-op if rejectNew already dropped this ticket.
          if (frame == 11) ticket.cancel();
          final result = await ticket.result;
          final outputs = result.value;
          try {
            return {
              'frame': frame,
              'status': result.status.name,
              if (outputs != null) 'output': outputs.first?.value,
            };
          } finally {
            if (outputs != null) {
              for (final output in outputs) {
                output?.release();
              }
            }
          }
        } finally {
          input.release();
        }
      }

      for (var i = 0; i < 12; i++) {
        pending.add(submit(i));
      }
      final results = await Future.wait(pending);
      await queue.closeAsync();
      return {
        'policy': policy.name,
        'submitted': results.length,
        for (final status in OrtTaskStatus.values)
          status.name: results.where((e) => e['status'] == status.name).length,
        'frames': results,
      };
    },
  );

  Future<Map<String, Object?>> vad({
    required double threshold,
    required bool Function() cancelled,
    required void Function(double progress, double probability) onProgress,
  }) async {
    final pcm = await rootBundle.load('assets/audio/vad_example.pcm');
    return withModel('silero_vad.onnx', (session, scope) async {
      const sampleRate = 16000;
      const window = 1024;
      final samples = pcm.lengthInBytes ~/ 2;
      var hidden = Float32List(128);
      var cell = Float32List(128);
      final options = scope.own(OrtRunOptions(), (v) => v.release());
      final segments = <Map<String, double>>[];
      final probabilities = <double>[];
      int? speechStart;
      var processed = 0;
      for (var start = 0; start < samples && !cancelled(); start += window) {
        final count = math.min(window, samples - start);
        final frame = Float32List(window);
        for (var i = 0; i < count; i++) {
          frame[i] = pcm.getInt16((start + i) * 2, Endian.little) / 32768;
        }
        final probability = await usingOrtScope((frameScope) async {
          OrtValueTensor own(List data, List<int> shape) => frameScope.own(
            OrtValueTensor.createTensorWithDataList(data, shape),
            (v) => v.release(),
          );
          final inputs = {
            'input': own(frame, [1, window]),
            'sr': frameScope.own(
              OrtValueTensor.createTensorWithData(sampleRate),
              (v) => v.release(),
            ),
            'h': own(hidden, [2, 1, 64]),
            'c': own(cell, [2, 1, 64]),
          };
          session.validateInputs(inputs);
          final outputs = await session.runAsyncOrThrow(options, inputs);
          try {
            hidden =
                (outputs[1] as OrtValueTensor).toTypedData() as Float32List;
            cell = (outputs[2] as OrtValueTensor).toTypedData() as Float32List;
            return ((outputs[0] as OrtValueTensor).toTypedData() as Float32List)
                .first;
          } finally {
            for (final output in outputs) {
              output?.release();
            }
          }
        });
        probabilities.add(probability);
        if (probability >= threshold) speechStart ??= start;
        if (probability < threshold - 0.15 && speechStart != null) {
          segments.add({
            'start': speechStart / sampleRate,
            'end': start / sampleRate,
          });
          speechStart = null;
        }
        processed = start + count;
        onProgress(processed / samples, probability);
        // Allow rendering and cooperative cancellation between frames.
        await Future<void>.delayed(Duration.zero);
      }
      if (speechStart != null) {
        segments.add({
          'start': speechStart / sampleRate,
          'end': processed / sampleRate,
        });
      }
      return {
        'cancelled': cancelled(),
        'audioSeconds': samples / sampleRate,
        'processedSeconds': processed / sampleRate,
        'frames': probabilities.length,
        'threshold': threshold,
        'segments': segments,
        'probabilities': probabilities,
      };
    });
  }
}
