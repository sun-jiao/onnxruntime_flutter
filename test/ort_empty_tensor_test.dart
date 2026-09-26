import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:onnxruntime/src/util/list_shape_extension.dart';

const _shapes = <List<int>>[
  [0, 2, 3, 4, 5, 6],
  [2, 0, 1, 1, 1, 1],
  [2, 1, 1, 0, 2, 3],
  [1, 1, 1, 1, 1, 0],
  [1, 1, 1, 1, 1, 1, 0],
  [2, 0, 1, 0, 1, 0],
];

const _expected = <List>[
  [],
  [[], []],
  [
    [
      [[]]
    ],
    [
      [[]]
    ],
  ],
  [
    [
      [
        [
          [[]]
        ]
      ]
    ]
  ],
  [
    [
      [
        [
          [
            [[]]
          ]
        ]
      ]
    ]
  ],
  [[], []],
];

// Use a separate isolate so a regression cannot block the test timeout.
void _readEmptyValues(List<Object> request) {
  final port = request[0] as SendPort;
  final native = request[1] as bool;
  try {
    final values = <List>[];
    for (final shape in _shapes) {
      if (native) {
        final tensor =
            OrtValueTensor.createTensorWithDataList(Float32List(0), shape);
        try {
          values.add(tensor.value as List);
        } finally {
          tensor.release();
        }
      } else {
        values.add(<double>[].reshape<double>(shape));
      }
    }
    port.send(values);
  } catch (error, stack) {
    port.send('$error\n$stack');
  }
}

void main() {
  for (final native in [false, true]) {
    test('high-rank empty ${native ? 'tensors' : 'lists'} retain shape',
        () async {
      final port = ReceivePort();
      final worker =
          await Isolate.spawn(_readEmptyValues, [port.sendPort, native]);
      try {
        final values = await port.first.timeout(const Duration(seconds: 5));
        expect(values, _expected);
        // Each outer element must have its own mutable list.
        final siblings = (values as List)[1] as List;
        expect(identical(siblings[0], siblings[1]), isFalse);
      } finally {
        worker.kill(priority: Isolate.immediate);
        port.close();
      }
    });
  }

  test('nonempty high-rank reshaping preserves row-major order', () {
    expect([1, 2, 3, 4].reshape<int>([1, 1, 1, 1, 2, 2]), [
      [
        [
          [
            [
              [1, 2],
              [3, 4]
            ]
          ]
        ]
      ]
    ]);
    expect([1, 2].reshape<int>([1, 1, 1, 1, 1, 1, 2]), [
      [
        [
          [
            [
              [
                [1, 2]
              ]
            ]
          ]
        ]
      ]
    ]);
  });

  test('scalar, vector and low-rank empty representations are unchanged', () {
    expect([1].reshape<int>([]), [1]);
    expect([1, 2].reshape<int>([2]), [1, 2]);
    expect(<int>[].reshape<int>([0]), <int>[]);
    expect(<int>[].reshape<int>([2, 0]), isA<List<List<int>>>());
    expect(<int>[].reshape<int>([2, 0]), [[], []]);
    expect(<int>[].reshape<int>([1, 1, 1, 1, 0]), [
      [
        [
          [[]]
        ]
      ]
    ]);
  });
}
