import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime_example/main.dart';

void main() {
  testWidgets('all feature pages fit a narrow phone screen', (tester) async {
    tester.view.physicalSize = const Size(360, 800);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    await tester.pumpWidget(const MyApp());
    await tester.pumpAndSettle();
    for (final label in ['Benchmark', 'Queue', 'VAD', 'Tensors']) {
      await tester.tap(
        find.descendant(
          of: find.byType(NavigationBar),
          matching: find.text(label),
        ),
      );
      await tester.pumpAndSettle();
      expect(tester.takeException(), isNull);
      expect(find.byType(FilledButton), findsOneWidget);
    }
  });
}
