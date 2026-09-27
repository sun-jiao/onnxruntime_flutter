import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'demo_service.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'ONNX Runtime Lab',
    debugShowCheckedModeBanner: false,
    theme: ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xff4263eb)),
      scaffoldBackgroundColor: const Color(0xfff4f6fb),
      inputDecorationTheme: const InputDecorationTheme(
        border: OutlineInputBorder(),
      ),
    ),
    home: const DemoPage(),
  );
}

class DemoPage extends StatefulWidget {
  const DemoPage({super.key});
  @override
  State<DemoPage> createState() => _DemoPageState();
}

class _DemoPageState extends State<DemoPage> {
  final _service = DemoService();
  final _input = TextEditingController(text: '1, 2, -3, -99, 99999');
  final _history = <String>[];
  final _results = <int, Map<String, Object?>>{};
  int _page = 0;
  String _type = 'FLOAT';
  OrtQueuePolicy _policy = OrtQueuePolicy.keepLatest;
  int _iterations = 20;
  double _threshold = 0.5;
  double _progress = 0;
  double _probability = 0;
  bool _busy = false;
  bool _cancelled = false;
  String? _error;
  String _runtime = 'Loading runtime…';
  static const _titles = [
    'Tensor explorer',
    'Benchmark',
    'Realtime queue',
    'Voice activity',
  ];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      try {
        final info = DemoService.initializeRuntime();
        setState(
          () =>
              _runtime =
                  '${info.runtimeVersion} · ${info.backend}\n${info.providerNames.join(', ')}',
        );
      } catch (error) {
        setState(() {
          _runtime = 'Runtime unavailable';
          _error = '$error';
        });
      }
    });
  }

  @override
  void dispose() {
    _cancelled = true;
    _input.dispose();
    // Operations drain their sessions even after UI disposal.
    // The shared OrtEnv lives for the lifetime of this application.
    super.dispose();
  }

  Future<void> _run() async {
    if (_busy) return;
    final page = _page;
    setState(() {
      _busy = true;
      _cancelled = false;
      _error = null;
      _progress = 0;
      _results.remove(page);
    });
    final watch = Stopwatch()..start();
    try {
      final Map<String, Object?> result;
      switch (page) {
        case 0:
          result = await _service.tensor(_type, _input.text);
          break;
        case 1:
          result = await _service.tensor(
            _type,
            _input.text,
            iterations: _iterations,
          );
          break;
        case 2:
          result = await _service.queue(_policy);
          break;
        default:
          result = await _service.vad(
            threshold: _threshold,
            cancelled: () => _cancelled,
            onProgress: (progress, probability) {
              if (mounted) {
                setState(() {
                  _progress = progress;
                  _probability = probability;
                });
              }
            },
          );
      }
      if (!mounted) return;
      setState(() {
        _results[page] = result;
        _history.insert(
          0,
          '${_titles[page]} · ${watch.elapsedMilliseconds} ms'
          '${result['cancelled'] == true ? ' · stopped' : ' · complete'}',
        );
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _error = '$error';
        _history.insert(0, '${_titles[page]} · failed');
      });
    } finally {
      if (mounted) {
        setState(() {
          _busy = false;
          if (_history.length > 8) _history.removeRange(8, _history.length);
        });
      }
    }
  }

  Widget _panel(String title, List<Widget> children) => Card(
    elevation: 0,
    child: Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(title, style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 16),
          ...children,
        ],
      ),
    ),
  );

  Widget _controls() {
    if (_page < 2) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Text(
            'Run the bundled identity models with shape [1, 5]. '
            'Inspect tensor types, input validation and outputs.',
          ),
          const SizedBox(height: 20),
          DropdownButtonFormField<String>(
            // Keep compatibility with the minimum Flutter 3.29 SDK.
            // ignore: deprecated_member_use
            value: _type,
            decoration: const InputDecoration(labelText: 'Tensor type'),
            items:
                DemoService.types
                    .map(
                      (type) =>
                          DropdownMenuItem(value: type, child: Text(type)),
                    )
                    .toList(),
            onChanged:
                _busy
                    ? null
                    : (value) => setState(() {
                      _type = value!;
                      _input.text =
                          _type == 'BOOL'
                              ? 'true, false, true, false, true'
                              : _type == 'STRING'
                              ? 'hello, ONNX, Flutter, 世界, runtime'
                              : '1, 2, -3, -99, 99999';
                    }),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _input,
            enabled: !_busy,
            minLines: 1,
            maxLines: 3,
            decoration: const InputDecoration(
              labelText: 'Five comma-separated values',
            ),
          ),
          const SizedBox(height: 12),
          const Text(
            'INT64 uses the exact integer range shared by native and Web. '
            'Strings cannot contain commas in this editor.',
          ),
          if (_page == 1) ...[
            const SizedBox(height: 20),
            Text('$_iterations measured runs · 3 warmups'),
            Slider(
              value: _iterations.toDouble(),
              min: 10,
              max: 100,
              divisions: 9,
              label: '$_iterations',
              onChanged:
                  _busy ? null : (v) => setState(() => _iterations = v.round()),
            ),
            const Text(
              'Initialization and warmups are excluded. Timings include '
              'async scheduling and output disposal. Identity models mostly measure wrapper overhead.',
            ),
          ],
        ],
      );
    }
    if (_page == 2) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Text(
            'Submit 12 independent frames with capacity for 3 pending requests. '
            'Cancel the final pending ticket. Inspect completed, dropped and cancelled results.',
          ),
          const SizedBox(height: 20),
          DropdownButtonFormField<OrtQueuePolicy>(
            // Keep compatibility with the minimum Flutter 3.29 SDK.
            // ignore: deprecated_member_use
            value: _policy,
            decoration: const InputDecoration(labelText: 'Backpressure policy'),
            items:
                OrtQueuePolicy.values
                    .map(
                      (policy) => DropdownMenuItem(
                        value: policy,
                        child: Text(policy.name),
                      ),
                    )
                    .toList(),
            onChanged:
                _busy ? null : (value) => setState(() => _policy = value!),
          ),
          const SizedBox(height: 16),
          const Text(
            'keepLatest retains only the newest pending frame. dropOldest keeps '
            'a bounded FIFO. rejectNew preserves queued frames. Cancellation cannot interrupt active inference.',
          ),
        ],
      );
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Text(
          'Analyze the bundled 16 kHz mono PCM recording with Silero VAD. '
          'Recurrent state is carried between 64 ms windows. The final window is zero-padded. '
          'No microphone permission is needed.',
        ),
        const SizedBox(height: 16),
        Text('Speech threshold: ${_threshold.toStringAsFixed(2)}'),
        Slider(
          value: _threshold,
          min: 0.2,
          max: 0.9,
          divisions: 14,
          onChanged:
              _busy ? null : (value) => setState(() => _threshold = value),
        ),
        const Text(
          'Speech ends below threshold − 0.15. Segments have 64 ms resolution; '
          'this demo does not add silence smoothing or padding.',
        ),
        const SizedBox(height: 16),
        LinearProgressIndicator(value: _progress),
        const SizedBox(height: 8),
        Text(
          '${(_progress * 100).round()}% processed · speech probability ${_probability.toStringAsFixed(3)}',
        ),
      ],
    );
  }

  Widget _result(Map<String, Object?> result) {
    final benchmark = result['benchmark'] as Map<String, Object?>?;
    final segments = result['segments'] as List<Map<String, double>>?;
    return _panel('Results', [
      if (benchmark != null)
        Wrap(
          spacing: 12,
          runSpacing: 8,
          children: [
            for (final key in ['meanUs', 'p50Us', 'p95Us'])
              Chip(
                label: Text(
                  '${key.replaceAll('Us', '')}: ${((benchmark[key] as num) / 1000).toStringAsFixed(3)} ms',
                ),
              ),
          ],
        ),
      if (segments != null) ...[
        Semantics(
          label: 'Speech probability per audio frame, from zero to one',
          child: SizedBox(
            height: 140,
            child: CustomPaint(
              painter: _ProbabilityPainter(
                result['probabilities'] as List<double>,
                result['threshold'] as double,
                Theme.of(context).colorScheme.primary,
              ),
            ),
          ),
        ),
        const SizedBox(height: 8),
        const Text(
          'Audio time → · Speech probability 0–1 · Horizontal line: threshold',
        ),
        const SizedBox(height: 12),
        Text(
          '${segments.length} speech segments${result['cancelled'] == true ? ' (partial analysis)' : ''}',
        ),
        if (segments.isEmpty)
          const Text('No speech detected at this threshold.'),
        for (final segment in segments)
          Text(
            '${segment['start']!.toStringAsFixed(2)} – ${segment['end']!.toStringAsFixed(2)} s',
          ),
      ],
      Align(
        alignment: Alignment.centerLeft,
        child: TextButton.icon(
          icon: const Icon(Icons.copy),
          label: const Text('Copy JSON'),
          onPressed: () async {
            try {
              await Clipboard.setData(
                ClipboardData(
                  text: const JsonEncoder.withIndent('  ').convert(result),
                ),
              );
              if (mounted) {
                ScaffoldMessenger.of(
                  context,
                ).showSnackBar(const SnackBar(content: Text('Result copied')));
              }
            } catch (error) {
              if (mounted) {
                setState(() => _error = 'Clipboard unavailable: $error');
              }
            }
          },
        ),
      ),
      ConstrainedBox(
        constraints: const BoxConstraints(maxHeight: 420),
        child: SingleChildScrollView(
          child: SelectableText(
            const JsonEncoder.withIndent('  ').convert(result),
            style: const TextStyle(fontFamily: 'monospace', fontSize: 13),
          ),
        ),
      ),
    ]);
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('ONNX Runtime Lab')),
    body: SafeArea(
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 1000),
          child: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              _panel('Runtime', [
                Text(_runtime),
                const SizedBox(height: 8),
                const Text(
                  'CPU sessions · strict async inference · explicit resource cleanup',
                ),
              ]),
              _panel(_titles[_page], [
                _controls(),
                const SizedBox(height: 24),
                Wrap(
                  spacing: 12,
                  runSpacing: 8,
                  children: [
                    FilledButton.icon(
                      onPressed: _busy ? null : _run,
                      icon: const Icon(Icons.play_arrow),
                      label: Text(_busy ? 'Running…' : 'Run ${_titles[_page]}'),
                    ),
                    if (_busy && _page == 3)
                      OutlinedButton.icon(
                        onPressed:
                            _cancelled
                                ? null
                                : () => setState(() => _cancelled = true),
                        icon: const Icon(Icons.stop),
                        label: Text(
                          _cancelled ? 'Stopping…' : 'Stop after frame',
                        ),
                      ),
                  ],
                ),
                if (_busy) ...[
                  const SizedBox(height: 16),
                  const LinearProgressIndicator(),
                ],
              ]),
              if (_error != null)
                _panel('Unable to complete', [
                  SelectableText(
                    _error!,
                    style: TextStyle(
                      color: Theme.of(context).colorScheme.error,
                    ),
                  ),
                ]),
              if (_results[_page] != null) _result(_results[_page]!),
              if (_history.isNotEmpty)
                _panel(
                  'Recent runs',
                  _history
                      .map(
                        (entry) => Padding(
                          padding: const EdgeInsets.symmetric(vertical: 4),
                          child: Text(entry),
                        ),
                      )
                      .toList(),
                ),
            ],
          ),
        ),
      ),
    ),
    bottomNavigationBar: NavigationBar(
      selectedIndex: _page,
      onDestinationSelected:
          _busy
              ? null
              : (value) => setState(() {
                _page = value;
                _error = null;
              }),
      destinations: const [
        NavigationDestination(icon: Icon(Icons.data_array), label: 'Tensors'),
        NavigationDestination(icon: Icon(Icons.speed), label: 'Benchmark'),
        NavigationDestination(icon: Icon(Icons.queue), label: 'Queue'),
        NavigationDestination(icon: Icon(Icons.graphic_eq), label: 'VAD'),
      ],
    ),
  );
}

class _ProbabilityPainter extends CustomPainter {
  final List<double> values;
  final double threshold;
  final Color color;
  const _ProbabilityPainter(this.values, this.threshold, this.color);

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(
      Offset.zero & size,
      Paint()..color = const Color(0xffeef1f8),
    );
    final thresholdY = size.height * (1 - threshold);
    canvas.drawLine(
      Offset(0, thresholdY),
      Offset(size.width, thresholdY),
      Paint()
        ..color = Colors.orange
        ..strokeWidth = 1,
    );
    if (values.isEmpty) return;
    final path = Path();
    for (var i = 0; i < values.length; i++) {
      final x = values.length == 1 ? 0.0 : size.width * i / (values.length - 1);
      final y = size.height * (1 - values[i].clamp(0.0, 1.0));
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }
    canvas.drawPath(
      path,
      Paint()
        ..color = color
        ..strokeWidth = 2
        ..style = PaintingStyle.stroke,
    );
  }

  @override
  bool shouldRepaint(_ProbabilityPainter oldDelegate) =>
      oldDelegate.values != values ||
      oldDelegate.threshold != threshold ||
      oldDelegate.color != color;
}
