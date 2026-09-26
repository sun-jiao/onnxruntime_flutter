import 'ort_provider.dart';
import 'providers/ort_flags.dart';

enum OrtProviderFallback { error, skipUnavailable, cpu }
enum QnnPerformanceMode {
  balanced('balanced'), burst('burst'), highPerformance('high_performance'),
  powerSaver('power_saver');
  final String value;
  const QnnPerformanceMode(this.value);
}

/// Registration configuration, not a claim about actual operator placement.
class OrtProviderConfig {
  final OrtProvider provider;
  final int flags;
  final Map<String, String> options;
  OrtProviderConfig._(this.provider, this.flags, Map<String, String> options)
      : options = Map.unmodifiable(options) {
    for (final entry in options.entries) {
      if (entry.key.isEmpty || entry.key.contains('\u0000') || entry.value.contains('\u0000')) {
        throw ArgumentError('Provider option names/values must not contain NUL.');
      }
    }
  }
  factory OrtProviderConfig.cpu({CPUFlags flags = CPUFlags.useArena}) =>
      OrtProviderConfig._(OrtProvider.cpu, flags.value, {});
  factory OrtProviderConfig.coreml({Set<CoreMLFlags> flags = const {}}) =>
      OrtProviderConfig._(OrtProvider.coreml, flags.fold(0, (a, b) => a | b.value), {});
  factory OrtProviderConfig.nnapi({Set<NnapiFlags> flags = const {}}) =>
      OrtProviderConfig._(OrtProvider.nnapi, flags.fold(0, (a, b) => a | b.value), {});
  factory OrtProviderConfig.qnn({String? backendPath, QnnPerformanceMode? performanceMode,
      Map<String, String> extraOptions = const {}}) => OrtProviderConfig._(OrtProvider.qnn, 0, {
    ...extraOptions,
    if (backendPath != null) 'backend_path': backendPath,
    if (performanceMode != null) 'htp_performance_mode': performanceMode.value,
  });
  factory OrtProviderConfig.xnnpack({int? intraOpNumThreads}) {
    if (intraOpNumThreads != null && intraOpNumThreads < 0) {
      throw ArgumentError.value(intraOpNumThreads, 'intraOpNumThreads');
    }
    return OrtProviderConfig._(OrtProvider.xnnpack, 0, {
      if (intraOpNumThreads != null) 'intra_op_num_threads': '$intraOpNumThreads',
    });
  }
}

class OrtProviderReport {
  final List<OrtProvider> registered;
  final Map<OrtProvider, String> skipped;
  OrtProviderReport(List<OrtProvider> registered, Map<OrtProvider, String> skipped)
      : registered = List.unmodifiable(registered), skipped = Map.unmodifiable(skipped);
}

class OrtRuntimeCapabilities {
  final String runtimeVersion;
  final String backend;
  /// Raw runtime provider names; unknown names are never mapped to CPU.
  final List<String> providerNames;
  final bool profilingFile;
  OrtRuntimeCapabilities(this.runtimeVersion, this.backend,
      List<String> providerNames, {required this.profilingFile})
      : providerNames = List.unmodifiable(providerNames);
}

// Shared registration policy. Native callers run this against cloned options.
OrtProviderReport registerOrtProviders(List<OrtProviderConfig> configs,
    OrtProviderFallback fallback, List<String> available,
    bool Function(OrtProviderConfig) register) {
  if (configs.isEmpty) throw ArgumentError('At least one provider is required.');
  if (configs.map((c) => c.provider).toSet().length != configs.length) {
    throw ArgumentError('Duplicate providers are not allowed.');
  }
  final registered = <OrtProvider>[];
  final skipped = <OrtProvider, String>{};
  for (final config in configs) {
    try {
      if (!available.contains(config.provider.value) || !register(config)) {
        throw UnsupportedError('${config.provider.value} is unavailable.');
      }
      registered.add(config.provider);
    } catch (error) {
      if (fallback == OrtProviderFallback.error) rethrow;
      skipped[config.provider] = error.toString();
    }
  }
  if (fallback == OrtProviderFallback.cpu && !registered.contains(OrtProvider.cpu)) {
    if (!register(OrtProviderConfig.cpu())) throw StateError('CPU registration failed.');
    registered.add(OrtProvider.cpu);
  }
  if (registered.isEmpty) throw StateError('No requested providers could be registered.');
  return OrtProviderReport(registered, skipped);
}
