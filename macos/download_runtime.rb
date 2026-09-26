# Local/path pods do not run prepare_command. Prepare the vendored dylib while
# evaluating the podspec, before CocoaPods enumerates and embeds its files.
require 'json'
require 'digest'
require 'fileutils'

module OnnxruntimeDownload
  def self.prepare(plugin_dir, manifest_path)
    manifest = JSON.parse(File.read(manifest_path))
    artifact = manifest.fetch('archives').find { |a| a.fetch('target') == 'macos-universal2' }
    cache = File.expand_path(ENV.fetch('ORT_CACHE_DIR', '~/.cache/onnxruntime_flutter'))
    entry = File.join(cache, artifact.fetch('sha256'))
    FileUtils.mkdir_p(entry)
    File.open(File.join(entry, 'macos.lock'), 'w') do |lock|
      lock.flock(File::LOCK_EX)
      files = artifact.fetch('files')
      staging = File.join(entry, 'macos-extracted')
      valid = files.all? do |destination, member|
        path = File.join(staging, member)
        digest = manifest.fetch('libraries')[destination]
        File.file?(path) && (!digest || Digest::SHA256.file(path).hexdigest == digest)
      end
      unless valid
        archive = File.join(entry, 'archive')
        unless File.file?(archive)
          part = "#{archive}.macos-part"
          begin
            ok = system('curl', '--fail', '--location', '--retry', '3', '--connect-timeout', '30',
                        '--max-time', '600', '--silent', '--show-error',
                        artifact.fetch('url'), '--output', part)
            raise 'ONNX Runtime download failed; check network/proxy or prefill ORT_CACHE_DIR' unless ok
            raise 'ONNX Runtime archive checksum mismatch' unless Digest::SHA256.file(part).hexdigest == artifact.fetch('sha256')
            File.rename(part, archive)
          ensure
            FileUtils.rm_f(part)
          end
        end
        raise "ONNX Runtime cached archive checksum mismatch: #{archive}; remove it and retry" unless Digest::SHA256.file(archive).hexdigest == artifact.fetch('sha256')
        FileUtils.rm_rf(staging)
        FileUtils.mkdir_p(staging)
        raise 'Could not extract ONNX Runtime archive' unless system('tar', '-xzf', archive, '-C', staging)
      end
      # Verify before every copy, including a freshly extracted archive.
      files.each do |destination, member|
        source = File.join(staging, member)
        digest = manifest.fetch('libraries')[destination]
        raise "Missing ONNX Runtime artifact: #{member}" unless File.file?(source)
        raise "ONNX Runtime library checksum mismatch: #{member}" if digest && Digest::SHA256.file(source).hexdigest != digest
      end
      generated = File.join(plugin_dir, '.onnxruntime')
      FileUtils.mkdir_p(generated)
      files.each_value do |member|
        source = File.join(staging, member)
        destination = File.join(generated, File.basename(member))
        next if File.file?(destination) && Digest::SHA256.file(destination).hexdigest == Digest::SHA256.file(source).hexdigest
        FileUtils.cp(source, destination)
      end
    end
  end
end
