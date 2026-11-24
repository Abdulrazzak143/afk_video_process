import 'dart:io';

import 'package:ffmpeg_kit_flutter_new_audio/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_new_audio/return_code.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

class AudioExtractor {
  /// Extracts mono 16 kHz PCM WAV audio from [videoPath] using FFmpeg.
  ///
  /// Returns the absolute path to the generated `.wav` file.
  Future<String> extractAudio(String videoPath) async {
    final tempDir = await getTemporaryDirectory();
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final outputPath = p.join(tempDir.path, 'extracted_$timestamp.wav');

    final outputFile = File(outputPath);
    if (await outputFile.exists()) {
      await outputFile.delete();
    }

    final command =
        '-i "$videoPath" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$outputPath"';

    try {
      final session = await FFmpegKit.execute(command);
      final returnCode = await session.getReturnCode();

      if (!ReturnCode.isSuccess(returnCode)) {
        final logs = await session.getAllLogsAsString();
        throw Exception(
          'FFmpeg failed (code: $returnCode).\n$logs',
        );
      }

      if (!await outputFile.exists()) {
        throw Exception('FFmpeg finished but output WAV was not created.');
      }

      return outputPath;
    } catch (e) {
      if (e is Exception) {
        rethrow;
      }
      throw Exception('FFmpeg execution error: $e');
    }
  }
}
