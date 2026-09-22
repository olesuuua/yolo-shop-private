# Browser video replay

Run the existing server on CPU, for example from the project directory:

```sh
LIGHTSTORE_DEVICE=cpu .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

This starts the configured real detector/OCR service. Replay uses the same Jev integration as camera mode; normal configured identification calls still apply. No model, threshold, acceptance rule, or GPU configuration was changed. The replay UI refuses playback if the server reports a non-CPU device.

1. Open `http://localhost:8000/`.
2. Select **Video file**, choose a local recording, and click **Play**. Play first
   validates that the browser can genuinely decode the video: detection uploads
   begin only after one decoded frame with nonzero dimensions is verified. A
   video that advances its timeline or plays audio without decoding frames
   (for example HEVC in a browser without that codec) never starts detection
   and shows an actionable error; no black fallback frames are sent.
3. **Pause** stops new detection submissions. Click **Play** to resume.
4. **Restart** resets the backend session and returns to time zero, then plays. Export evidence before restarting: reset clears it.
5. At the end, results remain visible; background OCR/Jev updates continue. **Stop** disconnects and releases the file. Choose it again to replay after Stop.
6. Select **Camera** and click **Start camera** to restore camera capture. Changing input releases the previous source and resets the session.

Decode validation uses `HTMLVideoElement.requestVideoFrameCallback` when
supported: it fires only when a video frame is actually presented, so loaded
metadata, timeline progress or audio playback are not treated as proof.
Without that API (documented fallback), validation polls until
`readyState` reaches `HAVE_CURRENT_DATA` with nonzero `videoWidth`/`videoHeight`.
Validation is bounded by a 10-second startup timeout and by media `error`
events; on failure replay stops (pause, no further detection uploads) with the
message *This browser could not decode the video; try a compatible H.264 MP4*
— the browser does not claim the codec is definitely at fault. Validation
callbacks and timers are cleared on Stop, source/file replacement, restart and
errors, so a late callback cannot start an obsolete replay.

The file stays local. Only 640×480 detection JPEGs and requested crop JPEGs leave the browser. The browser captures one native-resolution canvas, stretches that canvas into the detection JPEG, and extracts OCR crops from that same canvas. Boxes use `detect_640x480` coordinates with the existing 8% margin and independent horizontal/vertical scaling.

Playback follows the video timeline at normal speed, with one detection capture/upload in flight. When inference finishes, the browser samples the current available frame and skips intervening frames. This reproduces live sampling, **not deterministic frame-by-frame inference**. There are no speed or seek controls. Restart is the only timeline navigation. Browser scheduling, CPU load, background-tab throttling, and diagnostic encoding affect which frames are sampled.

## Evidence

Open `http://localhost:8000/?diag=1` before replay to opt in. The bottom panel lets you select an OCR request and inspect its matching original frame with the requested box, exact uploaded JPEG crop, IDs, capture timestamp, mapped rectangle, acceptance/rejection, OCR text/scores, and Jev outcome. The yellow box is the requested detection box; `mapped_rect` includes the crop margin.

Click **Export evidence JSON**. `replay-evidence.json` embeds original PNG and crop JPEG data URLs alongside frame/track logs and request records. Decode the portion after the comma with base64 to recover each image; the crop bytes are exactly those sent to OCR. No credentials or authorization headers are collected. Original frames can include the surroundings visible in the recording.

Jev attribution lists `source_request_ids` for the accumulated text actually sent, including earlier crops. `jev.lines` records that input. Missing source images can result from eviction; source IDs remain useful for joining records. A missing Jev entry means no associated completed call has been collected yet, not a negative match. Export again after pending work finishes if needed. Status is polled once per second even while paused/ended; annotated pixels are from the last detection, while the identification sidebar continues updating.

Bounds: latest 300 sampled-frame records, 60 crop records, 24 MiB of image strings in the browser, and 200 server request records. Large original images (raw dimensions above 12 MiB) are omitted with an explicit reason. Collection shows eviction/drop counts. OCR diagnostic output is capped at 100 lines of 1,000 characters each (recognition itself is unchanged). Per-text provenance retains 200 source IDs and reports `source_ids_dropped` if exceeded. Each crop retains its latest associated Jev outcome; `jev_outcomes_replaced` reports earlier outcomes replaced. Images can be evicted while processing continues. Reset/source/file changes clear collection; Stop preserves collected evidence for export. JSON export temporarily allocates another serialization of the bounded collection.

Timing units are milliseconds. Detection RTT uses the browser's monotonic clock. Crop `request_to_receipt_ms`, OCR `queue_wait_ms`/`ocr_ms`, and Jev `duration_ms` use only the server monotonic clock. Video timestamps are media time in seconds; they are never subtracted from server timestamps.

## Formats and limitations

The browser's installed codecs determine support; no transcoding or whole-file upload is provided. MP4 with H.264 and WebM with VP8/VP9 are common choices, but container extension alone does not guarantee support. Playback only starts after the decoded-frame validation above succeeds; unsupported, corrupt, or unreadable recordings stop replay with an actionable error and send zero detection uploads. Large-resolution files cost more memory and may exceed browser canvas limits. Keep the tab foregrounded for representative timing. Only one browser source/client can own the server pipeline.

Offline verification uses synthetic canvases/JPEGs and fake detector, OCR and Jev implementations; it does not access a camera, load model weights, or call Jev.
