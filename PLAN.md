# Local packing prototype plan

## Goal

Recognize products as they enter a bag, compare packed quantities with an order, and support Russian packaging with English brand names. Develop locally in `/home/olesya/Projects/yolo-shop`.

## Current position

- GPU detection and the camera website run locally. The website still uses the six-class detector and a fixed rectangular bag area.
- Five product references are prepared. Offline OCR and Jev tests selected the expected product on 15 of 17 reference photos; two lacked enough evidence. This is an initial check, not independent accuracy validation.
- The current detector missed four of five front photos and labeled the muesli as crisps. These were resized still images, so camera tests are also needed.
- A separate sample-image benchmark measured about 23 ms per processed frame, including 12 ms GPU inference. Actual camera FPS and the source of the slowdown remain unverified.
- OCR and Jev are not yet connected to the live camera.

## 1. Measure and improve camera FPS

- Measure camera capture rate, detection throughput, request latency, and preview FPS separately.
- Show the live camera continuously and draw detection overlays independently of server image responses.
- Process the newest available frame with a bounded queue. Drop stale frames instead of building a delay.
- Preserve packing events and coordinate alignment when changing the display pipeline.

Done when the preview stays responsive during detection and the page shows measured performance. Set a realistic detection target after measuring the actual camera pipeline.

## 2. Choose a detector for product tracking

- Compare broader product detectors on the five supplied products and new camera footage.
- Check bottles, cartons, cans, jars, fruit, and boxes as the test assortment grows.
- Choose by missed products, box quality, tracking stability, and GPU latency. Exact flavor recognition belongs in the product identification stage.
- Keep the existing detector as a baseline. Decide whether additional training is necessary from these results.

Done when the selected detector reliably finds the initial products across useful angles and motion.

## 3. Connect OCR to tracked products

- Keep higher-resolution camera frames for OCR while using smaller frames for detection.
- Select sharp product crops and run Russian/English OCR asynchronously.
- Accumulate useful text across views of the same tracked item. Retry when better text appears, rather than on every frame.
- Test the camera at an angle to expose side labels while keeping the bag visible.

Done when text follows the correct item and OCR does not stall the preview.

## 4. Connect Jev identification

- Use the local catalog built from store descriptions and verified packaging labels.
- Send accumulated OCR text, a soft packaging-class hint, and relevant candidate descriptions to Jev. Keep the API key on the server.
- Treat Jev state as request context; do not assume the service retains the catalog between requests. Cache catalog preparation and results locally.
- Preserve unknown and insufficient-evidence outcomes. A high score alone does not verify an exact SKU.
- Test real near-matches such as different flavors, sizes, and fat percentages, plus products outside the catalog.

Done when independent camera examples support useful acceptance thresholds and uncertain items remain unresolved.

## 5. Compare packing with orders

- Add local orders containing product IDs and required quantities.
- Record confirmed bag-entry events and show correct, missing, extra, and unresolved items.
- Handle repeated detections, temporary occlusion, item removal, and track-ID changes without silently corrupting quantities.
- Keep the fixed bag area for the first end-to-end test.

Done when a complete test order can be packed, corrected, and checked accurately.

## 6. Track the bag area dynamically

- Select and evaluate a method for locating the bag opening as its position and shape change.
- Replace the fixed rectangle with the tracked area. Pause counting when its position is uncertain.
- Test bag movement, hands covering the opening, and items passing near the bag without entering it.

Done when bag movement preserves reliable entry and removal decisions.

## MVP validation

Use new recordings that were not used to write the catalog. Measure product identification errors, unknown outcomes, missed and duplicate packing events, time to identification, and camera responsiveness. Include similar products and poor lighting. Agree on acceptable error rates before calling the prototype ready for a store pilot.

Immediate next task: measure the live camera pipeline and fix preview responsiveness, then compare detectors.
