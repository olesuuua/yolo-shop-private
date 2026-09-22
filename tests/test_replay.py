"""Replay lifecycle and attribution without cameras, model weights or external calls."""
import threading
import time
import unittest
from identification import IdentificationService, IdentConfig
from test_crop_protocol import processor_with, jpeg_bytes


class ReplayTests(unittest.TestCase):
    def test_reset_clears_model_tracking_and_timings(self):
        proc, model, service = processor_with([])
        class Tracker:
            count = 0
            def reset(self): self.count += 1
        model.tracker = Tracker()
        proc.process_detect(jpeg_bytes())
        old = service.touch(7, 'bottle')
        proc.reset()
        self.assertEqual(model.tracker.count, 1)
        self.assertFalse(proc.frame_times)
        self.assertFalse(service.submit(7, 'bottle', b'jpeg', expected=old))
        self.assertFalse(service.tracks)

    def test_queued_crop_carries_original_evidence_and_replacement_status(self):
        service = IdentificationService(config=IdentConfig(submit_interval_empty_s=0))
        first = {'request_id': 'first'}
        service.submit(7, 'bottle', b'one', diagnostic=first)
        old = service.tracks[7]
        service.submit(7, 'bottle', b'two', diagnostic={'request_id':'second'})
        self.assertEqual(first['ocr_status'], 'replaced')
        track, jpeg, (expected, queued, diagnostic, frame_id) = service._take_slot(with_meta=True)
        service.reset()
        service.touch(7, 'new bottle')
        self.assertIs(expected, old)
        self.assertIsNone(service._merge(track, 'bottle', [{'text':'OLD'}], expected=expected))

    def test_jev_attributes_all_input_crops_and_rejects_late_completion(self):
        entered, release = threading.Event(), threading.Event()
        def jev(*args):
            entered.set(); release.wait(2)
            return {'status':'unknown', 'answer':{}}
        service = IdentificationService(jev_fn=jev)
        service.diagnostics = {'a':{'request_id':'a'}, 'b':{'request_id':'b'}}
        evidence = service._merge(7, 'bottle', [{'text':'WATER', 'score':.9}], request_id='a')
        service._merge(7, 'bottle', [{'text':'LABEL', 'score':.8}], request_id='b')
        thread = threading.Thread(target=service._identify,args=(7,evidence)); thread.start()
        self.assertTrue(entered.wait(1)); release.set(); thread.join(2)
        for request in service.replay_debug()['requests']:
            self.assertEqual(request['jev']['source_request_ids'], ['a','b'])
            self.assertGreaterEqual(request['jev']['duration_ms'], 0)
        entered.clear(); release.clear()
        thread = threading.Thread(target=service._identify,args=(7,evidence)); thread.start()
        self.assertTrue(entered.wait(1)); service.reset()
        new = service.touch(7,'new'); service.jev_inflight.add(7)
        release.set(); thread.join(2)
        self.assertEqual(new.result, {'status':'needs_more_evidence'})
        self.assertIn(7, service.jev_inflight)
        self.assertFalse(service.replay_debug()['requests'])

    def test_diagnostics_are_opt_in_and_bounded(self):
        service = IdentificationService(config=IdentConfig(submit_interval_empty_s=0))
        service.submit(7,'bottle',b'jpeg')
        self.assertFalse(service.replay_debug()['requests'])
        for n in range(205):
            service.submit(7,'bottle',b'jpeg',diagnostic={'request_id':str(n)})
        debug = service.replay_debug()
        self.assertEqual(len(debug['requests']),200)
        self.assertEqual(debug['dropped'],5)

    def test_ocr_timing_and_text_follow_request(self):
        class OCR:
            def predict(self, jpeg): return [{'text':'WATER', 'score':.9}]
            def close(self): pass
        service = IdentificationService(ocr_factory=OCR, jev_fn=lambda *a: {'status':'unknown'},
                                        config=IdentConfig(dedup=False))
        service.start()
        try:
            service.submit(7,'bottle',b'jpeg',diagnostic={'request_id':'crop'})
            deadline = time.monotonic()+2
            while time.monotonic()<deadline:
                rows = service.replay_debug()['requests']
                if rows and rows[0].get('ocr_status') == 'done': break
                time.sleep(.01)
            row = service.replay_debug()['requests'][0]
            self.assertEqual(row['lines'][0]['text'],'WATER')
            self.assertGreaterEqual(row['queue_wait_ms'],0)
            self.assertGreaterEqual(row['ocr_ms'],0)
        finally: service.close()
