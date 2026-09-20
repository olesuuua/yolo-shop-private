# Product reference and first recognition check

The local catalog is `data/local/catalog.json`. Product IDs are demo IDs, not store SKU codes. Original photos and store descriptions are preserved under `data/local/products/`.

| Product | Verified variant and size | Packaging |
| --- | --- | --- |
| Домик в деревне | Пастеризованное молоко, 2,5%, 930 мл | Plastic bottle |
| 4Life | Crunchy Muesli Strawberry, 350 г | Cardboard box |
| Ясно Солнышко | Овсяные хлопья Экстра №2, 500 г | Cardboard box |
| Barilla | Penne Rigate №73, 450 г | Cardboard box |
| Святой Источник / Saint Spring | Негазированная / Still, 0,5 л | Plastic bottle |

Packaging photos supplied sizes omitted from most store descriptions. The catalog separates visually verified label wording from uncorrected OCR. It records misleading serving suggestions and advertisements: milk/yogurt on muesli, pesto on pasta, other numbered oat varieties on the oats box. Dates and batch numbers are not SKU identifiers.

## OCR and Jev

Russian/English GPU OCR ran on all 17 photos. The text and curated product references were sent to TypeSafe with user approval. Images, filenames, folder names and expected answers were not sent. The API key was read from `.env` without logging it.

Jev model: `jev-1.13.0`. 20 requests completed. Median end-to-end request time: 0.822 seconds in this run.

- 15/17 photo inputs selected the correct catalog candidate.
- Two side/back milk views returned insufficient evidence.
- Generic label text returned insufficient evidence.
- Coca-Cola Zero, absent from the catalog, returned other product.
- Barilla Spaghetti №5 500 g returned other product rather than the catalog's Penne №73 450 g.

These are reference-photo smoke tests, not independent accuracy measurements. The same packaging photos helped construct the reference catalog. The five real products contain no competing flavors of one brand. Some views expose only the brand, so selecting a candidate does not verify its size or exact variant. Jev confidence is not a measured probability of an error-free order. All results retain `exact_sku_verified: false`.

Raw evidence: `work/product-reference/ocr.json`, `jev-results.json`, and `request-preview.json`. `jev_catalog.py` implements request building, response validation, and unknown/insufficient-evidence outcomes. Its CLI prepares a local payload by default; `--send-to-jev` explicitly invokes the API.

## Existing detector

The released six-class model was tested locally on the five front photos at the current 416 input, 640×480 processing frame, and 0.50 new-track confidence threshold. It returned no qualifying boxes for milk, oats, pasta or water. It labeled muesli as pack of crisps with confidence 0.922. Evidence: `work/product-reference/detector-results.json`.

This result shows why the current detector cannot be relied upon for these products. It does not measure detector performance on future camera footage; the portrait photos were processed using the current app's resize path.

## Next integration work

Compare a generic product detector or broader packaging detector on these images and fresh angled-camera footage. Keep high-resolution images for OCR, use detector boxes only for cropping and tracking, accumulate text by track ID, and keep SKU recognition asynchronous. The current running camera app has not yet been connected to OCR/Jev; it still runs the original six-class packing pipeline.
