# Pipeline aşamaları

`src/tomato_pipeline.py` sekiz aşamadan oluşur. Her aşama bağımsız çalışır, tamamlanan işleri
atlar ve kesinti sonrası kaldığı yerden devam eder.

| # | Fonksiyon | Yaptığı iş | Kalıcı çıktı |
|---|---|---|---|
| 1 | `stage_index` | `samples.json` indirilir, `det` örnekleri ayrıştırılır, veri denetimi yapılır | `cache/metadata_full.csv`, `T0_metadata_audit.json` |
| 2 | `stage_select` | Oturum başına deterministik poz seçimi | `manifests/selection_manifest.csv`, `subset_images.csv` |
| 3 | `stage_images` | Seçmeli indirme, kareye tamamlama, tar parçaları | `cache/shards/shard_*.tar` + `.done` |
| 4 | `stage_folds` | Bitki düzeyinde beş kat, sınıf denetimi | `manifests/plant_folds.csv`, `T2_folds.csv` |
| 5 | `stage_train_eval` | 40 eğitim (resume destekli) + test katı tahminleri | `runs/<koşu>/weights`, `test_predictions.csv` |
| 6 | `stage_oof` | Kat-dışı tahminlerin birleştirilmesi ve geç füzyon | `results/oof/*.csv` |
| 7 | `stage_efficiency` | Parametre, GFLOPs, GPU/CPU/ONNX gecikmesi | `T7_cost.csv` |
| 8 | `stage_analysis` | Tablolar, şekiller, istatistiksel karşılaştırmalar | `results/tables`, `results/figures`, `SUMMARY_FOR_MANUSCRIPT.json` |

## Dayanıklılık

- İşlenmiş görüntüler Drive'da tar parçaları olarak saklanır; yeni oturumda yalnızca açılır.
- Ultralytics `project` klasörü Drive'dadır, `last.pt` her epokta yazılır; yarım kalan eğitim
  `resume=True` ile sürer.
- `TRAIN_DONE` ve `DONE` işaret dosyaları tamamlanan işleri belirtir.
- CSV yazımı atomiktir (önce geçici dosya, sonra yeniden adlandırma).
- Ortam bilgisi ve günlük `logs/` altına yazılır.

## Analizi tek başına çalıştırma

Depodaki `results/oof/` dosyaları eğitim çıktılarını içerdiği için `stage_analysis` GPU olmadan
çalıştırılabilir; bütün tablolar ve şekiller bu dosyalardan yeniden üretilir.
