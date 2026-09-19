# Görev promptu: çalışmanın başka bir yapay zekâ aracında yeniden kurulması

Bu belge, çalışmayı hiçbir ön bilgisi olmayan bir araca aktarmak için hazırlanmıştır.
Nihai makalenin başlığı: *Çok görünümlü domates görüntülerinde derin öğrenme tabanlı büyüme
evresi sınıflandırması: kamera görüşü, geç füzyon ve hesaplama maliyeti*.

## Rolün
Bilgisayarlı görü, derin öğrenme ve deneysel tasarım konusunda uzman bir araştırmacı gibi çalış. İddiaları kanıta dayandır; doğrulayamadığın bilgiyi "doğrulanmadı" diye işaretle; kaynak ve sayı uydurma.

## Bağlam
Türkiye'de bilgisayar teknolojileri alanında çalışan bir araştırmacıyım. YOLO ile ilk uygulamalı çalışmamı yapıp **IJEDT** (https://dergipark.org.tr/tr/pub/ijedt) dergisine Türkçe makale göndereceğim. Çalışma sade olmalı ama uygulamaya dönük gerçek bir soruya cevap vermeli. Ortam: **Google Colab Pro+**, Drive klasörü `YOLODomatesEylul2026`.

**Önemli:** Çalışmanın konusu veri sızıntısı DEĞİLDİR. Bitki düzeyinde bölme yalnızca yöntemsel güvencedir ve makalede kısaca gerekçelendirilir; bölme stratejileri karşılaştırılmaz.

## Veri seti
**TomatoMAP** — Zhang, Struckmeyer, Kolb, Reichardt (2026). Scientific Data 13:309. https://doi.org/10.1038/s41597-026-06926-9 · Veri: https://doi.org/10.5447/ipk/2025/14 (CC BY 4.0) · Kod: https://github.com/0YJ/TomatoMAP · HF aynası: `Voxel51/tomato-map` (FiftyOne; `samples.json` + `data/`).

- 101 domates bitkisi, tek sera kabini, 163 gün, düzensiz aralıklı çekim günleri.
- 4 sabit kamera × döner tablada 12 poz → bitki-gün oturumu başına 48 görüntü; toplam 64.464 görüntü (1080×1440).
- Dosya adı: `pi{kamera}_{sıra}_{bitki}_{poz}_{YYYYMMDDhhmmss}.jpg`; HF örneklerinde `plant_id`, `piid`, `pose_id`, `bbch_stage`, `capture_datetime` alanları; rig görüntüleri `"det"` etiketli.
- Etiket oturum düzeyinde bir BBCH kodu (50 farklı kod, çok dengesiz).
- Kaynak, kamera modüllerini 45°, 90°, 135° ve 180° düşey eğim açılarıyla tanımlar; veri dosyalarındaki kamera kimlikleriyle eşleşme doğrulanamadığı için kameralar K1–K4 olarak anılır ve sonuçlar açı değerleriyle ilişkilendirilmez.

## Araştırma soruları
1. Hangi kamera görüşü ana büyüme evrelerini sınıflandırmada daha başarılı?
2. Kamera görüşünün etkisi YOLO11n-cls ve YOLO26n-cls ağırlıklarında benzer mi?
3. Dört kameranın tahminlerinin birleştirilmesi en iyi tek kameradan daha iyi mi?
4. Füzyonun kazancı, dört kat çıkarım maliyetine değer mi?

## Deney tasarımı
**Sınıflar:** BBCH kodunun ilk hanesine göre 6 sıralı sınıf: 1 yaprak gelişimi, 2 yan sürgün oluşumu, 5 çiçek salkımı çıkışı, 6 çiçeklenme, 7 meyve gelişimi, 8 olgunlaşma. Eşleme tablosu makalede verilir.

**Alt küme:** Her oturumda dört kamerada da bulunan pozlar arasından, sabit tohumla deterministik (ör. `sha256(tohum:oturum)`) tek poz seçilir; o pozun 4 kamera görüntüsü alınır (≈1.343 oturum × 4 ≈ 5.372 görüntü; gerçek sayı metadata'dan hesaplanır). Seçim manifesti kaydedilir.

**Değerlendirme:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)`; grup = `plant_id`, hedef = 6 ana evre (oturum düzeyinde). Her katta 6 sınıfın bulunduğu denetlenir; sağlanmazsa `random_state` deterministik olarak artırılır ve kaydedilir. Kat manifesti **bir kez** oluşturulur ve tüm kamera/modellerde aynen kullanılır. Dış kat k: test = k, doğrulama = (k+1) mod 5, eğitim = kalan 3 kat.

**Eğitim:** 2 model × 4 kamera × 5 kat = **40 eğitim**. Tek sabit eğitim tohumu (çapraz doğrulamanın eğitim rassallığını ölçmediği makalede belirtilir). Görüntüler siyah dolguyla kareye tamamlanıp 384 px kaydedilir; `imgsz=320`, `epochs=50`, `patience=15`, `batch=64`; diğer hiperparametreler Ultralytics varsayılanı, tüm koşullarda aynı. `best.pt` doğrulama katına göre seçilir.

**Tahmin ve füzyon:** Test katı tahminleri birleştirilerek her kamera için tüm oturumları kapsayan kat-dışı (OOF) tahmin elde edilir. Geç füzyon: aynı oturumun dört kamera olasılıklarının ortalaması, argmax.

**Metrikler:** Makro F1 (ana), dengeli doğruluk, doğruluk, Cohen κ, sınıf bazında P/R/F1, normalize karışıklık matrisi, ordinal hata (1–6 evre indeksi farkı; BBCH kod farkı KULLANILMAZ), komşu sınıfa düşen hata oranı, kat bazında makro F1.

**İstatistik:** Bitki kümeli, eşleştirilmiş bootstrap (B = 2000): bitkiler yerine koymalı örneklenir, seçilen bitkilerin tüm oturumları alınır, aynı örnekler tüm sistemlere uygulanır. Farkların %95 GA'sı ve iki yönlü bootstrap p değeri (ters yönlü örnek yoksa değer 1/B tabanına sabitlenir; bu durumda makalede "p < 0,001" yazılır, Holm düzeltmeli değer de "≤" ile verilir). Holm düzeltmesi aileler içinde: (a) model başına 6 kamera çifti, (b) model başına füzyon − 4 tek kamera, (c) görünüm başına YOLO26 − YOLO11.

**Maliyet:** En iyi tek kamera (1 görüntü / 1 model çağrısı) vs dört kameralı füzyon (4 görüntü / 4 model çağrısı): farklı oturumlardan oluşan deterministik bir örneklem üzerinde, görüntüler zamanlamadan önce belleğe yüklenerek (disk okuma ve JPEG çözme hariç) oturum başına gecikme (medyan, p90; GPU, CPU, ONNX-CPU), oturum/saniye, GFLOPs (×1 vs ×4), toplam parametre, toplam ağırlık boyutu ve makro F1 kazancı. Kamera bazında ayrı gecikme verilmez.

## Pipeline gereksinimleri
- Tek `Config` sınıfı; Python modülü + Türkçe açıklamalı Colab not defteri.
- Aşamalar: indeks + veri denetimi → poz seçimi → görüntüler → katlar → eğitim/test → OOF + füzyon → maliyet → analiz.
- İndirme: önce yalnızca seçilen dosyalar (`snapshot_download(allow_patterns=...)`, yeniden deneme, `HF_TOKEN`), başarısızsa tam indirme ve alt küme.
- Dayanıklılık: işlenmiş görüntüler tar parçaları olarak Drive'da (`.done` işaretli); Ultralytics `project` Drive'da (`last.pt` her epokta), yarım eğitim `resume=True`; `TRAIN_DONE`/`DONE` işaretleri; atomik CSV yazımı; loglar ve ortam bilgisi Drive'da.
- Çıktılar: CSV + `ALL_TABLES.xlsx`; 300 dpi PNG + PDF Türkçe şekiller; `SUMMARY_FOR_MANUSCRIPT.json`.
- Tablolar: veri denetimi, sınıf eşleme ve alt küme; kat dağılımı; ana sonuçlar (GA'lı); kat bazında sonuçlar; sınıf bazında P/R/F1; kamera çifti karşılaştırmaları; füzyon ve model karşılaştırmaları; maliyet ve maliyet–başarı dengesi.
- Şekiller: sınıf dağılımı; evre örnekleri; bir oturumun dört görünümü; görünüm × model makro F1 (GA); karışıklık matrisleri; sınıf × görünüm F1 ısı haritası; kamera çifti fark orman grafiği; maliyet–başarı; eğitim eğrileri.
- Kod mümkünse sentetik veriyle test edilir; test edilemeyen kısımlar açıkça belirtilir.

## İş akışı
1. Veri seti bilgilerini ve kamera–açı eşleşmesini kaynaktan doğrula; kamera açısı veya çok açılı füzyonla büyüme evresi sınıflandırması üzerine yakın çalışmaları tara; kaynaklarıyla kısaca özetle.
2. Pipeline'ı yaz (veya verilen `tomato_pipeline.py`'yi gözden geçirip hataları bildir).
3. Sonuçlar verildiğinde makaleyi IJEDT formatında Türkçe yaz (Özet/Abstract, Giriş, Materyal ve Yöntem, Bulgular, Tartışma, Sonuç ve Sınırlılıklar, Kaynaklar). Yalnızca sonuç dosyalarındaki sayıları kullan. "Daha yüksek makro F1" ile "Holm düzeltmeli istatistiksel üstünlük" ifadelerini ayır.

**Sınırlılıklar:** tek çeşit, tek sera kabini, 101 bitki, oturum başına tek poz, tek eğitim tohumu; evreler zamanda örtüşebildiği için tek etiketli sınıflandırma yapısal belirsizlik taşır.


## Not

İki YOLO ağırlık seti, sınıflandırma görevinde aynı parametreli mimariyi paylaşır
(236 ağırlık tensörü ad ve boyut olarak eşleşir, 2.812.104 parametre). Bu nedenle model
karşılaştırması ana katkı değil, bulguların farklı başlangıç ağırlıkları altında tutarlılığını
sınayan ikincil bir analizdir.
