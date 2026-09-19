# Çalışma tasarımı

Bu belge, depodaki kodun hangi deney tasarımını uyguladığını özetler. Makalenin yöntem
bölümüyle birebir uyumludur.

## Araştırma soruları

1. Hangi kamera görüşü ana büyüme evresi sınıflandırmasında daha başarılıdır?
2. Kamera görüşünün etkisi iki ön eğitimli ağırlık setinde benzer midir?
3. Dört kameranın tahminlerinin birleştirilmesi en iyi tek kamerayı geçer mi?
4. Elde edilen kazanç, dört kat çıkarım maliyetini karşılar mı?

## Sınıf tanımı

BBCH kodunun ilk hanesine göre altı sıralı sınıf kullanılır. Kodlar veri setinde fiilen
bulunanlardır; aralıklardaki bazı ara kodlar veri setinde yer almaz.

| Sınıf | Evre | BBCH kodları | Oturum |
|---|---|---|---|
| S1 | Yaprak gelişimi | 13, 14, 15, 16, 17, 19 | 196 |
| S2 | Yan sürgün oluşumu | 20, 21, 22, 23, 27, 28, 29 | 29 |
| S5 | Çiçek salkımı çıkışı | 51, 52, 53, 54, 55, 56, 59 | 97 |
| S6 | Çiçeklenme | 60–69 | 220 |
| S7 | Meyve gelişimi | 70–79 | 611 |
| S8 | Olgunlaşma | 80–89 | 190 |

## Alt küme ve poz seçimi

Her bitki-gün oturumunda dört kamerada da bulunan pozlar belirlenir; aralarından
`sha256(pose_seed:oturum)` üzerinden biri seçilir ve o pozun dört kamera görüntüsü alınır.
Seçim oturumlar arası bağımsızdır ve `manifests/selection_manifest.csv` dosyasına yazılır.
Sonuç: 1.343 oturum, 5.372 görüntü.

Görüntüler siyah dolguyla kareye tamamlanır ve 384 × 384 piksele indirilir; eğitim 320 × 320
piksel ile yapılır.

## Değerlendirme protokolü

- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)`, grup = bitki kimliği,
  hedef = ana evre (oturum düzeyinde).
- Her katta altı sınıfın bulunduğu denetlenir; sağlanmazsa `random_state` deterministik olarak
  artırılır ve kullanılan değer kaydedilir.
- Kat manifesti bir kez üretilir, bütün kamera ve ağırlık setlerinde değiştirilmeden kullanılır.
- Dış kat k: test = k, doğrulama = (k + 1) mod 5, eğitim = kalan üç kat.
- Beş katın test tahminleri birleştirilerek her kamera için 1.343 oturumu kapsayan kat-dışı
  tahmin kümesi elde edilir.

## Eğitim

2 ağırlık seti × 4 kamera × 5 kat = 40 eğitim. `imgsz=320`, `epochs=50`, `patience=15`,
`batch=64`, tek sabit eğitim tohumu (0), `deterministic=True`. Diğer hiperparametreler
Ultralytics 8.4.155 varsayılanıdır ve bütün koşullarda aynıdır; koşu başına değerler
Ultralytics'in ürettiği `args.yaml` dosyalarında saklanır.

## Geç füzyon

Bir oturumun dört kamera olasılığının aritmetik ortalaması alınır, argmax ile sınıf seçilir.
Ek eğitim gerektirmez.

## Metrikler ve istatistik

Ana metrik makro F1. Yanında dengeli doğruluk, doğruluk, Cohen kappa, sınıf bazında P/R/F1,
normalize karışıklık matrisi, ordinal hata (1–6 evre indeksi farkı) ve komşu evreye düşen hata
oranı raporlanır. BBCH kodlarının doğrudan sayısal farkı kullanılmaz.

Güven aralıkları bitki kümeli bootstrap ile hesaplanır (B = 2.000): yeniden örnekleme birimi
bitkidir ve aynı örneklem bütün sistemlere uygulanır, böylece karşılaştırmalar eşleştirilmiştir.
İki yönlü bootstrap p değerinin çözünürlük sınırı 1/B'dir; bu tabana ulaşan değerler üst sınır
olarak raporlanır. Holm düzeltmesi üç aile içinde ayrı uygulanır: (a) ağırlık seti başına altı
kamera çifti, (b) ağırlık seti başına füzyon ile dört tek kamera, (c) görünüm başına iki ağırlık seti.

## Maliyet ölçümü

Tek kamera (oturum başına bir model çağrısı) ile dört kameralı füzyon (dört çağrı) karşılaştırılır.
Sınıflara dengeli dağıtılmış 100 oturumluk deterministik bir örneklem kullanılır; görüntüler
ölçümden önce belleğe alınır, böylece disk okuma ve JPEG çözme süreleri gecikmeye karışmaz.

## Sınırlılıklar

Tek çeşit, tek sera kabini, 101 bitki, oturum başına tek poz, tek eğitim tohumu. Evreler zamanda
örtüşebildiği için oturum başına tek etiketli sınıflandırma yapısal belirsizlik taşır. Kamera
kimlikleri ile düşey eğim açılarının eşleşmesi doğrulanamamıştır.
