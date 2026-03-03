# Báo cáo: Ứng dụng World Model (I-JEPA) cho phát hiện vùng bị chỉnh sửa trong ảnh

## 1. Ý tưởng cốt lõi

**World Model** (I-JEPA) học cách hiểu thế giới thực bằng cách dự đoán các phần bị che (masked) của ảnh trong **không gian đặc trưng** (latent space), không phải pixel. Sau khi được train trên ảnh thật, model xây dựng một "mô hình thế giới" — hiểu rằng các vùng trong ảnh phải có mối quan hệ nhất quán với nhau.

**Ý tưởng phát hiện chỉnh sửa:** Khi một vùng trong ảnh bị ghép/sửa, nó sẽ **phá vỡ sự nhất quán** mà World Model đã học → model dự đoán sai vùng đó → **prediction error cao = vùng bị chỉnh sửa**.

```
Ảnh thật:     Model dự đoán chính xác → error thấp  ✅
Ảnh bị ghép:  Model dự đoán sai vùng ghép → error CAO → PHÁT HIỆN! 🚨
```

## 2. Pipeline phát hiện

```
Ảnh nghi vấn → Chia thành các vùng nhỏ (patches)
             → Với mỗi vùng: che nó đi, dùng các vùng xung quanh để dự đoán
             → So sánh dự đoán vs thực tế
             → Vùng nào error cao = nghi vấn bị sửa
             → Tạo heatmap anomaly → Localize vùng chỉnh sửa
```

## 3. Phương pháp đang thử nghiệm

### Chiến lược chính: Multi-context Consistency Prediction (lấy cảm hứng từ V-JEPA2)

Dự đoán mỗi vùng từ **nhiều context khác nhau** (trái, phải, trên, dưới), rồi kiểm tra tính nhất quán:

| Bước | Mô tả |
|:---|:---|
| 1 | Che vùng target, dùng context **bên trái** để dự đoán → prediction A |
| 2 | Che vùng target, dùng context **bên phải** để dự đoán → prediction B |
| 3 | Che vùng target, dùng context **bên trên** để dự đoán → prediction C |
| 4 | So sánh A, B, C — nếu **nhất quán** → ảnh thật; nếu **mâu thuẫn** → bị sửa |

**Tại sao hoạt động?** Vùng thật: mọi context đều "kể cùng một câu chuyện" → predictions nhất quán. Vùng ghép: đến từ bối cảnh khác → context xung quanh "kể câu chuyện khác" → predictions mâu thuẫn.

### Bổ sung: Forensic Features (tầng thấp)

Phương pháp World Model chủ yếu phát hiện bất nhất ở tầng ngữ nghĩa/phong cách. Để phủ thêm các trường hợp ghép tinh vi (ví dụ: zebra→horse, cả hai đều hợp lý về ngữ nghĩa), kết hợp thêm đặc trưng forensic:

| Tín hiệu forensic | Phát hiện cái gì? |
|:---|:---|
| Noise pattern | Vùng ghép có noise/grain camera khác |
| JPEG artifacts | Vùng ghép bị nén 2 lần → blocking artifacts khác |
| Boundary artifacts | Viền ghép có micro-artifacts dù blend khéo |

## 4. Ưu điểm

- **Zero-shot**: Không cần train trên ảnh giả/manipulation → tổng quát hóa tốt cho mọi loại chỉnh sửa
- **Explainable**: Heatmap anomaly trực quan cho thấy vùng nào bị sửa và tại sao
- **Không phụ thuộc loại manipulation**: Hoạt động với splicing, copy-move, inpainting, AI-generated

## 5. Đánh giá

- **Dataset**: CASIA 2.0 (5,123 ảnh tampered có pixel-level ground truth mask)
- **Metrics**: Pixel-level F1, AUC, IoU
- **SOTA tham chiếu**: MVSS-Net (F1=0.587), ObjectFormer (AUC=0.817) — đều supervised
- **Mục tiêu zero-shot**: F1 ≥ 0.5, AUC ≥ 0.75

## 6. Tiến độ hiện tại

- ✅ Train I-JEPA World Model (ViT-Huge/14) — đã có pretrained
- ✅ Train RCDM (diffusion decoder) để visualize — đang train (step 6500+, 3 GPU)
- ✅ Xây dựng inference pipeline cơ bản (sliding window, cosine similarity)
- 🔄 Đang triển khai Multi-context prediction
- ⬜ Đánh giá trên CASIA 2.0
