---
title: "Explainability of Vision Transformers: Phần 3.3 – 8"
subtitle: "Báo cáo tóm tắt theo bài báo (Pruning-based, Inherently Explainable, Other tasks, Evaluation, Tools, Datasets, Discussion, Conclusion)"
author: "Nhóm 6 – Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM"
---

> **Nguồn:** R. Kashefi, L. Barekatain, M. Sabokrou, F. Aghaeipoor, *Explainability of Vision Transformers: A Comprehensive Review and New Perspectives*, arXiv:2311.06786, 2023. Báo cáo này tóm tắt các mục 3.3 đến 8 đúng theo cách bài báo trình bày; không bổ sung thêm công thức ngoài những gì bài báo có.

# 3.3. Pruning-based Methods

Pruning (cắt tỉa) là một hướng được dùng rộng rãi để tối ưu hiệu quả và độ phức tạp của transformer. Các kỹ thuật này cố gắng loại bỏ những thành phần dư thừa hoặc ít thông tin như token, patch, block hoặc attention head. Bài báo cho thấy các kỹ thuật này cũng cải thiện được khả năng giải thích của các kiến trúc dựa trên ViT. Một số phương pháp được phát triển chủ đích cho explainability, một số khác chủ yếu để tăng hiệu quả mà không nhắm tới explainability, nhưng vẫn có thể tác động tích cực đến nó; ngoài ra còn nhóm thứ ba chưa hề nhắc tới explainability nhưng có thể có hàm ý tốt. Vì vậy các phương pháp pruning cho ViT được chia thành ba nhóm.

| Nhóm | Ý nghĩa | Phương pháp |
|:--|:--|:--|
| Explicitly explainable | Được xây dựng chủ đích để mô hình ít phức tạp và dễ diễn giải hơn | IA-RED², X-Pruner, Vision DiffMask |
| Implicitly explainable | Chủ yếu để tăng hiệu quả nhưng cũng cải thiện đáng kể explainability | DynamicViT, EViT |
| Possibly explainable | Tác giả không nhắc đến explainability, nhưng có thể có hàm ý tốt | Patch Slimming, HVT |

## 3.3.1. Explicitly Explainable Methods

**IA-RED² (Pan et al., 2021a).** Phương pháp tìm sự cân bằng giữa hiệu quả và khả năng diễn giải, đồng thời giữ được tính đa dụng và linh hoạt của ViT gốc. Cách làm: loại bỏ động các patch ít thông tin để rút ngắn chuỗi đầu vào. IA-RED² chia cấu trúc ViT gốc thành hai thành phần: một *multi-head interpreter* và các *processing block*. Interpreter đánh giá mức độ thông tin của các patch token, loại bỏ token có điểm dưới một ngưỡng, và chuyển các token còn lại qua các processing block. IA-RED² đạt tăng tốc đáng kể (tới 1,4 lần) với mất mát độ chính xác rất nhỏ. Các tác giả cũng chứng minh module diễn giải học được vượt trội hơn các phương pháp diễn giải hiện có.

**X-Pruner (Yu & Xiang, 2023).** Phương pháp được thiết kế riêng để cắt các đơn vị ít quan trọng. Đầu tiên tạo một *mặt nạ nhận biết explainability* (explainability-aware mask) để đo đóng góp của mỗi đơn vị có thể cắt trong việc dự đoán một lớp cụ thể. Sau đó loại bỏ các đơn vị có giá trị mặt nạ thấp nhất. X-Pruner tiết kiệm đáng kể FLOPs và tạo ra các bản đồ giải thích trực quan chính xác, gọn và ít nhiễu trên mô hình đã cắt.

**Vision DiffMask (Nalmpantis et al., 2023).** Là bản chuyển thể của một phương pháp vốn dành cho xử lý ngôn ngữ (De Cao et al., 2020), nhằm xác định tập con tối thiểu của ảnh mà ViT cần để dự đoán đúng. Phương pháp pruning này gồm các cơ chế cổng (gate) trên từng lớp ViT, được tối ưu để giữ nguyên đầu ra gốc khi che đầu vào. Khi huấn luyện, các gate "bỏ phiếu" xem một patch nên bị loại hay giữ; các phiếu được gộp thành mặt nạ cuối cùng. Khi suy luận, gate dự đoán xác suất thay vì phiếu nhị phân, nên tạo ra bản đồ attribution liên tục trên đầu vào. Các tác giả kiểm tra bằng faithfulness tasks và perturbation tests, và cho thấy Vision DiffMask phân định rõ tập con kích hoạt khỏi phần còn lại của ảnh, từ đó giúp hiểu dự đoán của mô hình tốt hơn.

## 3.3.2. Implicitly Explainable Methods

Một số phương pháp pruning được phát triển chủ yếu để tăng hiệu quả mạng, nhưng cũng dẫn đến cải thiện đáng kể explainability.

**DynamicViT (Rao et al., 2021).** Dùng dynamic token pruning để tăng tốc ViT. Phương pháp dùng một *prediction module* nhẹ để ước lượng điểm quan trọng của từng token dựa trên đặc trưng hiện tại. Module này được thêm vào nhiều lớp khác nhau để cắt các token dư thừa theo từng cấp. Ngoài ra, một chiến lược *attention masking* được dùng để phân biệt token bằng cách chặn tương tác của chúng với các token khác. Kết quả thực nghiệm cho thấy DynamicViT cải thiện đáng kể hiệu quả mà vẫn giữ độ chính xác. DynamicViT còn tăng khả năng diễn giải bằng cách xác định, từng bước, các phần ảnh quan trọng nhất đối với phân loại. Đáng chú ý, việc cắt token có xu hướng giữ lại các token ở giữa ảnh, điều hợp lý vì vật thể trong phần lớn ảnh nằm ở vùng này.

**EViT – Efficient Vision Transformer (Liang et al., 2022).** Là một cách tiếp cận khác nhằm tăng tốc ViT bằng cách tổ chức lại các token của ảnh. Khái niệm chính là *attentiveness* (độ chú ý), đánh giá mối quan hệ giữa class token và các token của ảnh. Dựa trên điểm attentiveness của token, EViT giữ lại các token liên quan nhất và hợp nhất các token ít chú ý thành một token duy nhất. Điều này giảm số lượng token, và do đó giảm chi phí tính toán khi mạng càng sâu. EViT không chỉ tăng độ chính xác mà còn giữ chi phí tính toán tương đương hoặc thấp hơn so với DeiT/LV-ViT gốc. Để đánh giá khả năng diễn giải, các tác giả trực quan hóa quá trình xác định token được chú ý trên nhiều mẫu (Hình 4 của bài báo): các token ít chú ý bị hợp nhất dần hoặc bị loại, trong khi các token nhiều thông tin được giữ lại. Nhờ đó ViT tập trung vào các token đặc thù theo lớp trong ảnh, dẫn đến khả năng diễn giải tốt hơn.

## 3.3.3. Possibly Explainable Methods

Nhóm này gồm các phương pháp pruning có thể không được thiết kế để cải thiện khả năng diễn giải, nhưng có tiềm năng cho các nghiên cứu tiếp theo.

**Patch Slimming (Tang et al., 2022).** Là thuật toán tăng tốc ViT bằng cách nhắm vào các patch dư thừa trong ảnh đầu vào theo hướng *từ trên xuống*. Cụ thể, kỹ thuật này xác định các patch nhiều thông tin nhất ở lớp cuối, rồi dùng chúng để dẫn dắt việc chọn patch ở các lớp trước. Khả năng chọn lọc giữ lại các patch chủ chốt có thể làm nổi bật các đặc trưng thị giác quan trọng, từ đó có thể nâng cao khả năng diễn giải. Tuy nhiên các tác giả không bàn đến điều này trong nghiên cứu của họ.

**HVT – Hierarchical Visual Transformer (Pan et al., 2021b).** Được đề xuất để tăng khả năng mở rộng và hiệu năng của ViT. HVT giảm dần độ dài chuỗi khi độ sâu mô hình tăng. Ngoài ra, bằng cách chia các block ViT thành các stage và áp dụng pooling ở mỗi stage, hiệu quả tính toán được cải thiện đáng kể. HVT cũng dự đoán mà không cần class token, giúp giảm thêm chi phí. Kết quả thực nghiệm cho thấy HVT vượt DeiT trên nhiều benchmark phân loại ảnh với chi phí tính toán tương đương. Do mô hình tập trung dần vào các thành phần quan trọng nhất, có cơ hội khảo sát tác động của nó đến explainability.

## So sánh các phương pháp pruning (Bảng 2)

Kết quả trên DeiT-S, ImageNet, lấy từ báo cáo gốc của từng phương pháp.

| Phương pháp | GFLOPs ↓ (%) | Top-1 Accuracy ↓ (%) | Throughput ↑ (%) |
|:--|:-:|:-:|:-:|
| IA-RED² | – | 0,7 | 46 |
| X-Pruner | 47,9 | 1,09 | – |
| DynamicViT | 37 | 0,5 | 54 |
| EViT | 35 | 0,3 | 50 |
| Patch Slimming | 47,8 | 0,3 | 43 |
| HVT | 47,8 | 1,8 | – |

Phần lớn các phương pháp pruning đã được đánh giá kỹ về hiệu quả. Tuy nhiên có một khoảng trống đáng kể trong tài liệu về việc đánh giá explainability của chúng: đa số chỉ đưa ra trực quan hóa và phân tích định tính, không có đánh giá định lượng (xem mục 4).

# 3.4. Inherently Explainable Methods

Một số hướng tiếp cận tập trung phát triển các mô hình có khả năng tự giải thích. Các phương pháp này có thể dùng các công cụ dễ diễn giải như cây quyết định hoặc hệ thống dựa trên luật, hoặc áp đặt tính thưa lên kiến trúc mô hình. Tuy nhiên các mô hình này thường khó đạt cùng độ chính xác với các mô hình hộp đen phức tạp hơn, vì vậy cần cân nhắc kỹ sự cân bằng giữa explainability và hiệu năng (Rudin, 2019).

**ViT-CX (Xie et al., 2022).** Là phương pháp giải thích dựa trên mặt nạ (mask-based), được thiết kế riêng cho ViT. Phương pháp dựa vào các patch embedding và tác động nhân quả của chúng lên đầu ra của mô hình, thay vì tập trung vào attention dành cho chúng. Gồm hai pha: sinh mặt nạ và gộp mặt nạ, từ đó tạo ra các saliency map có ý nghĩa hơn. Giải thích do ViT-CX tạo ra có độ trung thực (faithfulness) với mô hình tốt hơn đáng kể.

**Phân tích tương tác giữa các patch (Ma et al., 2023).** Đề xuất một cách trực quan hóa có khả năng giải thích để phân tích tương tác theo từng cặp patch trong ViT. Các tác giả trước hết định lượng tương tác giữa các patch, rồi xác định các patch có thể không có tính phân biệt, nhằm cung cấp một cửa sổ attention thích ứng giới hạn trường tiếp nhận (receptive field) của mỗi patch. Theo quan sát của họ, kích thước trường tiếp nhận của patch tương quan với thông tin ngữ nghĩa của patch: các patch của đối tượng chính có xu hướng có trường tiếp nhận nhỏ hơn so với patch nền. Do đó kết cấu hoặc cấu trúc cục bộ được học nhiều hơn ở trường tiếp nhận nhỏ, còn mối tương quan giữa đối tượng và nền tập trung ở trường tiếp nhận lớn.

**ViT-NeT (Kim et al., 2022).** Là ViT có bộ giải mã dạng cây thần kinh (neural tree decoder), mô tả quá trình ra quyết định thông qua cấu trúc cây và các prototype, cho phép diễn giải trực quan kết quả. Mô hình xử lý tốt phân loại các đối tượng chi tiết (fine-grained) nhờ đồng thời xét sự tương đồng giữa các lớp khác nhau và sự biến thiên trong cùng một lớp.

**R-Cut (Niu et al., 2023).** Tăng khả năng giải thích của ViT bằng *Relationship Weighted Out* và *Cut*. Module Relationship Weighted Out tập trung trích thông tin đặc thù theo lớp từ các lớp trung gian, nhấn mạnh các đặc trưng liên quan. Module Cut thực hiện phân rã đặc trưng mịn, có xét các yếu tố như vị trí, kết cấu và màu sắc. Kết hợp hai module có thể tạo ra các bản đồ giải thích trực quan dày và đặc thù theo lớp.

# 3.5. Explainability of Other Tasks

Explainability của kiến trúc ViT cho các tác vụ thị giác ngoài phân loại vẫn đang được khám phá. Một số phương pháp được đề xuất riêng cho các tác vụ khác, và các kết quả ban đầu đều đầy hứa hẹn.

**eX-ViT (Yu et al., 2023).** Là ViT giải thích được, thiết kế cho phân đoạn ngữ nghĩa giám sát yếu (weakly supervised semantic segmentation). Đây là một mạng Siamese (Koch et al., 2015) với hai nhánh xử lý cặp đầu vào (hai phiên bản tăng cường của cùng một dữ liệu gốc) theo cách tự giám sát và cuối cùng học ra các bản đồ attention có thể diễn giải. Mỗi nhánh có transformer encoder với hai module mới: *Explainable Multi-Head Attention* và *Attribute-guided Explainer*. Để tăng khả năng diễn giải, một module mất mát dẫn hướng bởi thuộc tính được đưa vào với ba hàm mất mát: global-level attribute-guided loss, local-level attribute discriminability loss và attribute diversity loss. Hàm đầu dùng attention map để tạo đặc trưng có thể diễn giải, hai hàm sau tăng cường việc học thuộc tính.

**DINO (Caron et al., 2021).** Là phương pháp tự giám sát đơn giản, hoạt động như một cách tự chưng cất (self-distillation) không cần nhãn. Các tác giả báo cáo rằng các attention map học được cuối cùng giữ được hiệu quả các vùng ngữ nghĩa của ảnh, và có thể dùng để giải thích.

**Generic Attention-model (Chefer et al., 2021b).** Là cách tiếp cận mới để giải thích dự đoán của các kiến trúc dựa trên Transformer, bao gồm cả transformer hai phương thức (bi-modal) và transformer có co-attention. Phương pháp áp dụng cho ba kiến trúc phổ biến nhất: self-attention thuần, self-attention kết hợp co-attention, và encoder-decoder attention. Để kiểm tra, các tác giả dùng tác vụ hỏi đáp trực quan (VQA), tuy nhiên phương pháp cũng áp dụng được cho các tác vụ thị giác khác như phát hiện đối tượng và phân đoạn ảnh.

**ATMAN (Deb et al., 2023).** Tìm cách hiểu dự đoán của transformer thông qua thao tác attention hiệu quả về bộ nhớ. ATMAN là phương pháp nhiễu loạn (perturbation) không phụ thuộc phương thức (modality-agnostic), tận dụng cơ chế attention để tạo bản đồ liên quan của đầu vào đối với dự đoán đầu ra.

**Concept-Transformer (Rigotti et al., 2022).** Khái quát hóa attention từ các đặc trưng đầu vào cấp thấp lên các khái niệm cấp cao, bằng một module học sâu gọi là Concept-Transformer. Module này tạo giải thích cho đầu ra mô hình bằng cách làm nổi bật các điểm attention trên các khái niệm cấp cao do người dùng định nghĩa, đảm bảo cả tính hợp lý (plausibility) và độ trung thực (faithfulness).

# 4. Explanation Evaluation

Đánh giá một phương pháp giải thích có mô tả tốt quá trình suy luận của mô hình hay không là một thách thức. Tài liệu đề xuất một loạt tiêu chí đánh giá để giúp chọn và thiết kế kỹ thuật giải thích phù hợp nhất (Hình 5 của bài báo tóm tắt các tiêu chí này).

**Deletion và Insertion** (Petsiuk et al., 2018). Dùng để đánh giá độ trung thực của một saliency map với mô hình đích; đo mức độ saliency map xác định đúng các pixel có ảnh hưởng nhất đến dự đoán. *Deletion AUC* đo điểm của lớp mục tiêu giảm nhanh thế nào khi các pixel quan trọng bị loại khỏi ảnh. *Insertion AUC* đo điểm của lớp mục tiêu tăng nhanh thế nào khi các pixel quan trọng được thêm vào một canvas trống. Deletion AUC nhỏ hơn và Insertion AUC lớn hơn cho thấy saliency map xác định chính xác các pixel quan trọng, tức độ trung thực cao hơn.

**Effective Complexity.** Đánh giá số lượng attribution vượt một ngưỡng, thể hiện mức quan trọng hay không quan trọng của các đặc trưng tương ứng. Thước đo này quan trọng với các giải thích hướng trực quan hóa. Effective complexity thấp nghĩa là một số đặc trưng có thể bị bỏ qua dù chúng có ảnh hưởng, vì ảnh hưởng thực sự nhỏ. Điều này có thể làm mất một phần độ chính xác thực tế nhưng cho giải thích tích hợp và đơn giản hơn.

**Faithfulness.** Là phương pháp đánh giá chất lượng feature attribution mà không cần con người can thiệp. Nó đo mức attribution của đặc trưng phù hợp/tương quan thế nào với dự đoán của mô hình. Phương pháp loại bỏ các đặc trưng cụ thể và phân tích tương quan tuyến tính giữa logit dự đoán của mô hình và attribution trung bình tương ứng với tập đặc trưng đó. Quá trình tính trên nhiều lần chạy và nhiều mẫu thử, tạo ra giá trị từ −1 đến 1 cho mỗi cặp đầu vào–attribution.

**(In)fidelity.** Đánh giá một giải thích nắm bắt tốt thế nào sự thay đổi trong dự đoán của mô hình khi đầu vào bị nhiễu loạn đáng kể. Mục tiêu là tìm giải thích làm cực tiểu độ đo infidelity, tức giải thích có độ trung thực cao, phản ánh đúng hành vi của mô hình trong các kịch bản nhiễu loạn khác nhau. Infidelity là độ đo bổ sung cho sensitivity.

**Intersection over Union (IoU).** Là độ đo chuẩn cho bộ phát hiện đối tượng, cũng được dùng để đánh giá phương pháp giải thích bằng cách đo mức bản đồ giải thích chồng khớp với hộp bao ground-truth của đối tượng quan tâm. Quy trình: trước hết tăng độ phân giải của bản đồ giải thích cho bằng kích thước ảnh gốc; sau đó áp ngưỡng 0,2 để loại vùng nền; cuối cùng IoU là giao của hộp dự đoán và hộp ground-truth chia cho hợp của chúng. IoU càng cao, bản đồ càng định vị tốt đối tượng.

**Perturbation tests.** Che dần các token đầu vào dựa trên giải thích do phương pháp cho ra. Độ chính xác của các đầu vào bị che được đo bằng diện tích dưới đường cong (AUC). Có hai loại: *positive* (che từ token liên quan nhất đến ít nhất, mong đợi hiệu năng giảm mạnh khi che các pixel quan trọng) và *negative* (che từ ít liên quan nhất đến nhiều nhất, giải thích tốt sẽ giữ được độ chính xác khi bỏ các pixel không liên quan).

**Pointing Game.** Đánh giá saliency map so với các hộp bao do con người chú thích, là phần ảnh con người cho là thiết yếu cho dự đoán. Nếu pixel có saliency cao nhất nằm trong hộp thì tính là một hit, ngược lại là miss. Độ chính xác Pointing Game:

$$\text{accuracy} = \frac{\#\text{hits}}{\#\text{hits} + \#\text{misses}} \qquad (12)$$

*Grid Pointing Game* mở rộng cho lưới ảnh: tạo các lưới ảnh tổng hợp chứa nhiều đối tượng khác nhau, và đo tỉ lệ attribution dương mà phương pháp gán cho đúng ô con (sub-image).

**Segmentation tests.** Xem mỗi trực quan hóa như một phân đoạn mềm của ảnh và so với ground-truth của dataset. Giải thích được tạo theo lớp dự đoán và đánh giá bằng bốn độ đo: pixel accuracy, mean IoU (mIoU), mean Average Precision (mAP) và mean F1 (mF1). Pixel accuracy và mIoU tính sau khi nhị phân hóa giải thích bằng ngưỡng là trung bình các điểm attribution; còn mAP và mF1 tính bằng trung bình các điểm tương ứng trên nhiều mức ngưỡng.

**Sensitivity.** Đánh giá giải thích thay đổi thế nào khi đầu vào bị nhiễu loạn nhỏ; đảm bảo rằng khi các đầu vào giống nhau và đầu ra mô hình gần nhau thì giải thích tương ứng cũng phải giống nhau.

**Sensitivity-n.** Được đề xuất để kiểm tra các giá trị attribution cụ thể thay vì chỉ xét thứ hạng quan trọng. Phương pháp đo mô hình thay đổi dự đoán bao nhiêu khi gỡ một đặc trưng. Nó chọn ngẫu nhiên các tập con đặc trưng và tính tương quan giữa mức thay đổi dự đoán và tổng điểm attribution của các đặc trưng đó, trên nhiều tập con khác nhau rồi lấy trung bình. Điểm Sensitivity-n cao cho thấy attribution chính xác và tương quan tốt với tác động lên dự đoán khi gỡ đặc trưng.

# 5. Tools and Frameworks

Cài đặt các kỹ thuật giải thích từ đầu là thách thức đáng kể; có nhiều thư viện được thiết kế riêng để hỗ trợ.

- **Captum** (Kokhlikyan et al., 2020): toolkit mã nguồn mở thống nhất cho PyTorch, xây dựng chuyên cho khả năng diễn giải của mô hình ML. Cung cấp cài đặt chung cho nhiều thuật toán attribution dựa trên gradient và nhiễu loạn, dùng ở mức đặc trưng, neuron và lớp. Đi kèm bộ độ đo đánh giá mạnh, dễ mở rộng, dùng được cho cả mô hình phân loại và không phân loại, hỗ trợ nhiều kiểu dữ liệu: ảnh, văn bản, âm thanh, video.
- **InterpretDL** (Li et al., 2022): toolkit diễn giải mô hình cho nền tảng PaddlePaddle (nền tảng học sâu độc lập đầu tiên của Trung Quốc). Có nhiều thuật toán cổ điển và mới như LIME, Grad-CAM, Integrated Gradients… để giải thích CNN, perceptron nhiều lớp, transformer và ViT. Cũng cung cấp benchmark để các nhà nghiên cứu so sánh thuật toán mới với các thuật toán hiện có.
- **AttentionViz** (Yeh et al., 2023): công cụ trực quan hóa thiết kế riêng cho các mẫu attention của cả transformer ngôn ngữ và thị giác. Cho cái nhìn toàn cục về attention, cung cấp góc nhìn độc đáo về tương tác embedding query–key ở các lớp khác nhau, cho phép phân tích các mẫu toàn cục trên nhiều chuỗi đầu vào.
- **Quantus** (Hedström et al., 2023): toolkit hiện đại giúp đánh giá giải thích mạng nơ-ron. Trước Quantus thiếu công cụ chuyên biệt để đánh giá độ tin cậy và độ chính xác của giải thích. Đây là bộ công cụ Python với tập độ đo đánh giá phong phú, có tổ chức cùng các hướng dẫn, cung cấp mã nguồn mở trên PyPI.

# 6. Datasets

- **CIFAR-10:** 60 000 ảnh màu nhỏ, 10 lớp; thường dùng để huấn luyện và đánh giá thuật toán phân loại ảnh.
- **ImageNet:** tập dữ liệu rất lớn gồm hàng triệu ảnh có nhãn thuộc hàng nghìn lớp; ILSVRC-12 (bản 2012) là nguồn quan trọng để huấn luyện và đánh giá nhiều tác vụ thị giác.
- **COVID-QU-Ex:** tập dữ liệu y tế gồm 33 920 ảnh X-quang ngực hỗ trợ chẩn đoán COVID-19, ca khỏe mạnh và các nhiễm trùng không phải COVID như viêm phổi.
- **ImageNet-Segmentation:** được điều chỉnh cho các thí nghiệm phân đoạn.
- **Kinetics-400:** IA-RED² dùng cho phân loại video.

# 7. Discussion and Future Works

- **Attribution:** hầu hết phương pháp model-agnostic thiết kế cho DNN cũng áp dụng được cho transformer, nhưng tận dụng attention weight có thể tăng khả năng diễn giải. Feature attribution được quan tâm nhiều hơn feature visualization; cần bổ sung hướng sau để làm rõ quy trình hoạt động toàn cục của mô hình.
- **Attention weight:** việc dùng trực tiếp attention weight còn thiếu sót trong tác vụ thị giác, gây lo ngại về tính phù hợp của attention cho giải thích. Chúng chỉ là một phần của chuỗi phép biến đổi phi tuyến phức tạp, có thể chưa đủ để hiểu toàn diện các phụ thuộc trong mô hình. Cần thảo luận sâu về tính phù hợp của attention weight (Jain & Wallace, 2019). Cách kết hợp các ma trận attention trong các block multi-head self-attention cần được xem xét kỹ hơn (Voita et al., 2019); thay vì chỉ tổ hợp tuyến tính các attention cuối của các lớp, nên khám phá các phương pháp toán học khác.
- **Pruning:** cần chú ý hơn đến khía cạnh explainability để tận dụng các kiến trúc không dày đặc và minh bạch hơn. Hầu hết các phương pháp, kể cả nhóm explicit, chưa được đánh giá bằng tiêu chí định lượng về faithfulness và completeness. Cần nghiên cứu thêm và đánh giá nghiêm ngặt hơn.
- **Benchmark:** các benchmark đánh giá chủ yếu hướng về phân loại và cần mở rộng ra các ứng dụng khác. Explainability của ứng dụng đa phương thức và của mô hình sinh dựa trên thị giác vẫn là lĩnh vực non trẻ, nhiều cơ hội.
- **Ứng dụng thực tế:** thiếu nghiên cứu về việc tận dụng lợi ích của các phương pháp explainability để gỡ lỗi và cải thiện mô hình, cũng như tăng độ công bằng và tin cậy, đặc biệt trong ứng dụng ViT. Khai thác các hiểu biết này có thể giúp ViT đáng tin cậy và công bằng hơn, cải thiện ra quyết định và tăng tỉ lệ thành công trong ứng dụng thực tế.

# 8. Conclusion

Bài báo trình bày tổng quan toàn diện các kỹ thuật explainability đề xuất cho vision transformer, phân loại các phương pháp theo động cơ, cấu trúc và kịch bản ứng dụng thành năm nhóm. Bài cũng trình bày chi tiết các tiêu chí đánh giá explainability, các công cụ và khung làm việc. Cuối cùng, bài thảo luận một số vấn đề thiết yếu nhưng chưa được khám phá để tăng explainability của ViT và gợi ý các hướng nghiên cứu tiềm năng. Các tác giả kỳ vọng bài tổng quan giúp người đọc hiểu rõ hơn cơ chế bên trong của vision transformer và làm nổi bật các vấn đề mở cho nghiên cứu sau này.

# Tài liệu tham khảo (các công trình được nhắc trong phần này)

Pan et al. (2021a) IA-RED²; Yu & Xiang (2023) X-Pruner; Nalmpantis et al. (2023) Vision DiffMask; De Cao et al. (2020) DiffMask; Rao et al. (2021) DynamicViT; Liang et al. (2022) EViT; Tang et al. (2022) Patch Slimming; Pan et al. (2021b) HVT; Rudin (2019); Xie et al. (2022) ViT-CX; Ma et al. (2023); Kim et al. (2022) ViT-NeT; Niu et al. (2023) R-Cut; Yu et al. (2023) eX-ViT; Koch et al. (2015); Caron et al. (2021) DINO; Chefer et al. (2021b) Generic Attention-model; Deb et al. (2023) ATMAN; Rigotti et al. (2022) Concept-Transformer; Petsiuk et al. (2018) RISE; Kokhlikyan et al. (2020) Captum; Li et al. (2022) InterpretDL; Yeh et al. (2023) AttentionViz; Hedström et al. (2023) Quantus; Jain & Wallace (2019); Voita et al. (2019). Danh mục đầy đủ xem phần References của bài báo gốc (arXiv:2311.06786).
