---
title: "Script thuyết trình – Phần 3.3 đến 8 (20 phút)"
subtitle: "Explainability of Vision Transformers (Kashefi et al., 2023)"
---

# Cách dùng file này

- Mỗi slide có hai phần: **ĐỌC** (ngắn, đủ ý, để nói thành lời) và khối **GIẢI THÍCH CHO BẠN** (nền tím nhạt, *không đọc lên*) để bạn hiểu bản chất và trả lời khi thầy hỏi. Phần giải thích viết bằng lời đơn giản, không dùng công thức.
- Nội dung bám sát bài báo: slide, report và script đều chỉ tóm tắt những gì bài báo viết (công thức duy nhất là Pointing Game, phương trình 12).
- Số slide theo file `review_slides_33_to_8.pdf` (24 slide nội dung, sau đó là slide tham khảo và slide cảm ơn). Tổng thời gian khoảng **19 phút 55 giây**.
- Nếu bị gấp giờ: nói lướt slide 22 (Datasets) và slide 10.

## Ba khái niệm nền (hiểu 3 cái này là đủ để theo cả bài)

1. **Token / patch**: ViT cắt ảnh thành nhiều mảnh nhỏ (patch), mỗi mảnh được coi như một "từ" (token). Ngoài ra có thêm một token đặc biệt gọi là **class token (CLS)** để đưa ra dự đoán.
2. **Pruning (cắt tỉa)**: bỏ bớt token, patch, block hoặc head không cần thiết để mô hình chạy nhanh hơn. Phần còn lại là phần mô hình "cần", nên có thể dùng để giải thích.
3. **Faithfulness và Plausibility**: *faithful* = giải thích phản ánh đúng cái mô hình thật sự dùng. *Plausible* = giải thích trông hợp lý với con người. Hai thứ này có thể không giống nhau.

---

# Script từng slide

## Slide 1. Tiêu đề (0:10)

**ĐỌC:**
Xin chào cô và các bạn. Phần của em bắt đầu từ mục 3.3 của bài báo, gồm pruning-based methods, các phương pháp tự giải thích, giải thích cho các tác vụ khác, rồi phần đánh giá, công cụ và hướng phát triển.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> Chỉ cần chào và nối tiếp từ phần của bạn trước (attention-based methods).

## Slide 2. Outline (0:15)

**ĐỌC:**
Bài báo chia giải thích ViT thành năm nhóm. Hai nhóm đầu bạn em đã trình bày. Em trình bày ba nhóm in đậm là pruning-based, inherently explainable và other tasks. Sau đó là cách đánh giá giải thích, công cụ, dataset và phần thảo luận.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> Hình bên phải là taxonomy (cách phân loại) ở Hình 2 của bài báo. Năm nhóm là: common attribution, attention-based, pruning-based, inherently explainable, other tasks. Bạn chỉ cần nhớ ba nhóm cuối là của mình.

## Slide 3. Pruning-based methods: tổng quan (0:50)

**ĐỌC:**
Pruning là loại bỏ những phần dư thừa hoặc ít thông tin của transformer, như token, patch, block hay attention head. Nó được dùng rộng rãi để tăng hiệu quả và giảm độ phức tạp. Bài báo chỉ ra rằng pruning cũng giúp cải thiện khả năng giải thích. Các phương pháp được chia thành ba nhóm. Explicitly explainable là các phương pháp được làm riêng cho mục đích giải thích: IA-RED², X-Pruner và Vision DiffMask. Implicitly explainable là các phương pháp làm cho hiệu quả nhưng cũng cải thiện giải thích: DynamicViT và EViT. Possibly explainable là Patch Slimming và HVT, tác giả gốc không nhắc đến giải thích nhưng có thể có hàm ý tốt.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **Tại sao pruning liên quan đến giải thích?** Hãy tưởng tượng 196 người cùng chấm một bài thi, mỗi người xem một mảnh. Qua mỗi vòng, người nào không đóng góp thì cho nghỉ. Những người còn lại đến cuối chính là những mảnh ảnh mô hình *thật sự cần*. Nếu bỏ phần nền mà mô hình vẫn đoán đúng thì nghĩa là nó dựa vào vật thể. Đó là lý do pruning có thể coi như một lời giải thích.
> - **Cách nhớ ba nhóm:** "chủ ý, phụ, có thể". *Explicit* = chủ ý làm để giải thích. *Implicit* = làm để nhanh, giải thích là hệ quả phụ. *Possibly* = tác giả không nhắc, bài báo chỉ nói là "có thể".
> - **Thầy hỏi: "Pruning có phải giải thích không?"** Trả lời: nó sinh ra để tăng tốc, nhưng phần được giữ lại cho biết mô hình cần vùng nào; bài báo cũng nói cần đánh giá bằng các tiêu chí ở mục 4 mới kết luận chắc.

## Slide 4. IA-RED² (0:50)

**ĐỌC:**
IA-RED² tìm cách cân bằng giữa hiệu quả và khả năng diễn giải, mà vẫn giữ được tính linh hoạt của ViT gốc. Phương pháp chia ViT thành hai phần: multi-head interpreter và các processing block. Interpreter đánh giá mức độ thông tin của từng patch token. Những token có điểm dưới ngưỡng bị loại, các token còn lại đi tiếp qua các processing block. Nhờ vậy chuỗi đầu vào ngắn hơn và tăng tốc tới 1,4 lần với mất mát độ chính xác rất nhỏ. Các tác giả còn cho thấy module diễn giải học được vượt các phương pháp diễn giải hiện có.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **Hình dung:** interpreter là "người gác cổng". Mỗi patch phải xin điểm; điểm thấp thì không được vào các block tiếp theo, đỡ tốn tính toán.
> - **Giải thích ở đâu?** Điểm mà interpreter chấm cho từng patch chính là cho biết patch nào mô hình thấy quan trọng.
> - **Con số cần nhớ:** tăng tốc tới 1,4 lần; trong Bảng 2, giảm 0,7 điểm Top-1 và tăng 46% throughput.
> - **Thầy hỏi: "IA-RED² cắt cái gì?"** Cắt các patch token (cắt dữ liệu), không cắt trọng số mạng.

## Slide 5. X-Pruner (0:40)

**ĐỌC:**
X-Pruner là phương pháp thiết kế để cắt những đơn vị ít quan trọng. Trước hết tạo một explainability-aware mask để đo đóng góp của từng đơn vị có thể cắt đối với việc dự đoán một lớp cụ thể. Sau đó loại bỏ các đơn vị có giá trị mask thấp nhất. Kết quả là tiết kiệm đáng kể FLOPs và vẫn cho các bản đồ giải thích chính xác, gọn và ít nhiễu trên mô hình đã cắt.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **Hình dung:** mạng có nhiều "bộ phận" (như head). X-Pruner hỏi từng bộ phận: "bạn đóng góp bao nhiêu cho việc nhận ra từng loại vật?" Bộ phận nào đóng góp ít thì cắt.
> - **Khác IA-RED²:** IA-RED² cắt *token*; X-Pruner cắt *đơn vị bên trong mạng*. Từ "explainability-aware" nghĩa là tiêu chí cắt có tính đến đóng góp theo từng lớp, nên sau khi cắt bản đồ giải thích vẫn tốt.
> - **Con số:** giảm 47,9% GFLOPs, tụt 1,09 điểm Top-1 (Bảng 2). FLOPs là số phép tính.

## Slide 6. Vision DiffMask (0:55)

**ĐỌC:**
Vision DiffMask là bản chuyển thể từ một phương pháp cho xử lý ngôn ngữ của De Cao và cộng sự. Mục đích là tìm tập con nhỏ nhất của ảnh mà ViT cần để dự đoán đúng. Phương pháp gắn các cổng, gọi là gate, vào từng lớp của ViT, được tối ưu sao cho giữ nguyên đầu ra gốc khi đầu vào bị che. Khi huấn luyện, các gate bỏ phiếu xem patch nên bị loại hay giữ, và các phiếu được gộp thành mask cuối. Khi suy luận, gate cho xác suất, nên ta có một bản đồ attribution liên tục. Các tác giả kiểm tra bằng faithfulness tasks và perturbation tests, và thấy phương pháp tách rõ phần kích hoạt dự đoán khỏi phần còn lại của ảnh.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **Hình dung:** lấy giấy đen che ảnh con mèo, che càng nhiều càng tốt miễn là người khác nhìn phần còn lại vẫn đoán "mèo". Phần không bị che là phần *đủ để đoán*. Vision DiffMask dạy mô hình làm việc đó.
> - **Gate bỏ phiếu:** mỗi lớp của ViT có một cổng; mỗi cổng nói "giữ" hay "bỏ" cho từng patch; gộp các lá phiếu lại thành mask cuối. ViT bản thân không bị sửa.
> - **Lúc suy luận:** thay vì 0/1 thì cổng cho xác suất, nên bản đồ mượt, chỗ nào sáng nghĩa là cần giữ.
> - **Thầy hỏi: "Khác Grad-CAM?"** DiffMask *học* một mask với điều kiện "đầu ra không đổi", tức tìm tập đủ; Grad-CAM chỉ dựa trên gradient.

## Slide 7. DynamicViT (0:55)

**ĐỌC:**
DynamicViT dùng cắt token động để tăng tốc ViT. Một prediction module nhẹ ước lượng điểm quan trọng của từng token dựa trên đặc trưng hiện tại. Module này được thêm vào nhiều lớp để cắt token dư thừa dần dần theo từng cấp. Ngoài ra có chiến lược attention masking để chặn tương tác của các token đã bị cắt với các token còn lại. Kết quả: hiệu quả tăng rõ rệt mà vẫn giữ độ chính xác. DynamicViT còn cho thấy từng bước những phần ảnh quan trọng nhất cho phân loại; đáng chú ý là token được giữ thường nằm ở giữa ảnh, điều hợp lý vì vật thể hay nằm ở đó.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **Hình dung:** giống như một vòng loại nhiều vòng; ở mỗi vòng, một mạng nhỏ (prediction module) chấm điểm token, token điểm thấp bị loại. Sau nhiều vòng ta thấy rõ token nào "sống sót đến cuối".
> - **Attention masking để làm gì?** Trong lúc huấn luyện, token bị loại vẫn nằm trong dữ liệu; masking làm cho chúng không còn ảnh hưởng đến token khác, như thể không tồn tại.
> - **Phần giải thích:** nhìn từng tầng cắt để thấy vùng nào còn được giữ.
> - **Con số (Bảng 2):** giảm 37% GFLOPs, tụt 0,5 điểm, throughput tăng 54%.
> - **Thầy hỏi: "Khác EViT?"** DynamicViT *học thêm* một module để quyết định giữ/bỏ; EViT dùng sẵn attention của CLS, không học thêm.

## Slide 8. EViT (1:00)

**ĐỌC:**
EViT tăng tốc ViT bằng cách tổ chức lại các token của ảnh. Khái niệm chính là attentiveness, tức mối quan hệ giữa class token và các token của ảnh. Dựa trên điểm attentiveness, EViT giữ lại các token liên quan nhất và hợp nhất các token ít chú ý thành một token duy nhất. Nhờ vậy số token giảm đi và chi phí tính toán giảm khi mạng càng sâu. EViT không chỉ tăng độ chính xác mà còn giữ chi phí tương đương hoặc thấp hơn DeiT và LV-ViT. Hình 4 của bài báo minh họa: những token ít chú ý bị hợp nhất dần, vùng đen trong hình, còn token mang nhiều thông tin của vật thể được giữ lại, nên ViT tập trung vào các token đặc thù của lớp, giúp giải thích tốt hơn.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **Hình dung:** class token giống "trưởng nhóm" tổng hợp ý kiến. Token nào được trưởng nhóm chú ý nhiều là token quan trọng. EViT giữ các token đó; những token còn lại không vứt hẳn mà *gộp thành một token* để không mất hết thông tin nền.
> - **Điểm hay:** không cần học thêm module nào, dùng ngay attention của class token.
> - **Con số (Bảng 2):** giảm 35% GFLOPs, tụt chỉ 0,3 điểm, tăng 50% throughput. Đây là mức tụt thấp nhất cùng với Patch Slimming.
> - **Hình 4:** các vùng đen là token bị hợp nhất/loại; nền biến mất trước, vật thể còn lại. Nếu muốn, dán Hình 4 của bài báo vào khung trống trên slide (đường dẫn `figures/fig4_evit.png`).
> - **Thầy hỏi: "Vì sao gộp mà không bỏ?"** Để vẫn giữ lại một chút thông tin của các token ít quan trọng với chi phí rất thấp (chỉ một token).

## Slide 9. Patch Slimming và HVT (0:55)

**ĐỌC:**
Đây là hai phương pháp thuộc nhóm possibly explainable. Patch Slimming tăng tốc ViT bằng cách loại bỏ các patch dư thừa theo hướng từ trên xuống. Nó tìm các patch nhiều thông tin nhất ở lớp cuối, rồi dùng chúng để dẫn dắt việc chọn patch ở các lớp trước. Khả năng giữ lại các patch chủ chốt có thể làm nổi bật đặc trưng quan trọng, nhưng tác giả không bàn đến điều này. HVT, Hierarchical Visual Transformer, giảm dần độ dài chuỗi khi mô hình sâu hơn, bằng cách chia block thành các stage và dùng pooling ở mỗi stage. HVT còn dự đoán không cần class token. HVT vượt DeiT với chi phí tương đương, và việc tập trung dần vào thành phần quan trọng là một cơ hội để khảo sát giải thích.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **Patch Slimming "từ trên xuống":** lớp cuối đã chọn xong patch quan trọng; lùi về lớp trước, chỉ giữ những patch giúp ích cho các patch đã chọn.
> - **HVT:** giống thu nhỏ ảnh dần dần: cứ qua mỗi stage, chuỗi token ngắn hơn nhờ pooling (gộp nhóm). Không dùng class token nên cuối cùng phân loại bằng cách tổng hợp các token còn lại.
> - **Vì sao gọi là "possibly":** cả hai dần tập trung vào phần quan trọng nên *có thể* giải thích được, nhưng tác giả gốc không đề cập và chưa ai đánh giá. Bài báo nêu để chỉ ra cơ hội nghiên cứu.
> - **Con số (Bảng 2):** Patch Slimming giảm 47,8% GFLOPs, tụt 0,3; HVT giảm 47,8%, tụt 1,8.

## Slide 10. So sánh pruning (Bảng 2) (0:45)

**ĐỌC:**
Đây là Bảng 2 của bài báo, so sánh trên DeiT-S với ImageNet. Các phương pháp tiết kiệm từ 35 đến gần 48% FLOPs mà mất tối đa 1,8 điểm độ chính xác. Tuy nhiên điều quan trọng là bảng chỉ so sánh về hiệu quả. Bài báo nhận xét có một khoảng trống đáng kể: hầu hết các phương pháp chưa được đánh giá explainability, chỉ có hình minh họa và phân tích định tính, không có đánh giá định lượng như ở mục 4.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **Đọc bảng:** GFLOPs giảm 35% = giảm 35% số phép tính. Top-1 giảm 0,3 = độ chính xác giảm 0,3 điểm phần trăm. Throughput tăng 50% = mỗi giây xử lý nhiều hơn 50% ảnh. Dấu "–" là bài gốc không báo cáo.
> - **Lưu ý:** số liệu lấy từ báo cáo gốc của từng phương pháp nên điều kiện có thể không hoàn toàn giống nhau; đừng kết luận quá chặt về xếp hạng.
> - **Câu quan trọng nhất của slide:** "hiệu quả thì đo kỹ, giải thích thì không đo". Câu này nối sang mục 4 (tiêu chí đánh giá) và mục 7 (hướng phát triển).

## Slide 11. Inherently Explainable và ViT-CX (0:55)

**ĐỌC:**
Mục 3.4: các phương pháp tập trung phát triển mô hình có khả năng tự giải thích. Chúng có thể dùng công cụ dễ diễn giải như cây quyết định hoặc hệ thống dựa trên luật, hoặc áp đặt tính thưa lên kiến trúc. Tuy nhiên các mô hình này thường khó đạt độ chính xác bằng mô hình hộp đen, nên cần cân bằng giữa giải thích và hiệu năng, theo Rudin 2019. Phương pháp đầu tiên là ViT-CX, một phương pháp dựa trên mask dành riêng cho ViT. Nó dựa vào các patch embedding và tác động nhân quả của chúng lên đầu ra, thay vì dựa vào attention. Gồm hai pha, sinh mask và gộp mask, cho saliency map có ý nghĩa hơn và độ trung thực với mô hình tốt hơn đáng kể.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **"Inherently" = vốn có sẵn:** giải thích nằm sẵn trong cấu trúc mô hình, khác với giải thích sau khi đã huấn luyện xong.
> - **Cái giá:** thường kém chính xác hơn hộp đen một chút. Đây là ý của Rudin 2019.
> - **ViT-CX bằng lời thường:** thay vì tin attention, ta *thử che ảnh* bằng nhiều mask (sinh từ chính các embedding của mô hình), xem mask nào giữ lại thì mô hình vẫn đoán tốt; "nhân quả" nghĩa là ta *can thiệp* rồi quan sát kết quả, không chỉ nhìn tương quan. Sau đó gộp các mask lại thành bản đồ.
> - **Faithfulness** là độ trung thực của giải thích với mô hình (sẽ nói ở mục 4).

## Slide 12. Tương tác patch và ViT-NeT (1:05)

**ĐỌC:**
Công trình của Ma và cộng sự đề xuất cách trực quan hóa để phân tích tương tác giữa các patch trong ViT. Họ định lượng tương tác giữa các patch, xác định các patch không có tính phân biệt, và cấp cho mỗi patch một cửa sổ attention thích ứng giới hạn trường tiếp nhận. Quan sát chính: patch của đối tượng chính có trường tiếp nhận nhỏ hơn patch nền. Trường nhỏ học kết cấu và cấu trúc cục bộ, còn trường lớn học tương quan giữa đối tượng và nền. ViT-NeT là ViT có bộ giải mã dạng cây thần kinh. Nó mô tả quá trình ra quyết định bằng cấu trúc cây và các prototype, cho phép diễn giải trực quan. Mô hình xử lý tốt phân loại chi tiết vì cùng lúc xét sự giống nhau giữa các lớp và sự khác nhau trong cùng một lớp.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **Receptive field (trường tiếp nhận)** là vùng ảnh mà một patch "nhìn thấy". Phát hiện: patch vật thể chỉ nhìn các vùng gần để nắm kết cấu; patch nền nhìn xa. Đây là hiểu biết *toàn cục* về cách mô hình hoạt động, không phải bản đồ cho một ảnh.
> - **ViT-NeT như trò "20 câu hỏi":** thay vì một lớp phân loại duy nhất, mô hình đi qua một cây các câu hỏi dạng "có giống mẫu này không?". *Prototype* là mẫu đại diện của một bộ phận. Đường đi trên cây chính là lời giải thích.
> - **Fine-grained** là phân biệt những thứ rất giống nhau (ví dụ các loài chim gần nhau).

## Slide 13. R-Cut (0:40)

**ĐỌC:**
R-Cut tăng khả năng giải thích của ViT bằng hai module là Relationship Weighted Out và Cut. Module đầu trích thông tin đặc thù theo lớp từ các lớp trung gian và nhấn mạnh các đặc trưng liên quan. Module Cut phân rã đặc trưng mịn, có xét đến vị trí, kết cấu và màu sắc. Kết hợp hai module tạo ra các bản đồ giải thích dày và đặc thù theo lớp. Phương pháp được đánh giá bằng IoU test, em sẽ nói ở mục 4.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **Hình dung:** bước 1 lấy "thông tin của lớp cần giải thích" từ các tầng giữa. Bước 2 chia ảnh thành các mảng liên kết nhau dựa trên vị trí, kết cấu và màu (ví dụ mảng "thân", mảng "đầu", mảng "nền").
> - **Class-specific** (đặc thù theo lớp): bản đồ thay đổi tùy lớp ta đang hỏi (mèo hay chó).
> - **Dense** (dày): bản đồ chi tiết, không phải mấy ô thô.
> - Bài báo tổng quan không đi sâu chi tiết bên trong; nếu thầy hỏi sâu hơn, nói "chi tiết xem bài R-Cut gốc".

## Slide 14. eX-ViT và DINO (1:00)

**ĐỌC:**
Mục 3.5: giải thích cho các tác vụ khác ngoài phân loại, hiện vẫn đang được khám phá nhưng kết quả ban đầu rất hứa hẹn. eX-ViT là ViT giải thích được cho phân đoạn ngữ nghĩa giám sát yếu. Đây là mạng Siamese với hai nhánh, xử lý hai phiên bản tăng cường của cùng một đầu vào theo cách tự giám sát, và học ra các bản đồ attention có thể diễn giải. Mỗi nhánh có Explainable Multi-Head Attention và Attribute-guided Explainer, cùng ba hàm mất mát: global attribute-guided, local attribute discriminability và attribute diversity. DINO là phương pháp tự giám sát đơn giản, tự chưng cất không cần nhãn. Các tác giả báo cáo attention map học được giữ được các vùng ngữ nghĩa của ảnh và có thể dùng để giải thích.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **Weakly supervised segmentation:** chỉ có nhãn cả ảnh (ví dụ "có mèo") nhưng muốn có phân đoạn từng vùng.
> - **Mạng Siamese:** hai nhánh giống nhau xử lý hai bản biến đổi của cùng một ảnh; hai nhánh phải cho kết quả nhất quán.
> - **Ba loss của eX-ViT (đọc theo tên):** loss toàn cục (attention tạo đặc trưng giải thích được); loss phân biệt cục bộ; loss đa dạng (các thuộc tính phải khác nhau). Bài báo chỉ nêu vai trò như vậy, không cần nhớ công thức.
> - **DINO:** "tự chưng cất" = mô hình tự học từ chính nó, không cần nhãn. Điều thú vị là chỉ nhờ vậy mà attention map đã bao quanh đúng vật thể, nên dùng được làm giải thích.

## Slide 15. Generic Attention-model, ATMAN, Concept-Transformer (1:10)

**ĐỌC:**
Generic Attention-model của Chefer là cách mới để giải thích dự đoán của các kiến trúc transformer, kể cả transformer hai phương thức và co-attention. Nó áp dụng cho ba kiến trúc phổ biến là self-attention thuần, self-attention kết hợp co-attention và encoder-decoder attention. Các tác giả thử trên hỏi đáp trực quan, nhưng cũng áp dụng được cho phát hiện đối tượng và phân đoạn. ATMAN là phương pháp nhiễu loạn không phụ thuộc phương thức, tận dụng cơ chế attention, tiết kiệm bộ nhớ, để tạo bản đồ liên quan của đầu vào với dự đoán đầu ra. Concept-Transformer khái quát attention từ đặc trưng thấp lên các khái niệm cấp cao do người dùng định nghĩa; giải thích là điểm attention trên các khái niệm này, bảo đảm cả tính hợp lý lẫn độ trung thực.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **Bi-modal** là mô hình nhận hai loại dữ liệu cùng lúc (ví dụ ảnh và câu hỏi). VQA = visual question answering = hỏi đáp về ảnh. Generic Attention-model mở rộng phương pháp "Beyond Attention" (người bạn đã trình bày) sang trường hợp có hai nguồn.
> - **ATMAN (bằng lời thường):** muốn biết một phần quan trọng không, ta làm yếu sự chú ý dành cho nó ngay *bên trong* attention và xem kết quả đầu ra thay đổi ra sao. "Modality-agnostic" = dùng được cho mọi loại dữ liệu (chữ, ảnh...).
> - **Concept-Transformer:** thay vì nói "pixel này quan trọng", mô hình nói "khái niệm này quan trọng" (ví dụ "có cánh" khi đoán chim). *Plausible* = hợp lý với con người; *faithful* = đúng là cái mô hình dùng.

## Slide 16. Tổng quan đánh giá giải thích (0:30)

**ĐỌC:**
Mục 4: đánh giá giải thích. Việc xem một phương pháp có mô tả đúng quá trình suy luận của mô hình hay không là một thách thức. Tài liệu đề xuất nhiều tiêu chí đánh giá, được tóm tắt ở Hình 5 của bài báo, gồm Deletion và Insertion, Effective Complexity, Faithfulness, Infidelity, IoU, Perturbation tests, Pointing Game, Segmentation tests, Sensitivity và Sensitivity-n. Em sẽ đi qua từng nhóm.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - Mười tiêu chí trong hình đúng là mười bong bóng của Hình 5 của bài.
> - **Cách nhớ cho dễ (cách nhớ của riêng bạn, không phải của bài):** có hai loại câu hỏi. Loại 1: giải thích có *đúng với mô hình* không (Deletion/Insertion, Perturbation, Faithfulness, Infidelity, Sensitivity-n). Loại 2: giải thích có *khớp với vị trí vật thể do con người đánh dấu* không (IoU, Pointing Game, Segmentation). Còn Sensitivity hỏi "có ổn định không" và Effective Complexity hỏi "có gọn không".
> - Ví dụ để hiểu khác biệt: mô hình đoán "tàu" nhờ có biển ở phía sau. Bản đồ tô biển trông "sai" với con người nhưng lại *trung thực* với mô hình.

## Slide 17. Deletion, Insertion và Perturbation tests (1:00)

**ĐỌC:**
Deletion và Insertion đánh giá độ trung thực của saliency map với mô hình. Deletion AUC đo điểm của lớp mục tiêu giảm nhanh thế nào khi các pixel quan trọng bị loại khỏi ảnh. Insertion AUC đo điểm đó tăng nhanh thế nào khi các pixel quan trọng được thêm vào một canvas trống. Deletion AUC nhỏ hơn và Insertion AUC lớn hơn nghĩa là giải thích trung thực hơn. Perturbation tests thì che dần các token theo thứ tự quan trọng mà giải thích đưa ra và đo độ chính xác bằng AUC. Positive là che từ quan trọng nhất, mong đợi hiệu năng giảm mạnh. Negative là che từ ít quan trọng nhất, giải thích tốt thì vẫn giữ được độ chính xác.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **Ý tưởng chung:** bản đồ nói "vùng này quan trọng nhất"; ta kiểm chứng bằng cách *xóa vùng đó* và xem mô hình có "mù" nhanh không. Xóa đúng chỗ quan trọng thì điểm của lớp phải rơi nhanh.
> - **AUC** là diện tích dưới đường cong (đường điểm theo lượng pixel bị xóa/chèn). Điểm rơi càng nhanh thì diện tích càng nhỏ, nên Deletion AUC nhỏ là tốt. Với Insertion thì ngược lại, điểm lên càng nhanh càng tốt, nên AUC lớn là tốt.
> - **Positive / Negative:** positive là xóa cái quan trọng trước, mong đợi độ chính xác tụt mạnh. Negative là xóa cái ít quan trọng trước, mong đợi độ chính xác ổn định.
> - **Thầy hỏi: "Vì sao Deletion nhỏ là tốt?"** Vì xóa đúng vùng mô hình dựa vào thì điểm lớp phải sụp nhanh, đường cong thấp, diện tích nhỏ.

## Slide 18. Effective Complexity, Faithfulness, (In)fidelity (1:00)

**ĐỌC:**
Effective Complexity đếm số lượng attribution vượt một ngưỡng. Giá trị thấp nghĩa là một số đặc trưng có thể bỏ qua vì ảnh hưởng thực sự nhỏ; giải thích đơn giản hơn nhưng có thể mất một phần độ chính xác thực tế. Faithfulness đánh giá chất lượng attribution mà không cần con người. Phương pháp loại bỏ các đặc trưng cụ thể rồi xem mối tương quan tuyến tính giữa logit dự đoán của mô hình và attribution của tập đặc trưng đó, trên nhiều lần chạy và nhiều mẫu, cho giá trị từ âm một đến một. (In)fidelity đo giải thích nắm bắt tốt đến đâu sự thay đổi của dự đoán khi đầu vào bị nhiễu loạn lớn. Mục tiêu là làm infidelity nhỏ nhất, và nó bổ sung cho sensitivity.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **Effective Complexity:** giải thích nên liệt kê *mọi* pixel hay chỉ *vài cái quan trọng*? Thước đo này đếm số đặc trưng "đáng kể". Đếm càng ít = giải thích càng gọn, dễ đọc. Đổi lại có thể bỏ sót ảnh hưởng nhỏ.
> - **Faithfulness (bằng lời thường):** nếu giải thích nói "nhóm này đóng góp nhiều" thì khi bỏ nhóm đó, kết quả của mô hình phải thay đổi nhiều. Ta kiểm tra "đóng góp mà giải thích nói" có đi cùng "mức thay đổi thật" không. Gần 1 = rất đi cùng nhau (tốt), gần 0 hoặc âm = không đáng tin.
> - **(In)fidelity:** kiểm tra "giải thích có đoán đúng mô hình sẽ thay đổi bao nhiêu khi ta làm nhiễu ảnh nặng không". Càng thấp càng tốt.
> - Ba tiêu chí này đều **không cần con người đánh dấu**.

## Slide 19. Sensitivity và Sensitivity-n (0:55)

**ĐỌC:**
Sensitivity đánh giá giải thích thay đổi thế nào khi đầu vào bị nhiễu loạn nhỏ. Khi các đầu vào giống nhau và đầu ra của mô hình gần nhau thì giải thích của chúng cũng phải giống nhau. Sensitivity-n được đề xuất để kiểm tra chính các giá trị attribution thay vì chỉ xét thứ hạng quan trọng. Nó chọn ngẫu nhiên các tập con gồm n đặc trưng và tính tương quan giữa mức thay đổi dự đoán của mô hình khi gỡ các đặc trưng đó với tổng điểm attribution của chúng, rồi lấy trung bình. Điểm Sensitivity-n cao cho thấy attribution chính xác và tương quan tốt với tác động lên dự đoán.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **Sensitivity:** chụp hai ảnh gần như giống nhau, mô hình đoán cùng một lớp. Nếu bản đồ giải thích đổi hoàn toàn thì giải thích đó *mong manh*, không đáng tin.
> - **Sensitivity-n:** giống Faithfulness nhưng chỉ xét các tập con có đúng n đặc trưng. Nó kiểm tra cả *độ lớn* của attribution, không chỉ xếp hạng "cái nào hơn cái nào".
> - Sensitivity cũng là thước đo bổ sung của infidelity: infidelity xét thay đổi *lớn* của đầu vào; sensitivity xét thay đổi *nhỏ*.

## Slide 20. IoU, Pointing Game, Segmentation tests (1:10)

**ĐỌC:**
Nhóm này so giải thích với chú thích của con người. IoU test đo bản đồ giải thích chồng khớp thế nào với hộp bao ground-truth. Trước hết phóng bản đồ về kích thước ảnh gốc, áp ngưỡng 0,2 để loại nền, rồi lấy giao của hộp dự đoán và hộp ground-truth chia cho hợp của chúng; IoU càng cao, định vị càng tốt. Pointing Game: nếu pixel có saliency cao nhất nằm trong hộp do con người chú thích thì tính một hit, không thì là miss; độ chính xác bằng số hit chia tổng số hit và miss. Grid Pointing Game dùng lưới ảnh tổng hợp nhiều đối tượng và đo phần attribution dương rơi đúng ô con. Segmentation tests xem bản đồ như một phân đoạn mềm và đánh giá bằng pixel accuracy, mIoU, mAP và mF1.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **IoU = "giao chia hợp":** hai hộp chồng nhau càng nhiều thì IoU càng gần 1; không chồng thì bằng 0. Dùng để chấm xem vùng sáng của bản đồ có trùng với vật thể không.
> - **Pointing Game:** đơn giản, chỉ xét *một điểm* sáng nhất có nằm trong hộp không. Ví dụ 7 hit trên 10 ảnh thì 70%. Đây là công thức duy nhất trong bài ở phần này (phương trình 12).
> - **Grid Pointing Game:** ghép nhiều ảnh thành lưới, giải thích cho một lớp, xem phần lớn "độ sáng" có rơi vào ô ảnh của lớp đó không.
> - **Segmentation tests:** coi bản đồ là một bản phân đoạn và chấm bằng bốn số (pixel accuracy, mIoU, mAP, mF1). Hai số đầu dùng một ngưỡng (trung bình attribution); hai số sau lấy trung bình trên nhiều ngưỡng.
> - **Lưu ý quan trọng:** các tiêu chí này cần ground-truth do con người tạo nên đo *plausibility* (hợp lý với con người), chưa chắc là faithfulness. Vì vậy phải dùng cả hai loại.

## Slide 21. Tools và frameworks (1:00)

**ĐỌC:**
Mục 5: các công cụ. Captum là toolkit mã nguồn mở cho PyTorch, có nhiều thuật toán attribution dựa trên gradient và nhiễu loạn ở mức đặc trưng, neuron và lớp, kèm các độ đo đánh giá, hỗ trợ ảnh, văn bản, âm thanh và video. InterpretDL dành cho PaddlePaddle, có LIME, Grad-CAM, Integrated Gradients cho CNN, transformer và ViT, và là benchmark để so sánh thuật toán mới. AttentionViz là công cụ trực quan hóa attention của cả transformer ngôn ngữ và thị giác, cho cái nhìn toàn cục về tương tác query và key trên nhiều đầu vào. Quantus là bộ công cụ Python mã nguồn mở với nhiều độ đo và hướng dẫn để đánh giá giải thích.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **Nhớ nhanh:** Captum = *tạo* giải thích (PyTorch). InterpretDL = giống Captum nhưng cho PaddlePaddle (nền tảng học sâu của Trung Quốc). AttentionViz = *nhìn* attention. Quantus = *đo* chất lượng giải thích.
> - Có thể nối với bài demo của nhóm (`demo_slides.tex`): bạn đã dùng Captum và Quantus, đúng hai công cụ này.

## Slide 22. Datasets (0:30)

**ĐỌC:**
Mục 6: các dataset trong các công trình đã nêu. CIFAR-10 có sáu mươi nghìn ảnh màu nhỏ, mười lớp. ImageNet với hàng triệu ảnh là nguồn chính để huấn luyện và đánh giá các tác vụ thị giác. COVID-QU-Ex là dữ liệu y tế gồm hơn ba mươi ba nghìn ảnh X-quang ngực. ImageNet-Segmentation dùng cho thí nghiệm phân đoạn, và Kinetics-400 được IA-RED² dùng cho phân loại video.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - Slide ngắn, chỉ cần điểm danh. COVID-QU-Ex gồm COVID-19, ca khỏe mạnh và các nhiễm trùng khác như viêm phổi.
> - Nếu thiếu giờ, đọc lướt slide này.

## Slide 23. Thảo luận và hướng phát triển (1:05)

**ĐỌC:**
Mục 7 có năm điểm chính. Một, với attribution, phần lớn phương pháp model-agnostic dùng được cho transformer, nhưng tận dụng attention có thể giúp thêm; feature attribution được quan tâm nhiều hơn feature visualization, trong khi hướng sau giúp hiểu cách mô hình hoạt động toàn cục. Hai, dùng trực tiếp attention weight còn thiếu sót trong thị giác vì nó chỉ là một phần của chuỗi phép biến đổi phi tuyến; cần thảo luận tính phù hợp của attention, cách kết hợp các head, và thử các phương pháp toán học khác thay cho tổ hợp tuyến tính. Ba, với pruning cần chú ý hơn đến explainability; ngay cả nhóm explicit cũng chưa được đánh giá bằng tiêu chí định lượng. Bốn, benchmark chủ yếu là phân loại; giải thích cho mô hình đa phương thức và mô hình sinh còn rất mới. Năm, còn thiếu nghiên cứu dùng giải thích để gỡ lỗi, cải thiện mô hình và tăng độ công bằng, tin cậy cho ViT.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - **Feature attribution vs feature visualization:** attribution trả lời "pixel nào quan trọng với dự đoán này" (cục bộ). Visualization trả lời "mô hình học ra *cái gì*" (toàn cục). Bài nói hướng thứ hai đang bị bỏ ngỏ.
> - **"Attention không phải là giải thích":** bài nhắc Jain & Wallace 2019. Attention chỉ là một khâu trong chuỗi nhiều bước, nên bản thân nó chưa chắc cho biết mô hình dựa vào đâu.
> - **Ý 3 là phần của bạn:** liên hệ ngay với slide 10: pruning đo hiệu quả kỹ nhưng chưa đo giải thích.
> - **Nếu thầy hỏi hướng nào hứa hẹn nhất:** chọn ý 3 và ý 5, vì khả thi (dùng các tiêu chí mục 4 để đo cho các mô hình pruning).

## Slide 24. Kết luận (0:45)

**ĐỌC:**
Tóm lại, bài báo tổng quan các kỹ thuật explainability cho vision transformer, chia thành năm nhóm theo động cơ, cấu trúc và kịch bản ứng dụng, đồng thời trình bày các tiêu chí đánh giá, công cụ và hướng nghiên cứu. Trong phần của em: pruning giúp hiệu quả và có thể giúp giải thích nhưng ít được đánh giá về explainability; các phương pháp inherent xây dựng giải thích ngay trong mô hình với cái giá về độ chính xác; giải thích cho các tác vụ khác ngoài phân loại mới bắt đầu; và giải thích cần được đánh giá bằng nhiều tiêu chí. Em xin cảm ơn cô và các bạn đã lắng nghe.

> **GIẢI THÍCH CHO BẠN (không đọc):**
>
> - Bốn ý chốt đúng là bốn gạch đầu dòng của slide. Nếu chỉ nhớ được một câu: "Pruning giúp giải thích nhưng chưa ai đo giải thích của nó".
> - Nếu còn dư vài chục giây, nhắc lại ví dụ EViT (giữ token được chú ý nhiều, gộp phần còn lại) vì dễ hiểu nhất.

---

# Câu hỏi có thể gặp và gợi ý trả lời

1. **Vì sao pruning lại liên quan đến giải thích?** Pruning bỏ phần không cần; phần còn lại cho biết mô hình cần gì. Nhưng bài báo lưu ý phải đánh giá bằng tiêu chí ở mục 4 mới chắc.
2. **EViT khác DynamicViT thế nào?** DynamicViT học thêm một module nhỏ để quyết định token nào giữ. EViT dùng sẵn mức chú ý của class token và gộp token ít quan trọng thành một token.
3. **Faithfulness khác plausibility thế nào?** Faithfulness: đúng với mô hình. Plausibility: hợp lý với con người. Mô hình có thể dựa vào nền, khi đó giải thích đúng nhưng trông "sai" với con người.
4. **Vì sao Deletion AUC nhỏ là tốt?** Xóa đúng vùng quan trọng thì điểm lớp rơi nhanh nên diện tích dưới đường cong nhỏ.
5. **Bài báo có so sánh explainability giữa các phương pháp pruning không?** Không. Bảng 2 chỉ có GFLOPs, Top-1 và throughput. Đây là khoảng trống mà bài báo nêu.
6. **Phương pháp inherently explainable có nhược điểm gì?** Thường khó đạt độ chính xác bằng hộp đen, nên cần cân bằng giữa giải thích và hiệu năng (Rudin 2019).
7. **Bài nói gì về attention weight?** Dùng trực tiếp thì chưa đủ, vì nó chỉ là một phần của chuỗi các phép biến đổi phi tuyến; cần thảo luận thêm về tính phù hợp của nó cho giải thích.
