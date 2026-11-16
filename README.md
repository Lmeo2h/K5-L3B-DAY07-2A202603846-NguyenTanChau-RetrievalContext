# Day 7 — Agentic RAG

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.

## Hướng dẫn chạy & tái lập (Nguyễn Tấn Châu — 2A202603846)

1. **Cấu hình môi trường:** Tạo file `.env` ở thư mục gốc với `GROQ_API_KEY=<your-key>` (hoặc `GEMINI_API_KEY`). Vector store Day 6 nằm tại `S:\Lab6\K5-L3-Day06-DataIngestionEngineering\data\chroma_db` (collection `strategy_b`, embedding `BAAI/bge-m3`).
2. **Chạy Unit Test (offline):**
   ```bash
   python -m unittest discover -s tests -v
   ```
3. **Xuất dữ liệu & chuẩn hóa benchmark từ Day 6 (24 câu: 17 dev / 7 test):**
   ```bash
   python src/day6_adapter.py
   ```
4. **Đánh giá Retrieval trên `dev` và `test`:**
   ```bash
   python src/eval_retrieval.py --split dev
   python src/eval_retrieval.py --split test --skip-rewrite
   ```
5. **Chạy Agent thật (`openai/gpt-oss-120b` qua Groq) & tổng hợp metrics:**
   ```bash
   python src/agent_runner.py --config baseline_tfidf --split dev --provider groq
   python src/agent_runner.py --config improved_hybrid_neighbor --split dev --provider groq
   python src/agent_runner.py --config baseline_tfidf --split test --provider groq
   python src/agent_runner.py --config improved_hybrid_neighbor --split test --provider groq
   python starter/agent_metrics.py --cases data/day6/cases.json --traces results/agent/test/improved_hybrid_neighbor/traces.json --reviews results/agent/test/improved_hybrid_neighbor/reviews.json --split test --output results/agent/test/improved_hybrid_neighbor/metrics.json
   ```
6. **Báo cáo cá nhân:** Xem [report/individual/2A202603846_NguyenTanChau.md](report/individual/2A202603846_NguyenTanChau.md).
