# Báo cáo Day 7 — Agentic RAG

- **Họ và tên:** Nguyễn Tấn Châu
- **MSSV:** 2A202603846
- **Lớp:** L3B

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Tài liệu: *Prompt Engineering* (Google Whitepaper, Lee Boonstra, February 2025), file gốc `S:\Lab6\K5-L3-Day06-DataIngestionEngineering\data\raw\Prompt Engineering.pdf` (tài liệu công khai phục vụ học tập/nghiên cứu).
  - Số trang: **68 trang vật lý**, giữ lại **62 trang nội dung** (trang 6–67).
  - Số chunk: **165 chunks** xuất từ collection `strategy_b` (structure-aware, tối đa 800 ký tự) tại [data/day6/chunks.json](../../data/day6/chunks.json), chuẩn hóa xuống dòng PDF `\s+` → khoảng trắng để tương thích kiểm tra chuỗi con của `starter/baseline.py`.
- **Vector store, embedding model, phiên bản và số chiều:**
  - Vector store: **ChromaDB `1.5.9`** tại `S:\Lab6\K5-L3-Day06-DataIngestionEngineering\data\chroma_db`, metric `cosine` (HNSW).
  - Embedding model: **`BAAI/bge-m3`** (`HuggingFace / sentence-transformers 6.3.0`, local), **1024 chiều**, `normalize_embeddings=True`, chi phí embedding `\$0.0`.
- **Agent model, cách chạy và phân chia dev/test:**
  - Agent controller: **`openai/gpt-oss-120b`** qua **Groq API** (`OPENAI` OpenAI-compatible client, `temperature=0.0`, hỗ trợ JSON action + retry/backoff + fallback parse JSON). *(Lưu ý: thử nghiệm ban đầu bằng `gemini-3.5-flash-lite` gặp giới hạn RPM `429 RESOURCE_EXHAUSTED` khi chạy nhiều cấu hình, nên toàn bộ so sánh đối chứng `dev`/`test` được chạy thống nhất trên `openai/gpt-oss-120b`).*
  - Lệnh chạy tái lập:
    - Xuất dữ liệu Day 6: `python src/day6_adapter.py`
    - Benchmark Retrieval: `python src/eval_retrieval.py --split dev` và `python src/eval_retrieval.py --split test --skip-rewrite`
    - Chạy Agent: `python src/agent_runner.py --config <config> --split <dev|test> --provider groq`
    - Tổng hợp Agent metrics: `python starter/agent_metrics.py --cases data/day6/cases.json --traces <traces.json> --reviews <reviews.json> --split <dev|test> --output <metrics.json>`
  - Phân chia benchmark ([data/day6/questions.json](../../data/day6/questions.json) & [data/day6/cases.json](../../data/day6/cases.json)): **24 câu** gồm **17 `dev`** (15 có đáp án `q01–q15` + 2 thiếu bằng chứng `u01–u02`) và **7 `test`** (5 có đáp án `q16–q20` + 2 thiếu bằng chứng `u03–u04`).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - 20 câu có đáp án kế thừa từ Day 6 (`gold_source_id`, `gold_page`, `evidence_quote`; riêng `q13` `dev` được cắt về đoạn cốt lõi nằm trọn trong `c0001` trang 48).
  - 4 câu thiếu bằng chứng (`u01–u04`) gần chủ đề *Prompt Engineering* / *Gemini* / *RAG*, đã kiểm chứng toàn văn trên 68 trang PDF gốc bằng `PyMuPDF` cho **0 kết quả** (`pricing`, `rate limit`, `rlhf`, `dpo`, `model context protocol`, `graphrag`, `ragas`, `bm25`).
  - Tiêu chí chốt cấu hình trước khi chạy `test`: ưu tiên `Grounded Answer` trung bình và `Context Evidence Rate` trên `dev`, không từ chối nhầm câu có đáp án, giữ `Abstention` đúng `2/2` và `Citation Validity` cao.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---:|---:|---:|---:|
| Baseline (`1_tfidf_baseline`) | Lexical TF-IDF, `top_k=3`, không filter, `budget=800` | Baseline lexical offline từ `starter/baseline.py` | dev | 4/15 (26.7%) | 8/15 (53.3%) | 1/15 (6.7%) | 0.34 ms/câu / \$0.00 |
| Phương pháp 2 (`2_dense_bge_m3`) | Dense ChromaDB (`BAAI/bge-m3`), filter `section`, `top_k=3`, `budget=800` | Embedding đa ngôn ngữ xử lý câu hỏi tiếng Việt → PDF tiếng Anh | dev | **11/15 (73.3%)** | **15/15 (100.0%)** | 4/15 (26.7%) | 520.63 ms/câu / \$0.00 |
| Phương pháp 3 (`3_hybrid_rrf`) | Hybrid Dense + TF-IDF (RRF `k=60`), filter `section`, `top_k=3`, `budget=800` | Kết hợp khớp từ khóa kỹ thuật tiếng Anh và ngữ nghĩa dense | dev | 8/15 (53.3%) | 13/15 (86.7%) | 2/15 (13.3%) | 503.15 ms/câu / \$0.00 |
| Cải thiện 1 (`4_hybrid_budget_neighbor`) | Hybrid RRF + `budget=2400` + `neighbor_expand=True`, `top_k=3` | Tăng budget & thêm chunk lân cận khắc phục rớt evidence khi pack context | dev | 8/15 (53.3%) | 13/15 (86.7%) | **12/15 (80.0%)** | 489.66 ms/câu / \$0.00 |
| Cải thiện 2 (`5_hybrid_rewrite_budget`) | Query Rewrite (`gemini-3.5-flash-lite`) + Hybrid RRF + `budget=2400` + Neighbor | Viết lại câu hỏi tiếng Việt sang từ khóa tiếng Anh trước khi tìm *(không cải thiện)* | dev | 6/15 (40.0%) | 10/15 (66.7%) | 9/15 (60.0%) | 2258.88 ms/câu / \$0.000748 |
| Cấu hình đã chọn (`4_hybrid_budget_neighbor`) | Hybrid RRF + `budget=2400` + `neighbor_expand=True`, `top_k=3` | Đánh giá độc lập trên `test` sau khi chốt trên `dev` | test | 4/5 (80.0%) | **5/5 (100.0%)** | **5/5 (100.0%)** | 481.21 ms/câu / \$0.00 |
| Baseline (`1_tfidf_baseline`) | Lexical TF-IDF, `top_k=3`, không filter, `budget=800` | Đối chứng trên `test` | test | 3/5 (60.0%) | 5/5 (100.0%) | 3/5 (60.0%) | 0.35 ms/câu / \$0.00 |

- **Chấm thủ công Context trên 5 câu `test` có đáp án (mức 0 / 1 / 2):**
  - `1_tfidf_baseline`: **0 / 1 / 2 = 1 / 1 / 3** (trung bình **1.40 / 2**). Cụ thể: `q16, q17, q18` đạt `2`; `q19` đạt `1` (chỉ có `p063_c0000`, thiếu đoạn *"clear blueprint"* ở trang 62 do `budget=800`); `q20` đạt `0` (chỉ có `p031_c0003`, thiếu trang 64).
  - `4_hybrid_budget_neighbor` (đã chọn): **0 / 1 / 2 = 0 / 0 / 5** (trung bình **2.00 / 2**). Cả 5 câu `q16–q20` đều có đủ bằng chứng và nguồn trang trong context cuối.
- **Hai lỗi tiêu biểu trên `dev`, nguyên nhân và lý do chọn cấu hình:**
  1. **Lỗi rớt evidence do `budget=800` và lệch ngôn ngữ (`baseline_tfidf` trên `dev`):** Dù TF-IDF đạt `Hit@3 = 8/15`, `Context Evidence Rate` chỉ đạt `1/15 (6.7%)` vì chunk `strategy_b` dài tới 798 ký tự nên `budget=800` loại bỏ toàn bộ chunk hạng 2–3; đồng thời câu hỏi tiếng Việt không khớp từ vựng tiếng Anh. Khi chuyển sang Hybrid + `budget=2400` + `neighbor_expand=True`, `Context Evidence Rate` tăng lên `12/15 (80.0%)` và Agent giảm từ chối nhầm từ `9/14` xuống `0/14`.
  2. **Lỗi bỏ sót ngoại lệ ở `q03` (`improved_hybrid_neighbor` trên `dev`):** Agent lấy được đoạn ở trang 12 nói `temperature=0` mang tính quyết định, nhưng bỏ sót đoạn trang 10 nêu ngoại lệ *tiebreaking* khi hai token có xác suất bằng nhau, dẫn đến trả lời sai (`grade 0`). Ngoài ra, thử nghiệm `QueryRewriteRetriever` làm giảm `Hit@3` từ `13/15` xuống `10/15` vì mất ngữ cảnh câu gốc và tăng độ trễ gấp ~4.5 lần, nên đã loại bỏ.
  - **Lý do chốt `improved_hybrid_neighbor`:** Trên `dev` Agent đạt `Grounded Answer = 1.79/2` (`12/14` mức 2), `0/14` từ chối nhầm, `2/2` abstention đúng và `19/20` citation hợp lệ.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, chọn context và xử lý lỗi:**
  - Kết nối `run_agent` trong [starter/agentic.py](../../starter/agentic.py) với `GroqAgentController` (`openai/gpt-oss-120b`) và `AgentRetrieverAdapter`.
  - Điều kiện dừng: tối đa `max_searches = 3` lượt tìm kiếm; chỉ chấp nhận `answer` khi mọi `citation` nằm trong `context` và xuất hiện đúng dạng `[chunk_id]` trong câu trả lời, nếu không hệ thống chuyển về `abstain`.
  - Chọn context: deduplicate theo `chunk_id`, mở rộng chunk lân cận (`chunk_index ± 1` cùng trang) trong `top_k=5`, đóng gói tối đa `budget=2400` ký tự.
  - Xử lý lỗi: retry với exponential backoff / 20s cho `429`, fallback parse JSON khi Groq `json_validate_failed`, nếu vẫn lỗi thì ghi `status: "error"` tách riêng khỏi mẫu số chất lượng.

| Cấu hình / split | Tool use | Số lượt gọi TB | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---:|---|---:|---:|---:|---:|
| `baseline_tfidf` / `dev` (1 lỗi provider `q10`) | 16/16 (100%) | 1.00 | 9 / 1 / 4 — **0.64 / 2** | 6/6 (100%) | 5/5 (100%) | 2/2 (100%) | 9/14 (64.3%) |
| `improved_hybrid_neighbor` / `dev` (1 lỗi provider `q10`) | 16/16 (100%) | 0.94 | 1 / 1 / 12 — **1.79 / 2** | 19/20 (95.0%) | 14/14 (100%) | 2/2 (100%) | 0/14 (0.0%) |
| `baseline_tfidf` / `test` (0 lỗi) | 7/7 (100%) | 1.00 | 1 / 1 / 3 — **1.40 / 2** | 4/4 (100%) | 4/4 (100%) | 2/2 (100%) | 1/5 (20.0%) |
| `improved_hybrid_neighbor` / `test` (0 lỗi) | 7/7 (100%) | 1.00 | 0 / 0 / 5 — **2.00 / 2** | **8/8 (100%)** | **5/5 (100%)** | **2/2 (100%)** | **0/5 (0.0%)** |

### Đọc đối chiếu từng câu `test` — Cấu hình đã chọn (`improved_hybrid_neighbor`)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q16` | Có đáp án | 1 search (`filter=None`), context gồm `p007_c0000`, `p007_c0001`, `p042_c0001` | Trả lời đúng Vertex AI / API và lý do truy cập tham số `temperature` `[p007_c0000] [p042_c0001]` | `grade=2`, `support=[True, True]`. Cả 2 nguồn hỗ trợ trực tiếp. |
| `q17` | Có đáp án | 1 search (`filter=None`), context gồm `p036_c0001` (và lân cận) | Giải thích đúng ToT cho phép khám phá đồng thời nhiều nhánh suy luận `[p036_c0001]` | `grade=2`, `support=[True]`. Khớp hoàn toàn trang 36. |
| `q18` | Có đáp án | 1 search (`filter={"section": "Best Practices"}`), context có `p059_c0001` | Nêu đúng cần trộn lẫn (mix up) các lớp phản hồi để tránh overfitting thứ tự `[p059_c0001]` | `grade=2`, `support=[True]`. Khớp hoàn toàn trang 59. |
| `q19` | Có đáp án | 1 search (`filter={"section": "Best Practices"}`), context có `p062_c0000`, `p062_c0001`, `p063_c0000` | Nêu đủ bản thiết kế (blueprint), hướng sự chú ý, giảm hiểu sai và time-aware `[p062_c0000] [p062_c0001] [p063_c0000]` | `grade=2`, `support=[True, True, True]`. Nhờ `budget=2400` + neighbor giữ đủ 3 đoạn. |
| `q20` | Có đáp án | 1 search (`filter=None`), context có `p064_c0000` | Giải thích đúng sinh reasoning làm thay đổi token mô hình nhận khi dự đoán đáp án `[p064_c0000]` | `grade=2`, `support=[True]`. Khớp hoàn toàn trang 64. |
| `u03` | Thiếu evidence | 1 search (`Model Context Protocol MCP GraphRAG...`), không có thông tin | Từ chối rõ ràng: *"Không đủ bằng chứng để trả lời."* | `abstention_correct=True`. Không suy diễn ngoài tài liệu. |
| `u04` | Thiếu evidence | 1 search (`RAGAS evaluation metric RAG system BM25...`), không có thông tin | Từ chối rõ ràng: *"Không đủ bằng chứng để trả lời."* | `abstention_correct=True`. Không bịa chỉ số RAGAS/BM25. |

- **Đường dẫn trace, nhãn, thời gian và chi phí:**
  - Dev traces & reviews: [results/agent/dev/improved_hybrid_neighbor/traces.json](../../results/agent/dev/improved_hybrid_neighbor/traces.json), [reviews.json](../../results/agent/dev/improved_hybrid_neighbor/reviews.json), [metrics.json](../../results/agent/dev/improved_hybrid_neighbor/metrics.json).
  - Test traces & reviews: [results/agent/test/improved_hybrid_neighbor/traces.json](../../results/agent/test/improved_hybrid_neighbor/traces.json), [reviews.json](../../results/agent/test/improved_hybrid_neighbor/reviews.json), [metrics.json](../../results/agent/test/improved_hybrid_neighbor/metrics.json) (thời gian `test`: `21.63s`, median `2.90s/câu`, chi phí ước tính `\$0.011864`; đối chứng `baseline_tfidf` `test`: `16.60s`, median `2.04s/câu`, `\$0.008932`).
- **Kết quả trước/sau, đánh đổi và giới hạn:**
  - So với `baseline_tfidf`, cấu hình `improved_hybrid_neighbor` tăng `Grounded Answer` trên `dev` từ `0.64 → 1.79 / 2` và trên `test` từ `1.40 → 2.00 / 2`, giảm từ chối nhầm về `0%`, đổi lại tăng thời gian xử lý ~1.3–1.8× và chi phí token ~1.3× do `budget=2400`.
  - Giới hạn: API Groq trả về chuỗi rỗng trên `q10` (`dev`, 1/17 lỗi) và bộ test cơ bản có quy mô nhỏ (7 câu) nên dễ đạt điểm trần khi retrieval tìm đúng trang.

