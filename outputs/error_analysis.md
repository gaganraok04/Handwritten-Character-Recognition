# Error Analysis: Top 20 Worst Predictions

Below is an inspection of the 20 test instances where the trained Deep CNN had the highest confidence in an incorrect prediction.

| Rank | True Label | Predicted Label | Model Confidence | Why It Failed (Visual Inspection) |
|---|---|---|---|---|
| #1 | **6** | **4** | 99.84% | Atypical stroke pattern / ambiguous writing style. |
| #2 | **8** | **1** | 99.83% | Atypical stroke pattern / ambiguous writing style. |
| #3 | **7** | **2** | 99.80% | Slanted top stroke or loop resembles the head of a 2/7. |
| #4 | **8** | **0** | 99.73% | Atypical stroke pattern / ambiguous writing style. |
| #5 | **6** | **1** | 99.64% | Atypical stroke pattern / ambiguous writing style. |
| #6 | **0** | **7** | 99.59% | Atypical stroke pattern / ambiguous writing style. |
| #7 | **9** | **7** | 99.55% | Atypical stroke pattern / ambiguous writing style. |
| #8 | **5** | **3** | 99.29% | Top horizontal bar curves into the middle junction, resembling 3 or 5. |
| #9 | **6** | **0** | 99.18% | Loop closed weakly or extended top stem mimics 0 or 6. |
| #10 | **5** | **3** | 98.84% | Top horizontal bar curves into the middle junction, resembling 3 or 5. |
| #11 | **5** | **6** | 98.70% | Bottom round loop and upper hook resemble 5 or 6. |
| #12 | **9** | **7** | 98.06% | Atypical stroke pattern / ambiguous writing style. |
| #13 | **4** | **9** | 97.80% | Top loop or vertical stem ambiguous between open-top 4 and closed 9. |
| #14 | **9** | **0** | 96.95% | Atypical stroke pattern / ambiguous writing style. |
| #15 | **6** | **0** | 96.77% | Loop closed weakly or extended top stem mimics 0 or 6. |
| #16 | **2** | **0** | 96.72% | Atypical stroke pattern / ambiguous writing style. |
| #17 | **2** | **7** | 96.70% | Slanted top stroke or loop resembles the head of a 2/7. |
| #18 | **8** | **2** | 96.39% | Atypical stroke pattern / ambiguous writing style. |
| #19 | **2** | **7** | 96.03% | Slanted top stroke or loop resembles the head of a 2/7. |
| #20 | **6** | **5** | 95.83% | Bottom round loop and upper hook resemble 5 or 6. |