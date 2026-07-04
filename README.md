# Dark Pattern Recognition

This project uses machine learning to detect deceptive dark-pattern language in e-commerce website text. It compares text classification approaches and documents the ethical tradeoffs of using AI to flag potentially manipulative design.

**GitHub Page:** https://radrebelsam.github.io/ai4all-20c-darkpatterns/

## Research Question

Can machine learning models detect dark-pattern language in e-commerce website text, and which categories of dark patterns are hardest to classify?

## Project Evolution

The project began with a broad survey of multiple dark-pattern datasets from Kaggle. Early attempts to combine several category-labeled datasets into one all-purpose model ran into inconsistent label schemas and imbalanced class distributions. After comparing label quality and coverage, we split the modeling into two parts: a balanced binary detector trained on the Krish Uppal dataset, and a separate second-stage category model trained on compatible category-labeled dark-pattern examples.

The modeling pipeline went through three main iterations. The first used a simple bag-of-words (CountVectorizer) approach and showed that even basic text features could achieve over 85% accuracy. The second switched to TF-IDF with unigram and bigram features, which improved both precision and F1 score by giving more weight to distinctive terms rather than raw counts. The third added a second trained model for category/type recognition so the app can explain whether suspicious text looks more like urgency, scarcity, social proof, or another dark-pattern type.

## Design Decisions and Reasoning

This section documents the main project decisions and why we made them.

| Decision | What we did | Reasoning |
| --- | --- | --- |
| Topic selection | Focused on dark-pattern recognition instead of AI job replacement. | Dark-pattern datasets were more directly usable for text classification and allowed a concrete demo with website text. |
| Dataset strategy | Used the Krish Uppal balanced dataset as the primary baseline; used Devitachi and Akash Nath as expanded/secondary sources. | The primary dataset has a clean 50/50 binary label split. The other datasets add source coverage, but they have different schemas, overlap, and class imbalance, so they are better for robustness checks than the default app model. |
| Common schema | Normalized different datasets into shared fields such as `text`, `label`, `category`, and `source`. | Each dataset has different columns. A common schema lets the same training and evaluation code work across sources. |
| Train/test split | Used an 80/20 random stratified split with `random_state=42`. | Random splitting avoids row-order bias. Stratification keeps the dark-pattern/not-dark-pattern ratio similar in train and test sets, making the evaluation fairer. |
| Feature method | Used `TF-IDF` with unigrams and bigrams. | TF-IDF is not the classifier itself; it turns text into numeric features. Bigrams help capture phrases such as `limited time` and `only left`. |
| Model comparison | Tested Logistic Regression, Linear SVM, Naive Bayes, Decision Tree, and Random Forest. | Comparing several models gives a stronger baseline than using only one model. Logistic Regression had the best primary F1 score and gives confidence scores for the app/scanner. |
| Demo approach | Built both a Streamlit app and a Playwright webpage scanner. | Streamlit is easy to explain for manual examples. Playwright makes the demo feel closer to real webpage scanning by collecting visible page text automatically. |
| Confidence threshold | The Playwright scanner shows dark-pattern results above a default 65% confidence threshold. | This hides weaker predictions during the demo and keeps attention on higher-confidence examples. Confidence is model certainty, not a guarantee that the prediction is correct. |
| Discount false-positive filter | Suppressed plain price/discount snippets unless they include pressure language. | Real sale pages often include harmless text like `Previous price: $99.00 47% off`. The filter reduces obvious false positives while still allowing pressure phrases like `Hurry, 47% off ends soon`. |
| Product-title false-positive filter | Suppressed catalog-style product titles/spec snippets unless they include pressure language. | Live tests showed false positives on product titles with model numbers, storage sizes, camera specs, or labels like `US Stock`. These are usually catalog metadata, not manipulative language. |
| Second-stage category model | Trained a separate category classifier after the main binary detector. | The main model answers whether text looks suspicious. The category model adds a likely type such as `Urgency`, `Scarcity`, or `Social Proof`, but it is limited by category imbalance and should be presented as supporting detail rather than a final category judgment. |
| Sentiment/context decision | Documented sentiment and UI context as future work instead of adding them now. | Sentiment can help with shame/fear/guilt language, but dark patterns depend heavily on button roles, popup placement, timing, and user flow. Adding this now would require new labeled data and a more complex pipeline. |
| Production limitation | Documented OCR, screenshot, color/layout, DOM/CSS, and UX-flow detection as future production layers. | Even with Playwright, this project is text-first. A real production detector would need additional evidence for image text, visual hierarchy, hidden links, checkout friction, cancellation difficulty, and other non-text patterns. |

## Two-Model Pipeline

The current project uses **two trained machine learning models**:

1. **Model 1: Binary dark-pattern detector**
   - Input: a piece of website or e-commerce text.
   - Output: `Dark Pattern` or `Not Dark Pattern`.
   - Best model: `TF-IDF + Logistic Regression`.
   - Purpose: decide whether the text looks suspicious at all.

2. **Model 2: Dark-pattern type/category classifier**
   - Input: text that was flagged as suspicious by Model 1.
   - Output: a likely type such as `Urgency`, `Scarcity`, `Social Proof`, `Misdirection`, `Obstruction`, `Sneaking`, or `Forced Action`.
   - Best model: `TF-IDF + Linear SVM`.
   - Purpose: make the result easier to explain by showing what kind of dark pattern the text may be.

Both models use TF-IDF text features with unigrams and bigrams. TF-IDF converts each snippet into a numeric vector where distinctive words and phrases receive higher weight. Bigrams help capture phrases such as `limited time` and `only left`.

The second model should be treated as an explanation aid, not a final label. It is trained only on dark-pattern examples, and some categories have much less data than others. In the current category dataset, common classes such as `Scarcity` and `Social Proof` have hundreds of examples, while `Sneaking` has 22 examples and `Obstruction` has 55 examples after deduplication. This means the likely-type output is useful for interpretation, but rare-category predictions need extra human review.

## Model 1: Binary Detection Results

All five models were evaluated on the same 20% held-out test set (stratified split, `random_state=42`).

| Model | Accuracy | Precision | Recall | F1 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| **Logistic Regression** | **0.939** | **0.973** | **0.903** | **0.936** | Best overall; interpretable coefficients |
| Linear SVM | 0.932 | 0.959 | 0.903 | 0.930 | Close second; no probability output |
| Naive Bayes | 0.928 | 0.928 | 0.928 | 0.928 | Fast; assumes feature independence |
| Random Forest | 0.928 | 0.986 | 0.869 | 0.923 | High precision but lower recall |
| Decision Tree | 0.917 | 0.980 | 0.852 | 0.912 | Prone to overfitting on text features |

**Why Logistic Regression for Model 1?** It achieves the highest binary F1 score, produces probability estimates for confidence scores in the app/scanner, and its coefficients can be inspected to verify that the model is learning meaningful text features.

**Cross-validation for stability.** Because the table above reflects a single 20% held-out split, we also ran 5-fold stratified cross-validation on the full primary dataset (`python scripts/cross_validate.py`, saved to `reports/cv_results.csv`). The mean ± standard deviation F1 scores confirm the models are stable rather than lucky on one split:

| Model | Accuracy (mean ± std) | F1 (mean ± std) |
| --- | ---: | ---: |
| Linear SVM | 0.944 ± 0.010 | 0.943 ± 0.011 |
| Logistic Regression | 0.941 ± 0.013 | 0.939 ± 0.014 |
| Random Forest | 0.936 ± 0.012 | 0.933 ± 0.014 |
| Decision Tree | 0.923 ± 0.014 | 0.919 ± 0.016 |
| Naive Bayes | 0.879 ± 0.053 | 0.889 ± 0.045 |

Logistic Regression and Linear SVM are within one standard deviation of each other; Logistic Regression remains the app default because it also exposes calibrated probability scores.

## Model 2: Category Classification Results

The second-stage model is trained only on examples that are already labeled as dark patterns. It predicts the likely category/type after the binary model has flagged text as suspicious.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| **Linear SVM** | **0.920** | **0.927** | **0.870** | **0.894** | Best category model |
| Random Forest | 0.893 | 0.930 | 0.777 | 0.816 | High precision, lower recall on smaller classes |
| Logistic Regression | 0.906 | 0.790 | 0.757 | 0.769 | Easier to explain, but weaker macro F1 |
| Naive Bayes | 0.914 | 0.798 | 0.750 | 0.768 | Fast baseline |
| Decision Tree | 0.828 | 0.746 | 0.677 | 0.695 | Lowest category performance |

Macro F1 is used here because the category dataset is imbalanced. It gives each category more equal weight, so small classes like `Sneaking`, `Obstruction`, and `Forced Action` still matter in the score. Even with a strong macro F1, the rarest categories should be presented cautiously because the model has fewer examples to learn their language patterns.

## Dataset

- **Primary binary dataset:** Krish Uppal balanced dataset, 2,356 examples, 50/50 split between dark-pattern and not-dark-pattern text.
- **Second-stage category dataset:** compatible category-labeled dark-pattern rows from Adarsh, Devitachi, and Mohit sources, deduplicated to 1,861 dark-pattern examples.
- **Category labels predicted by Model 2:** Forced Action, Misdirection, Obstruction, Scarcity, Sneaking, Social Proof, and Urgency.
- **Sources:** Kaggle datasets from Krish Uppal, Adarsh M09, Devitachi, Mohit Sharma, and Akash Nath; all are English-language e-commerce or deceptive-pattern text sources.
- **Category imbalance note:** after deduplication, `Scarcity` has 506 examples and `Social Proof` has 460, while `Sneaking` has 22 and `Obstruction` has 55. This is why the app describes Model 2 output as a likely type.

## Visualizations

Publication-quality figures are in [`reports/figures/`](reports/figures/) and embedded in the GitHub Page.

| Figure | Description |
| --- | --- |
| `two_stage_pipeline.png` | Diagram showing the binary detector feeding the second-stage category classifier |
| `two_model_summary_card.png` | Compact summary comparing the two trained models and their jobs |
| `model_comparison.png` | Grouped horizontal bar chart comparing all 5 models across 4 metrics |
| `category_model_comparison.png` | Grouped horizontal bar chart comparing the second-stage category classifiers |
| `confusion_matrix.png` | Heatmap for Logistic Regression on the 20% test set |
| `category_confusion_matrix.png` | Heatmap showing which dark-pattern types the second-stage model confuses |
| `category_per_class_f1.png` | Per-category F1 chart for the second-stage model |
| `category_distribution.png` | Dark-pattern category counts showing class imbalance within categories |
| `top_features.png` | Top 15 TF-IDF coefficients showing which words drive dark-pattern predictions |

## Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

The dependency versions in `requirements.txt` are pinned to compatible ranges so that fresh installs are more reproducible. The `numpy` upper bound avoids known SciPy/scikit-learn compatibility warnings on newer NumPy releases.

Train models and save metrics:

```bash
python scripts/train_models.py
```

This trains two models:

- `artifacts/best_binary_model.joblib`: the main dark-pattern vs not-dark-pattern detector.
- `artifacts/best_category_model.joblib`: a second-stage category classifier for text that already looks suspicious.

It also writes:

- `reports/model_metrics.csv`: binary model comparison metrics.
- `reports/category_model_metrics.csv`: category model comparison metrics.

The category model is trained only on dark-pattern examples with category labels. Its output should be described as a likely or possible type, not a definitive category, because some categories have much less training data than others.

To train only the binary model:

```bash
python scripts/train_models.py --skip-category-model
```

Train the expanded dataset experiment that also incorporates Devitachi and Akash Nath sources:

```bash
python scripts/train_models.py --dataset expanded
```

The expanded option writes `reports/model_metrics_expanded.csv` and `artifacts/best_binary_model_expanded.joblib`. The default app still uses the primary Logistic Regression model because it is balanced, easy to explain, and provides confidence scores.

Generate visualizations:

```bash
python scripts/generate_visualizations.py
```

Run tests:

```bash
python -m pytest
```

Run the full Google Colab workflow:

Open [`AI4ALL_20C_Dark_Pattern_Recognition.ipynb`](AI4ALL_20C_Dark_Pattern_Recognition.ipynb) in Colab. The notebook includes the complete code for loading datasets, training the binary model, training the second-stage category model, saving artifacts, generating figures, analyzing example text, and scanning example webpages.

Run the Streamlit demo:

```bash
streamlit run app/streamlit_app.py
```

Scan a live webpage with Playwright:

```bash
python -m playwright install chromium
python scripts/scan_page.py https://example.com
```

The scanner opens the page, waits briefly for dynamic content or popups, extracts visible body text, splits it into short snippets, and runs the trained text model over each snippet. This is a demo-friendly webpage scanner, not full visual UI understanding.

By default, the Streamlit app and Playwright scanner also apply two lightweight demo safeguards:

- They suppress simple price/discount snippets such as `sale price $10.00` or `Previous price: $99.00 47% off` unless the snippet includes pressure language like `hurry`, `limited time`, `ends soon`, or `only 2 left`.
- They suppress catalog-style product title/spec snippets such as model numbers, storage sizes, camera/lens specs, and labels like `US Stock` unless the snippet includes pressure language.
- They suppress low-context webpage fragments such as `Thanks Charan!`, `Why I bought this!`, `Deals bought: 284`, or short course/module titles unless the snippet includes pressure language.

These post-processing filters reduce obvious false positives in the live demo. A longer-term model improvement would be to add more normal sale/discount and product-catalog examples labeled as `Not Dark Pattern` and retrain.

## Demo Filters

All rule-based demo filters live in [`src/filters.py`](src/filters.py). The webpage scanner applies them in [`src/web_scanner.py`](src/web_scanner.py), and the terminal interface exposes filter switches in [`scripts/scan_page.py`](scripts/scan_page.py).

These filters do not retrain or change the machine learning model. They are post-processing safeguards that hide common false positives after the model predicts a snippet as suspicious.

| Filter | Function | Suppresses | Keeps |
| --- | --- | --- | --- |
| Pressure-language detector | `contains_pressure_language` | Nothing by itself; this is used by the other filters. | Words and phrases such as `hurry`, `last chance`, `limited time`, `ends soon`, `only 2 left`, `low stock`, and `act now`. |
| Simple price/discount filter | `is_simple_price_or_discount_snippet` | Plain sale or price labels such as `sale price $10.00`, `Previous price: $99.00 47% off`, `was $50`, or `now $25`. | Discount text that also includes pressure language, such as `Hurry, 47% off ends soon`. |
| Product title/spec filter | `is_low_context_product_snippet` | Catalog metadata such as model numbers, storage sizes, camera/lens specs, `US Stock`, `open box`, `near mint`, or `refurbished` when there is no pressure language. | Product snippets that include urgency or scarcity, such as `Only 2 left in stock for this tablet`. |
| Low-context webpage fragment filter | `is_low_context_web_snippet` | Snippets too thin to judge alone, such as `Thanks Charan!`, `Why I bought this!`, `Deals bought: 284`, short testimonials, bare counters, or short course/module titles. | Any of those snippets if they also include real pressure language like `only 2 left`, `deal ends`, or `act now`. |

In Streamlit, the `Scan webpage` tab has a `Hide vague short snippets` checkbox. That controls the low-context webpage fragment filter.

To show raw scanner output from the terminal without these safeguards:

```bash
python scripts/scan_page.py https://example.com --no-price-filter --no-product-filter --no-context-filter
```

Sentiment and context are important future considerations. Sentiment could help identify emotionally manipulative language such as guilt or fear, but it would not be enough by itself because dark patterns often depend on where the text appears, what button it is attached to, and what action the user is being pushed toward. For this version, we keep the model focused on text classification and treat richer UI context, screenshot features, DOM structure, and sentiment features as future work rather than adding extra training complexity now.

For a real production detector, Playwright text extraction would only be one layer. A stronger system could capture full-page screenshots and run OCR to recover text embedded in images, banners, or canvas elements. It could also use computer-vision features to analyze button color contrast, visual hierarchy, disabled-looking choices, hidden links, and misleading emphasis. To detect UX-flow dark patterns, the system would need to record multi-step interactions such as signup, checkout, cancellation, and unsubscribe flows, then compare how easy the preferred business action is versus the user-protective action.

To show the raw model behavior without this filter:

```bash
python scripts/scan_page.py https://www.ebay.com/deals --no-price-filter --no-product-filter --no-context-filter
```

## Error Analysis and External Validity

The model performs strongly on the curated held-out test set, but live webpage scans revealed a different kind of challenge: many snippets on real shopping pages are short, repetitive, or missing visual context. Examples such as `Previous price: $99.00 47% off`, camera specifications, bare counters, and short testimonials can look suspicious to a text-only model even when they are ordinary catalog content.

The current demo handles those cases with post-processing filters, but the better long-term solution is better validation data. A stronger next evaluation would keep the original held-out test set for model comparison, then add a separate external validation set from live websites that were not part of the Kaggle datasets. That external set should report precision, recall, F1, false positives, and false negatives by website type and dark-pattern category.

## Responsible AI

This model should be used as a review aid rather than a final judgment. Dark patterns can depend on full page layout, timing, checkout flow, and visual design; this version only analyzes text. A false positive from this model could expose a legitimate business to unwarranted scrutiny; a false negative could leave a consumer unprotected. Model predictions should be one input into a larger review process, not a standalone verdict.

Key limitations:
- Training data is English-language, Western-focused e-commerce; the model may not generalize to other languages, regions, or domains
- Some categories have very few examples after deduplication, especially `Sneaking` with 22 examples and `Obstruction` with 55, so likely-type predictions for rare categories are more likely to be misclassified
- Text-only features cannot capture visual manipulation, flow-based dark patterns, or context-dependent intent
- The Playwright scanner can often capture popup text when it appears in the DOM, but it may miss cross-origin iframes, image/canvas text, closed shadow DOM content, or websites that block automation
- The Streamlit/scanner price-discount filter is a rule-based demo safeguard, not a substitute for better labeled training data
- The product-title/spec filter is also a rule-based demo safeguard; it can reduce catalog false positives but should not be treated as proof that a page has no dark patterns
- Sentiment alone cannot determine whether a design is deceptive; the model would need surrounding UI context and human review for borderline cases
- Screenshot + OCR could recover image-based text, but OCR can misread stylized fonts, low-contrast text, animations, and responsive layouts
- Color and layout analysis would require computer-vision or DOM/CSS features, not just TF-IDF text features
- Production UX-flow detection would require scripted user journeys and event logs to measure friction, repeated interruptions, hidden fees, cancellation difficulty, and forced-choice paths

Bias mitigation steps used or planned:
- Use stratified train/test splits so the binary model sees similar dark-pattern and non-dark-pattern proportions in evaluation
- Report macro F1 for category classification so rare categories influence the score instead of being hidden by common classes
- Keep category predictions labeled as likely or possible types, with human review for low-data categories and high-impact decisions
- Add more negative examples from normal sale pages, product catalogs, and benign marketing copy before treating scanner output as evidence
- Build an external validation set from real webpages across multiple e-commerce domains, then audit false positives and false negatives by category, source, and website type
- Expand beyond English e-commerce only after collecting multilingual and non-commerce examples with consistent labels

## Next Steps

- **By August 2026:** collect and label at least 300 real webpage snippets from live e-commerce pages, including normal discounts and product catalog text, then retrain to reduce demo false positives.
- **By September 2026:** create a 100-example external validation set from websites not represented in the Kaggle datasets and report precision, recall, F1, and false-positive examples separately from the original test set.
- **By October 2026:** add at least 50 examples for each rare category, especially `Sneaking` and `Obstruction`, then rerun the category model and compare per-class F1 before and after balancing.
- **By November 2026:** prototype screenshot + OCR extraction on 50 product or checkout pages to test whether image-based banners add missing text evidence.
- **By December 2026:** script three UX-flow audits, such as checkout, cancellation, and cookie-consent journeys, and measure friction using step counts and repeated interruptions.
- **After stronger validation:** consider a browser-extension prototype that clearly labels predictions as review leads, not proof of deceptive intent.

## Troubleshooting

- **Windows permission error during `pip install`**

	This happens when `pip` tries to modify files in a system-wide Python installation without sufficient permissions. Recommended fixes:
	
	- Install into a virtual environment (recommended):
	
	```powershell
		python -m venv .venv
	.\.venv\Scripts\Activate.ps1
		pip install -r requirements.txt
	```
	
	- Install for the current user only:
	
		```powershell
		pip install --user -r requirements.txt
	```
	
- Run the terminal as Administrator (not recommended for everyday use) or adjust folder permissions if appropriate.
	
	- Check antivirus or file-locking processes that may block `f2py.exe` or other scripts from being modified.
	
These steps should resolve the permission error when installing dependencies on Windows.

## Citations

1. Mathur, A., Acar, G., Friedman, M. J., Lucherini, E., Mayer, J., Chetty, M., & Narayanan, A. (2019). Dark Patterns at Scale: Findings from a Crawl of 11K Shopping Websites. *Proceedings of the ACM on Human-Computer Interaction*, 3(CSCW), Article 81. https://doi.org/10.1145/3359183

2. Brignull, H. (2023). Deceptive Patterns. *deceptive.design*. https://www.deceptive.design/

3. Gray, C. M., Kou, Y., Battles, B., Hoggatt, J., & Toombs, A. L. (2018). The Dark (Patterns) Side of UX Design. *Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems*, Paper 534. https://doi.org/10.1145/3173574.3174108

4. Federal Trade Commission. (2022). Bringing Dark Patterns to Light. *FTC Staff Report*. https://www.ftc.gov/reports/bringing-dark-patterns-light

5. European Parliament. (2022). Digital Services Act. *Regulation (EU) 2022/2065*. https://eur-lex.europa.eu/eli/reg/2022/2065/oj

6. Luguri, J. & Strahilevitz, L. J. (2021). Shining a Light on Dark Patterns. *Journal of Legal Analysis*, 13(1), 43-109. https://doi.org/10.1093/jla/laaa006

7. Narayanan, A., Mathur, A., Chetty, M., & Kshirsagar, M. (2020). Dark Patterns: Past, Present, and Future. *Queue*, 18(2), 67-92. https://doi.org/10.1145/3400899.3400901

8. Ramteke, A., Tembhurne, S., Sonawane, G., & Bhimanpallewar, R. N. (2024). Detecting Deceptive Dark Patterns in E-commerce Platforms. *arXiv preprint* arXiv:2406.01608. https://arxiv.org/abs/2406.01608

9. Koh, W. C., & Seah, V. (2023). Unintended consumption: The effects of four e-commerce dark patterns. *Cleaner and Responsible Consumption*, 11, 100145. https://www.sciencedirect.com/science/article/pii/S2666784323000463

Data sources: Krish Uppal dark-patterns dataset, Adarsh M09 dark-pattern dataset, Devitachi dark-pattern dataset, Mohit Sharma dark-patterns on e-commerce platforms dataset, and Akash Nath deceptive-patterns classification data from Kaggle.

## Deliverables

### Student Norms & Conflict Resolution Agreement
https://docs.google.com/document/d/1xZe-Gzp7QJp7ArmpTayKea5mC11Ep3e2UeTyxgF7_Ao/edit?usp=sharing

### Datasets
https://docs.google.com/spreadsheets/d/1qfgSGvv1ksOrKMf8d0CXDxduyCVPPC2qsw52Xyswe6I/edit?usp=sharing

### Proposal
https://docs.google.com/document/d/1yw7u5fxK5i6h2kTN7Yf3jU65YGKgjuPTuyHD_1Dgsgk/edit?usp=sharing

### Proposal Presentation
https://docs.google.com/presentation/d/1G-T5sNNgWVGZw4izbyFwNgn2sTScSlYtnYZWRO04eN4/edit?usp=sharing

### Final Presentation
https://docs.google.com/presentation/d/1VXO8nk5VA0iO1YYWCzUIlaQ-IDXtOgfFWqVQdmMusXc/edit?usp=sharing

### Poster
https://docs.google.com/presentation/d/1BkSWZKBQGaChyAHT9umTdrrqzU6iINmbU-dE4dtqUC0/edit?usp=sharing
