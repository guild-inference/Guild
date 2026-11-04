"""Built the request set for the StartLux-Decision gate, requests.jsonl in this directory: one /v1/systemone
request a line ({"id", "state", "questions", "group", ...}).  (The long requests were then cut to their measured
sizes and renamed: long-25k, long-31k - Guild's dense attention for this architecture holds 32,768 cells - and
long-json-20k.)

  * the 231 public JevBench items (Intern-Decision benchmarks/accuracy-v1/jevbench, MIT), one question each;
  * the model card's three-question example and edge cases of the rendering (wide choice, list criteria, hidden
    bare ids, legend scores, noul criteria, the "bool" alias, JSON states with every value kind, empty state,
    a single option, non-English text);
  * long inputs (8K to 32K tokens): slices of /home/ubuntu/lanes/prompts/corpus.txt (docs and C++) and a long JSON
    state, with several questions each, so the shared-prefix path runs.
"""
import json
import random
from pathlib import Path

J = Path("/home/ubuntu/work/startlux/jevbench")
CORPUS = Path("/home/ubuntu/lanes/prompts/corpus.txt").read_text(encoding="utf-8")
out = []

for split in ("easy", "original", "hard"):
    for line in (J / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out.append({"id": r["id"], "group": f"jevbench-{split}", "state": r["state"], "questions": {"q": r["question"]},
                    "expected": r["expected"]})

card = {
    "id": "card-example", "group": "edge",
    "state": {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."},
    "questions": {
        "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Payments, refunds and invoices", "shipping": "Delivery and tracking",
                              "technical": "App, login and account problems"}},
        "urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"},
        "severity": {"type": "score", "instructions": "How severe is the impact?",
                     "criteria": ["cosmetic", "annoying", "blocks the customer"]},
    },
}
out.append(card)

countries = ["Argentina", "Australia", "Austria", "Belgium", "Brazil", "Canada", "Chile", "China", "Denmark", "Egypt",
             "Finland", "France", "Germany", "Greece", "India", "Indonesia", "Ireland", "Italy", "Japan", "Kenya",
             "Mexico", "Netherlands", "Norway", "Peru", "Poland", "Portugal", "Spain", "Sweden", "Turkey", "Vietnam"]
out.append({"id": "edge-wide-30", "group": "edge",
            "state": "The parcel was posted in Lyon, crossed one border by truck and was delivered in Turin.",
            "questions": {"origin": {"type": "choice", "instructions": "In which country was the parcel posted?",
                                     "criteria": {c.lower(): None for c in countries}},
                          "dest": {"type": "choice", "instructions": "In which country was it delivered?",
                                   "criteria": {c.lower(): f"The country {c}" for c in countries}}}})
out.append({"id": "edge-list-criteria", "group": "edge", "state": "The sky was clear all day and nobody needed an umbrella.",
            "questions": {"w": {"type": "choice", "instructions": "What was the weather?",
                                "criteria": ["sunny", "rainy", "snowy", "foggy"]},
                          "num": {"type": "choice", "instructions": "How many umbrellas were needed?",
                                  "criteria": [0, 1, 2, 3.5, True, None]}}})
out.append({"id": "edge-hidden-ids", "group": "edge", "state": "Water boils at 100 degrees Celsius at sea level.",
            "questions": {"a": {"type": "choice", "instructions": "At what temperature does water boil at sea level?",
                                "criteria": {"A": "50 degrees", "B": "100 degrees", "C": "150 degrees", "D": "0 degrees"}},
                          "n": {"type": "choice", "instructions": "Pick the boiling point.",
                                "criteria": {"1)": "90 C", "2)": "100 C", "option_3": "110 C"}}}})
out.append({"id": "edge-score-legend", "group": "edge", "state": "The review says: decent food, slow service, would maybe return.",
            "questions": {"s": {"type": "score", "instructions": "How positive is the review?",
                                "criteria": {"3": "very positive", "1": "negative", "2": "mixed", "0": "hostile"}},
                          "t": {"type": "score", "question": "  How likely is a return visit?  ",
                                "options": {"low": "unlikely", "mid": "possible", "high": "certain"}}}})
out.append({"id": "edge-noul-criteria", "group": "edge", "state": "Invoice 77 was paid on 3 March by bank transfer.",
            "questions": {"paid": {"type": "noul", "instructions": "Has invoice 77 been paid?",
                                   "criteria": {"true": "The invoice is settled", "false": "Payment is still open"}},
                          "card": {"type": "bool", "instructions": "Was it paid by card?"},
                          "noinst": {"type": "noul", "criteria": {"true": "Yes", "false": "No"}}}})
records = [{"sku": f"A-{i:03d}", "qty": i % 7, "price": round(1.25 * i + 0.5, 2), "ok": i % 3 != 0,
            "note": None if i % 4 else "fragile – étagère", "tags": ["x", "y"][: i % 3]} for i in range(10)]
out.append({"id": "edge-json-state", "group": "edge",
            "state": {"order": "#4411", "total": 12.0, "items": records, "nested": {"deep": [1, 2.5, -3, 1e-05, 1e16, 123456789012345678901234567890]},
                      "quote": "she said \"no\"\n\ttab\\slash", "unicode": "你好 \U0001F600", "empty": [], "none": None},
            "questions": {"fragile": {"type": "noul", "instructions": "Does any item carry a fragile note?"},
                          "most": {"type": "choice", "instructions": "Which SKU has the largest quantity?",
                                   "criteria": {"A-006": None, "A-005": None, "A-003": None}}}})
out.append({"id": "edge-empty-state", "group": "edge", "state": "",
            "questions": {"q": {"type": "noul", "instructions": "Is there any evidence?"},
                          "only": {"type": "choice", "instructions": "Pick one.", "criteria": {"solo": "the only option"}}}})
out.append({"id": "edge-chinese", "group": "edge", "state": "客户说他两次被扣款，要求退款。",
            "questions": {"team": {"type": "choice", "instructions": "哪个团队应该处理？",
                                   "criteria": {"财务": "付款和退款", "物流": "配送", "技术": "应用问题"}}}})

# long inputs: slices of the corpus (cut at a line end), questions about them
def slice_chars(start, n):
    s = CORPUS[start:start + n]
    return s[: s.rfind("\n") + 1]

long_qs = {
    "lang": {"type": "choice", "instructions": "Which programming language does most of the code in the evidence use?",
             "criteria": {"cpp": "C or C++", "python": "Python", "rust": "Rust", "go": "Go", "none": "There is no code"}},
    "cuda": {"type": "noul", "instructions": "Does the evidence mention CUDA or GPU kernels?"},
    "tech": {"type": "score", "instructions": "How technical is the evidence?",
             "criteria": ["not technical", "somewhat technical", "very technical"]},
}
out.append({"id": "long-8k", "group": "long", "state": slice_chars(0, 34000), "questions": long_qs})
out.append({"id": "long-16k", "group": "long", "state": slice_chars(200000, 68000),
            "questions": {k: long_qs[k] for k in ("lang", "cuda")}})
out.append({"id": "long-32k", "group": "long", "state": slice_chars(600000, 136000), "questions": {"lang": long_qs["lang"]}})
rng = random.Random(7)
rows = [{"order_id": 10000 + i, "customer": f"customer-{rng.randrange(500):03d}", "status": rng.choice(["paid", "open", "refunded", "late"]),
         "amount": round(rng.uniform(3, 900), 2), "items": rng.randrange(1, 9), "express": rng.random() < 0.2} for i in range(400)]
out.append({"id": "long-json-10k", "group": "long", "state": {"source": "orders export", "orders": rows},
            "questions": {"late": {"type": "noul", "instructions": "Is at least one order marked late?"},
                          "common": {"type": "choice", "instructions": "Which status is the most common?",
                                     "criteria": {"paid": None, "open": None, "refunded": None, "late": None}}}})

with open("/home/ubuntu/work/startlux/ref/requests.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(out), "requests,", sum(len(r["questions"]) for r in out), "questions")
