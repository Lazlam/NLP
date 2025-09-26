# NLP
# Text Reconstruction & Paraphrasing Pipelines

Αυτό το αποθετήριο περιέχει πειραματική υλοποίηση σε Python για:
- Ανακατασκευή/βελτίωση προτάσεων μέσω κανόνων (heuristics).
- Παραφράσεις με χρήση **Hugging Face Transformers** (BART).
- Ορθογραφικό/γραμματικό έλεγχο με **LanguageTool**.
- Αξιολόγηση ομοιότητας μέσω **Sentence-BERT embeddings**.
- Πειράματα **mask-filling** με ελληνικά και πολυγλωσσικά BERT μοντέλα.

---

## Περιγραφή Έργου

Η υλοποίηση οργανώνεται γύρω από διαφορετικές **pipelines** που επεξεργάζονται κείμενο:  
- `custom` → Χειροποίητοι κανόνες (regex-based rewrites).  
- `transformer` → Παραφράσεις μέσω BART.  
- `languagetool` → Διορθώσεις με LanguageTool API.  
- `paraphrase_lt` → Συνδυασμός LanguageTool + Transformer.  

Στόχος είναι να συγκρίνουμε τις παραγόμενες εκδοχές προτάσεων μετρώντας την **συνημιτονοειδή ομοιότητα (cosine similarity)** μεταξύ αρχικών και ανακατασκευασμένων κειμένων.

Επιπλέον, περιλαμβάνεται task που επιλύει [MASK] tokens σε ελληνικά νομικά αποσπάσματα με χρήση BERT.

---

## Προαπαιτούμενα

- Python 3.9+
- Εικονικό περιβάλλον (**venv** ή **conda**)
- Εγκατάσταση βιβλιοθηκών:

bash:
pip install -r requirements.txt

