# Stack decisions

1. **Python, FastAPI, and SQLAlchemy 2.0; SQLite locally and Postgres in production through `DATABASE_URL`.** This keeps the local test setup light while allowing the same application code to use the production database.
2. **No Docker or Redis; run the background worker inside the app and keep job state in the database so work can resume.** This avoids separate infrastructure while making job state durable across app restarts.
3. **Use Argon2 and JWT for user authentication, plus API keys for the CLI.** This gives the browser app and command-line client authentication methods suited to how each is used.
4. **Serve plain HTML, CSS, and JavaScript from `backend/static` through FastAPI.** This avoids a separate frontend build and server for this small test lab.
5. **Encrypt each user's own Gnani key with Fernet using `SECRET_KEY` from the environment, and target free hosting.** Per-user encrypted storage keeps keys separate and avoids storing them in plaintext; the single-app design fits the stated low-cost deployment target.

## Scoring normalization and trade-offs

The default scorer uses lenient normalization to reduce script and formatting variation. `normalize_strict` remains available when those distinctions need to be inspected. These choices can hide real transcript differences:

- **NFC composition** treats canonically equivalent Unicode sequences as the same text. It hides encoding-level differences, though not normally a spoken-word difference.
- **Removing nukta** merges letters such as `ज` / `ज़`, `क` / `क़`, `ख` / `ख़`, `ग` / `ग़`, `फ` / `फ़`, `ड` / `ड़`, and `ढ` / `ढ़`; those can be different words or pronunciations.
- **Mapping chandrabindu to anusvara** merges nasalization spellings that may distinguish pronunciation or meaning.
- **Lowercasing** ignores Roman-script case, including differences in acronyms and proper names.
- **Replacing punctuation with spaces** ignores punctuation differences, including danda, and can hide sentence-boundary, question, or emphasis differences.
- **Collapsing whitespace** ignores repeated and leading/trailing spaces.
- **Treating texts as spacing-only matches after removing every space** makes WER zero for different word segmentation, such as `ईएमआई` and `ई एम आई`; this can hide a real word-boundary or tokenization error.
- **Accepted references** allow listed alternatives, including Roman/Devanagari forms. An incorrect or overly broad alternative can make a genuinely wrong transcript score as a match.

## Audio condition cache

Raw synthesized audio stays keyed only by its TTS inputs so different test conditions can reuse it without duplicating TTS calls. A separate degraded-audio key adds the condition name, all condition parameters, and random seed, so changed degradation settings cannot reuse stale processed audio.

Packet loss chooses a seeded, fixed rounded count of 20 ms chunks without replacement instead of independently flipping a 10% chance for each chunk. This keeps the realized loss fraction close to the requested rate while preserving random placement.

## Entity scoring limits and spellings to review

Amount parsing supports the listed Hindi number vocabulary, Devanagari digits, grouped numerals, and the Latin scale forms `sau`, `hazaar`, `hazar`, `lakh`, `lakhs`, `lac`, `lacs`, `crore`, and `crores`. Romanized Hindi number words such as `pandra` are not guessed.

Date matching is approximate: it recognizes day words/numerals near a date marker or month name, and common Hindi and English month names. UPI matching is approximate: Devanagari names are transliterated heuristically, with `ओकेएसबीआई` explicitly mapped to `oksbi`. IFSC checking is approximate and validates only the expected character shape; it does not confirm a bank or branch.

Please review these spelling choices by hand if the suite uses them: `पन्द्रह` / `पंद्रह`, `बाईस` / `बाइस`, `चौबीस` / `चौबिस`, `इक्यानवे` / `इकानवे`, `पंचानवे` / `पचानवे`; and the Latin scale alternatives `hazar`, `hazaar`, `lac`, and `lakh`. Month spellings include `फ़रवरी` / `फरवरी`, `सितंबर` / `सितम्बर`, `नवंबर` / `नवम्बर`, and `दिसंबर` / `दिसम्बर`.
