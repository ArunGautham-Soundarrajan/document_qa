from textacy import preprocessing
from functools import partial


def normalizing_text(text: str) -> str:
    preproc = preprocessing.make_pipeline(
        preprocessing.normalize.hyphenated_words,
        preprocessing.normalize.quotation_marks,
        partial(preprocessing.normalize.repeating_chars, chars=".", maxn=2),
        partial(preprocessing.normalize.repeating_chars, chars=",", maxn=2),
        partial(preprocessing.normalize.repeating_chars, chars=" ", maxn=2),
        preprocessing.normalize.unicode,
        preprocessing.normalize.whitespace,
        preprocessing.remove.html_tags,
    )

    try:
        return preproc(text)
    except Exception as e:
        raise e
