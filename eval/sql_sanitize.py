import html
import re

# Some models (notably gpt-5.6-sol / gpt-5.6-luna via the Responses API) emit SQL with
# HTML entities, e.g. `x &lt;&gt; 'NM'` instead of `x <> 'NM'`. The escaping is in the
# raw model output (prompts contain none), and MySQL rejects it with a 1064 syntax error,
# so an otherwise-correct query scores 0. Unescaping is lossless for real SQL: a literal
# "&lt;" inside a string constant does not occur anywhere in the BEAVER gold queries.
_ABSTAIN_RE = re.compile(r"^\s*none\s*;?\s*$", re.IGNORECASE)


def sanitize_sql(sql):
    """Normalize an extracted SQL prediction before it is written/executed.

    - None / the prompt's abstention token ("None") -> "" so it is scored as an empty
      prediction instead of a misleading MySQL syntax error.
    - HTML entities are unescaped (repeatedly, in case of double escaping).
    """
    if sql is None:
        return ""
    if _ABSTAIN_RE.match(sql):
        return ""
    prev = None
    while prev != sql:
        prev, sql = sql, html.unescape(sql)
    return sql.strip()


def is_abstention(sql):
    """The model answered the prompt's "None" (could not determine the SQL)."""
    return sql is not None and bool(_ABSTAIN_RE.match(sql))


def has_html_entities(sql):
    """sanitize_sql would unescape something in this prediction."""
    return bool(sql) and not is_abstention(sql) and html.unescape(sql) != sql
